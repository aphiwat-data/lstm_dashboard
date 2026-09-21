#!/usr/bin/env bash
# =============================================================================
# Deploys the daily inference-refresh automation: ECR repo -> container
# image -> Lambda function -> EventBridge Scheduler (cron: once daily at
# 01:00 UTC = 08:00 Thailand time, chosen to land safely after COMEX Gold's
# daily 5:00pm ET session halt / 1:30pm ET settlement -- see
# automation/README.md and lambda_function.py's own docstring for why).
#
# WHO RUNS THIS AND WHERE: the student's Mac (it already has AWS CLI-
# compatible credentials configured for IAM user `Auto` -- see main
# README.md section 10, "Local Mac setup" -- and Docker Desktop is the only
# new prerequisite) or the SageMaker Notebook Instance's terminal. NOT
# meant to run from anywhere without real AWS network access and real
# credentials for the project's AWS account (see main README.md section 4
# for which account that is -- deliberately not hardcoded here since this
# file is public).
#
# This script is idempotent -- safe to re-run after fixing an error, it
# creates-or-updates rather than failing on "already exists".
#
# WHAT THIS COSTS: Lambda's free tier is 1M requests + 400,000 GB-seconds
# of compute per month. This function runs once/day (~30/month) for a few
# hundred ms each at 512MB -- several orders of magnitude under the free
# tier, even more trivially so than an hourly schedule would have been.
# ECR storage for one small image is a few cents/month at most. Expected
# real cost: effectively $0, same conclusion as the rest of this project's
# AWS usage (see main README section on cost -- S3 + occasional compute,
# nothing left running idle).
# =============================================================================
set -euo pipefail

# --- Configuration -----------------------------------------------------------
AWS_REGION="ap-southeast-2"
S3_BUCKET="gold-lstm-forecast"
FUNCTION_NAME="gold-lstm-forecast-daily-refresh"
ECR_REPO_NAME="gold-lstm-forecast-daily-refresh"
LAMBDA_ROLE_NAME="gold-lstm-forecast-lambda-exec-role"
SCHEDULER_ROLE_NAME="gold-lstm-forecast-scheduler-role"
SCHEDULE_NAME="gold-lstm-forecast-daily"
# 01:00 UTC = 08:00 ICT (Thailand does not observe DST, so this offset is
# constant year-round even though the ET->ICT gap it's timed against shifts
# by an hour across US daylight saving -- the multi-hour buffer absorbs that).
SCHEDULE_EXPRESSION="cron(0 1 * * ? *)"
LAMBDA_MEMORY_MB=512
LAMBDA_TIMEOUT_S=60

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GRANT_PERMISSIONS=false
for arg in "$@"; do
  case "$arg" in
    --grant-permissions) GRANT_PERMISSIONS=true ;;
  esac
done

echo "============================================================"
echo " Gold LSTM Forecast -- Daily Automation Deployment"
echo "============================================================"
echo "Region            : ${AWS_REGION}"
echo "S3 bucket          : ${S3_BUCKET}"
echo "Lambda function   : ${FUNCTION_NAME}"
echo "Schedule           : ${SCHEDULE_EXPRESSION}"
echo "Grant new IAM perms to Auto : ${GRANT_PERMISSIONS} (pass --grant-permissions to enable)"
echo "============================================================"

# --- Prerequisite checks -----------------------------------------------------
command -v aws >/dev/null 2>&1 || { echo "ERROR: AWS CLI not found. Install it first (https://aws.amazon.com/cli/)."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "ERROR: Docker not found. Install Docker Desktop first."; exit 1; }
docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon not running. Start Docker Desktop and retry."; exit 1; }

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text --region "${AWS_REGION}")"
CALLER_ARN="$(aws sts get-caller-identity --query Arn --output text --region "${AWS_REGION}")"
echo "AWS identity in use : ${CALLER_ARN} (account ${ACCOUNT_ID})"
# Optional guard: export EXPECTED_ACCOUNT_ID locally (never committed) if you
# want a sanity check against deploying to the wrong AWS account.
if [[ -n "${EXPECTED_ACCOUNT_ID:-}" && "${ACCOUNT_ID}" != "${EXPECTED_ACCOUNT_ID}" ]]; then
  echo "WARNING: this doesn't match EXPECTED_ACCOUNT_ID (${EXPECTED_ACCOUNT_ID})."
  echo "Double-check ~/.aws/credentials before continuing."
fi

read -r -p "Continue with deployment? [y/N] " CONFIRM
if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
  echo "Aborted."
  exit 0
fi

# --- Step 0 (opt-in only): grant Auto the new permissions it needs ----------
# Auto currently has AmazonS3FullAccess + AmazonSageMakerFullAccess +
# IAMFullAccess (see main README section 4) -- none of which cover Lambda,
# ECR, or EventBridge Scheduler. IAMFullAccess is what lets Auto attach
# these to itself, the same bootstrapping pattern already used once before
# for the access-key step. This is left OFF by default (only runs with
# --grant-permissions) because it changes the AWS account's permission
# surface -- a deliberate, visible step rather than something this script
# does silently on your behalf.
if [[ "${GRANT_PERMISSIONS}" == "true" ]]; then
  echo ""
  echo "--- Attaching managed policies to IAM user 'Auto' ---"
  for POLICY_ARN in \
    "arn:aws:iam::aws:policy/AWSLambda_FullAccess" \
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess" \
    "arn:aws:iam::aws:policy/AmazonEventBridgeSchedulerFullAccess"
  do
    echo "  Attaching ${POLICY_ARN} ..."
    aws iam attach-user-policy --user-name Auto --policy-arn "${POLICY_ARN}" --region "${AWS_REGION}"
  done
  echo "Done. (These are broad managed policies, matching the existing dev-speed"
  echo "trade-off already made for Auto's other permissions -- see main README's"
  echo "IAM least-privilege discussion. The Lambda's OWN execution role, created"
  echo "below, is scoped tightly instead -- that's the role that matters, since"
  echo "it's what runs unattended.)"
else
  echo ""
  echo "Skipping IAM permission grant for Auto (--grant-permissions not passed)."
  echo "If the next steps fail with AccessDenied, either re-run with"
  echo "--grant-permissions, or attach these three managed policies to Auto"
  echo "yourself first: AWSLambda_FullAccess, AmazonEC2ContainerRegistryFullAccess,"
  echo "AmazonEventBridgeSchedulerFullAccess."
fi

# --- Step 1: ECR repo ---------------------------------------------------------
echo ""
echo "--- ECR repository ---"
if aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "  Repo ${ECR_REPO_NAME} already exists."
else
  aws ecr create-repository --repository-name "${ECR_REPO_NAME}" --region "${AWS_REGION}" >/dev/null
  echo "  Created repo ${ECR_REPO_NAME}."
fi
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

# --- Step 2: build & push image ----------------------------------------------
echo ""
echo "--- Docker build & push ---"
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
docker build --platform linux/amd64 -t "${ECR_REPO_NAME}:latest" "${SCRIPT_DIR}"
docker tag "${ECR_REPO_NAME}:latest" "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"
IMAGE_DIGEST="$(aws ecr describe-images --repository-name "${ECR_REPO_NAME}" --region "${AWS_REGION}" \
  --query 'sort_by(imageDetails,& imagePushedAt)[-1].imageDigest' --output text)"
echo "  Pushed ${ECR_URI}:latest (digest ${IMAGE_DIGEST})"

# --- Step 3: Lambda execution role (least-privilege -- see iam_*.json) ------
echo ""
echo "--- Lambda execution role ---"
if aws iam get-role --role-name "${LAMBDA_ROLE_NAME}" >/dev/null 2>&1; then
  echo "  Role ${LAMBDA_ROLE_NAME} already exists -- updating its policy."
  aws iam update-assume-role-policy --role-name "${LAMBDA_ROLE_NAME}" \
    --policy-document "file://${SCRIPT_DIR}/iam_trust_policy.json"
else
  aws iam create-role --role-name "${LAMBDA_ROLE_NAME}" \
    --assume-role-policy-document "file://${SCRIPT_DIR}/iam_trust_policy.json" >/dev/null
  echo "  Created role ${LAMBDA_ROLE_NAME}."
fi
aws iam put-role-policy --role-name "${LAMBDA_ROLE_NAME}" \
  --policy-name "gold-lstm-forecast-lambda-least-privilege" \
  --policy-document "file://${SCRIPT_DIR}/iam_permissions_policy.json"
LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"
echo "  Policy attached. Role ARN: ${LAMBDA_ROLE_ARN}"
echo "  (Scoped to exactly 3 S3 read paths + 2 S3 read/write paths + its own"
echo "  CloudWatch log group -- no SageMaker, no other S3 prefixes, no delete"
echo "  permission anywhere. Contrast with Auto's own AmazonS3FullAccess.)"

echo "  Waiting a few seconds for IAM role propagation..."
sleep 10

# --- Step 4: Lambda function --------------------------------------------------
echo ""
echo "--- Lambda function ---"
if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "  Function exists -- updating code + config."
  aws lambda update-function-code --function-name "${FUNCTION_NAME}" \
    --image-uri "${ECR_URI}:latest" --region "${AWS_REGION}" >/dev/null
  aws lambda wait function-updated --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}"
  aws lambda update-function-configuration --function-name "${FUNCTION_NAME}" \
    --memory-size "${LAMBDA_MEMORY_MB}" --timeout "${LAMBDA_TIMEOUT_S}" \
    --environment "Variables={S3_BUCKET=${S3_BUCKET}}" \
    --region "${AWS_REGION}" >/dev/null
else
  aws lambda create-function --function-name "${FUNCTION_NAME}" \
    --package-type Image --code "ImageUri=${ECR_URI}:latest" \
    --role "${LAMBDA_ROLE_ARN}" \
    --memory-size "${LAMBDA_MEMORY_MB}" --timeout "${LAMBDA_TIMEOUT_S}" \
    --environment "Variables={S3_BUCKET=${S3_BUCKET}}" \
    --region "${AWS_REGION}" >/dev/null
  echo "  Created function ${FUNCTION_NAME}."
fi
aws lambda wait function-active --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}"
LAMBDA_ARN="arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"
echo "  Function ARN: ${LAMBDA_ARN}"

echo ""
echo "--- Smoke-testing the function once, synchronously ---"
aws lambda invoke --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}" \
  --cli-read-timeout 90 /tmp/gold-lstm-invoke-result.json >/tmp/gold-lstm-invoke-meta.json
cat /tmp/gold-lstm-invoke-result.json
echo ""
echo "  (If this shows a Python traceback instead of JSON, check"
echo "  'aws logs tail /aws/lambda/${FUNCTION_NAME} --region ${AWS_REGION}'"
echo "  before proceeding to the schedule -- no point scheduling a broken function.)"

# --- Step 5: EventBridge Scheduler role --------------------------------------
echo ""
echo "--- EventBridge Scheduler role ---"
if aws iam get-role --role-name "${SCHEDULER_ROLE_NAME}" >/dev/null 2>&1; then
  echo "  Role ${SCHEDULER_ROLE_NAME} already exists."
else
  aws iam create-role --role-name "${SCHEDULER_ROLE_NAME}" \
    --assume-role-policy-document "file://${SCRIPT_DIR}/scheduler_trust_policy.json" >/dev/null
  echo "  Created role ${SCHEDULER_ROLE_NAME}."
fi
aws iam put-role-policy --role-name "${SCHEDULER_ROLE_NAME}" \
  --policy-name "invoke-daily-refresh-lambda-only" \
  --policy-document "file://${SCRIPT_DIR}/scheduler_permissions_policy.json"
SCHEDULER_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${SCHEDULER_ROLE_NAME}"
sleep 5

# --- Step 6: the schedule itself ---------------------------------------------
echo ""
echo "--- EventBridge Scheduler schedule ---"
if aws scheduler get-schedule --name "${SCHEDULE_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "  Schedule exists -- updating."
  aws scheduler update-schedule --name "${SCHEDULE_NAME}" --region "${AWS_REGION}" \
    --schedule-expression "${SCHEDULE_EXPRESSION}" \
    --flexible-time-window '{"Mode":"OFF"}' \
    --target "{\"Arn\":\"${LAMBDA_ARN}\",\"RoleArn\":\"${SCHEDULER_ROLE_ARN}\"}" >/dev/null
else
  aws scheduler create-schedule --name "${SCHEDULE_NAME}" --region "${AWS_REGION}" \
    --schedule-expression "${SCHEDULE_EXPRESSION}" \
    --flexible-time-window '{"Mode":"OFF"}' \
    --target "{\"Arn\":\"${LAMBDA_ARN}\",\"RoleArn\":\"${SCHEDULER_ROLE_ARN}\"}" >/dev/null
  echo "  Created schedule ${SCHEDULE_NAME} (${SCHEDULE_EXPRESSION})."
fi

echo ""
echo "============================================================"
echo " Done."
echo "============================================================"
echo "Verify it's really running (give it a day, or invoke manually):"
echo "  aws lambda invoke --function-name ${FUNCTION_NAME} --region ${AWS_REGION} /tmp/out.json && cat /tmp/out.json"
echo "  aws logs tail /aws/lambda/${FUNCTION_NAME} --region ${AWS_REGION} --follow"
echo "  aws s3 cp s3://${S3_BUCKET}/gold/xauusd_daily/predictions/latest_forecast.json -"
echo ""
echo "To stop it later without deleting anything (e.g. before the account owner"
echo "worries about it running unattended after the presentation):"
echo "  aws scheduler update-schedule --name ${SCHEDULE_NAME} --region ${AWS_REGION} --state DISABLED \\"
echo "    --schedule-expression \"${SCHEDULE_EXPRESSION}\" --flexible-time-window '{\"Mode\":\"OFF\"}' \\"
echo "    --target '{\"Arn\":\"${LAMBDA_ARN}\",\"RoleArn\":\"${SCHEDULER_ROLE_ARN}\"}'"
echo ""
echo "To tear everything down completely, see automation/README.md \"Tearing it down\"."
