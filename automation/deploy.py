"""Idempotent deployment of the daily pipeline Lambda (zip + AWS SDK for pandas layer, no Docker).

  python deploy.py plan        # show what would be created
  python deploy.py apply       # create or update everything (safe to repeat)
  python deploy.py teardown    # delete what apply created (not the S3 data)

Creates: Lambda role (least-privilege inline policy), the function, the EventBridge Scheduler schedule (daily 01:00 UTC = 08:00 Thailand) with its own
invoke-only role, an SNS topic with two CloudWatch alarms, and uploads automation/../daily/serving_artifacts/serving_model.npz to s3://<bucket>/model/serving/.
"""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent
DAILY = HERE.parent / "daily"
REGION, BUCKET = "ap-southeast-2", "gold-lstm-forecast"
FN = "gold-lstm-forecast-daily-refresh"
ROLE, SCHED_ROLE, SCHEDULE, TOPIC = f"{FN}-role", f"{FN}-scheduler-role", FN, "gold-lstm-forecast-alerts"
LAYER = "arn:aws:lambda:ap-southeast-2:336392948345:layer:AWSSDKPandas-Python312:32"  # AWS-managed: pandas, numpy, pyarrow
FILES = {"lambda_function.py": HERE / "lambda_function.py", "serving.py": DAILY / "serving.py", "build_daily_dataset.py": DAILY / "build_daily_dataset.py"}
MODEL_SRC, MODEL_KEY = DAILY / "serving_artifacts" / "serving_model.npz", "model/serving/serving_model.npz"

iam, lam, s3 = (boto3.client(n, region_name=REGION) for n in ("iam", "lambda", "s3"))
sch, sns, cw, sts = (boto3.client(n, region_name=REGION) for n in ("scheduler", "sns", "cloudwatch", "sts"))


def build_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in FILES.items():
            z.write(path, name)
    return buf.getvalue()


def ensure_role(name: str, trust: str, policy: str, policy_name: str) -> str:
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=(HERE / trust).read_text())["Role"]["Arn"]
        print("created role", name)
    iam.put_role_policy(RoleName=name, PolicyName=policy_name, PolicyDocument=policy)
    return arn


def apply() -> None:
    account = sts.get_caller_identity()["Account"]
    fn_arn = f"arn:aws:lambda:{REGION}:{account}:function:{FN}"
    try:  # the weekly retrain owns this object: seed it from the repo copy only when it does not exist yet
        s3.head_object(Bucket=BUCKET, Key=MODEL_KEY)
        print("model already at", f"s3://{BUCKET}/{MODEL_KEY}", "(left untouched)")
    except ClientError:
        s3.upload_file(str(MODEL_SRC), BUCKET, MODEL_KEY)
        print("seeded model ->", f"s3://{BUCKET}/{MODEL_KEY}")
    role_arn = ensure_role(ROLE, "iam_trust_policy.json", (HERE / "iam_permissions_policy.json").read_text(), "pipeline-least-privilege")
    sched_arn = ensure_role(SCHED_ROLE, "scheduler_trust_policy.json", (HERE / "scheduler_permissions_policy.json").read_text(), "invoke-only-this-lambda")
    code, cfg = build_zip(), dict(Runtime="python3.12", Handler="lambda_function.lambda_handler", Timeout=120, MemorySize=1024, Layers=[LAYER],
                                  Environment={"Variables": {"S3_BUCKET": BUCKET}}, Description="Daily gold pipeline: ingest, validate, forecast, log")
    for attempt in range(8):  # a new role needs a few seconds before Lambda can assume it
        try:
            try:
                lam.get_function(FunctionName=FN)
                lam.update_function_code(FunctionName=FN, ZipFile=code)
                lam.get_waiter("function_updated").wait(FunctionName=FN)
                lam.update_function_configuration(FunctionName=FN, Role=role_arn, **cfg)
            except lam.exceptions.ResourceNotFoundException:
                lam.create_function(FunctionName=FN, Role=role_arn, Code={"ZipFile": code}, **cfg)
            break
        except ClientError as e:
            if "cannot be assumed" not in str(e) or attempt == 7:
                raise
            time.sleep(5)
    lam.get_waiter("function_active_v2").wait(FunctionName=FN)
    print("function ready", fn_arn)
    target = {"Arn": fn_arn, "RoleArn": sched_arn, "RetryPolicy": {"MaximumRetryAttempts": 2}}
    args = dict(Name=SCHEDULE, ScheduleExpression="cron(0 1 * * ? *)", ScheduleExpressionTimezone="UTC", FlexibleTimeWindow={"Mode": "OFF"}, Target=target,
                Description="Daily gold pipeline at 08:00 Thailand time", State="ENABLED")
    try:
        sch.update_schedule(**args)
    except sch.exceptions.ResourceNotFoundException:
        sch.create_schedule(**args)
    print("schedule ready: cron(0 1 * * ? *) UTC")
    try:  # alarms need SNS and CloudWatch permissions that the project's IAM user may not have; the metrics are emitted either way
        topic = sns.create_topic(Name=TOPIC)["TopicArn"]
        for name, ns, metric, dims in ((f"{FN}-pipeline-failure", "GoldLSTM", "Failure", [{"Name": "FunctionName", "Value": FN}]),
                                       (f"{FN}-lambda-errors", "AWS/Lambda", "Errors", [{"Name": "FunctionName", "Value": FN}])):
            cw.put_metric_alarm(AlarmName=name, Namespace=ns, MetricName=metric, Dimensions=dims, Statistic="Sum", Period=86400, EvaluationPeriods=1, Threshold=1,
                                ComparisonOperator="GreaterThanOrEqualToThreshold", TreatMissingData="notBreaching", AlarmActions=[topic])
        print("alarms ready ->", topic, "(subscribe an email: aws sns subscribe --topic-arn", topic, "--protocol email --notification-endpoint YOU@example.com)")
    except ClientError as e:
        print("alarms skipped (missing SNS/CloudWatch permission):", str(e).split(":", 2)[-1][:110])


def teardown() -> None:
    for fn in (lambda: sch.delete_schedule(Name=SCHEDULE), lambda: lam.delete_function(FunctionName=FN),
               lambda: cw.delete_alarms(AlarmNames=[f"{FN}-pipeline-failure", f"{FN}-lambda-errors"]),
               lambda: sns.delete_topic(TopicArn=f"arn:aws:sns:{REGION}:{sts.get_caller_identity()['Account']}:{TOPIC}")):
        try:
            fn()
        except ClientError as e:
            print("skip:", str(e)[:90])
    for role, pol in ((ROLE, "pipeline-least-privilege"), (SCHED_ROLE, "invoke-only-this-lambda")):
        try:
            iam.delete_role_policy(RoleName=role, PolicyName=pol)
            iam.delete_role(RoleName=role)
        except ClientError as e:
            print("skip:", str(e)[:90])
    print("torn down (S3 data untouched)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if cmd == "plan":
        print(f"function {FN} (python3.12, layer {LAYER.split(':')[-2]}:{LAYER.split(':')[-1]}), zip = {len(build_zip()) / 1024:.0f} KB with {list(FILES)}")
        print(f"model {MODEL_SRC.name} ({MODEL_SRC.stat().st_size / 1024:.0f} KB) -> s3://{BUCKET}/{MODEL_KEY}")
        print(f"roles {ROLE}, {SCHED_ROLE}; schedule cron(0 1 * * ? *) UTC; topic {TOPIC}; 2 alarms")
    elif cmd == "apply":
        apply()
    elif cmd == "teardown":
        teardown()
    else:
        sys.exit(__doc__)
