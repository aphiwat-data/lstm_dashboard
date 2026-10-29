"""Usage: python lstm_quantile.py FEATURES.parquet OUT_DIR  (writes OUT_DIR/lstm_quantile_bands.npz)
Walk-forward LSTM that predicts the 5% and 95% quantiles of the next-day log-return directly (pinball loss). Same protocol as walk_forward.py."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hourly"))
from train_models import windows
from tensorflow import keras
import tensorflow as tf

FEATURES, SCR = Path(sys.argv[1]), Path(sys.argv[2]); W, UNITS, SEEDS, EPOCHS, EVERY, TRAIN = 20, 32, 2, 30, 7, 365
df = pd.read_parquet(FEATURES).sort_index()
feats = [c for c in df.columns if c.startswith("f_")]
F, y = df[feats].to_numpy(float), df["y_ret_next"].to_numpy(float)
start = int(np.flatnonzero((df["meta_split"] == "test").to_numpy())[0]); n = len(df)
TAUS = tf.constant([0.05, 0.95], dtype=tf.float32)

def pinball(yt, q):
    e = tf.reshape(yt, (-1, 1)) - q
    return tf.reduce_mean(tf.maximum(TAUS * e, (TAUS - 1) * e))

def build(seed):
    keras.utils.set_random_seed(seed)
    m = keras.Sequential([keras.layers.Input((W, F.shape[1])), keras.layers.LSTM(UNITS), keras.layers.Dropout(0.1), keras.layers.Dense(2)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=pinball)
    return m

lo_out, hi_out = np.zeros(n - start), np.zeros(n - start)
s, t0, block = start, time.time(), 0
while s < n:
    e = min(s + EVERY, n); tr = np.arange(max(0, s - TRAIN), s); tg = np.arange(s, e)
    sd_r = float(y[tr].std()); mu, sd = np.nanmean(F[tr], 0), np.nanstd(F[tr], 0); sd[sd == 0] = 1
    Z = np.clip(np.nan_to_num((F - mu) / sd), -10, 10)
    lo_i = tr.min() - W + 1; hi_i = max(tr.max(), tg.max()) + 1
    Zw = windows(Z[lo_i:hi_i], W); pick = lambda pos: Zw[pos - (lo_i + W - 1)]
    qs = []
    for seed in range(SEEDS):
        keras.backend.clear_session(); m = build(seed)
        m.fit(pick(tr), (y[tr] / sd_r).astype(np.float32), validation_split=0.15, epochs=EPOCHS, batch_size=64, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
        qs.append(m.predict(pick(tg), batch_size=512, verbose=0) * sd_r)
    q = np.mean(qs, axis=0); q.sort(axis=1)
    lo_out[s - start:e - start], hi_out[s - start:e - start] = q[:, 0], q[:, 1]
    block += 1; print(f"block {block} {df.index[s].date()} {time.time() - t0:.0f}s", flush=True); s = e
np.savez(SCR / "lstm_quantile_bands.npz", lo=lo_out, hi=hi_out); print("DONE")
