"""
Validation error of the frozen landmark regressor (the supervisor for B, F and G).

Rebuilds the exact split train_regressor() used in train_model_b.py: images in
the seed-42 order from data.load_images(), restricted to MediaPipe detections,
last 10% held out (Keras validation_split=0.1 takes the tail, no reshuffle).
Reports MSE/MAE in normalised units and in pixels, next to a constant-predictor
baseline (training-set mean landmark vector applied to validation).

Needs the two gitignored artifacts from the training machine:
  outputs/cgan_B_128mp/checkpoints/landmark_regressor.weights.h5
  outputs/cgan_B_128mp/landmarks_128px.npy

Reproduction on another machine (no original weights): point ARASL_LM_CACHE at a
re-extracted landmark cache, ARASL_REG_WEIGHTS at a NEW weights path and set
ARASL_REG_RETRAIN=1. The regressor is then trained with train_model_b's own
train_regressor() (timed) and the output is flagged "reproduced": true.

Writes reports/paper/results/landmark_regressor_eval.json (override: ARASL_REG_OUT).
Run:  python src/eval_landmark_regressor.py
"""
import os, json, math, time
import numpy as np
import tensorflow as tf

import config as C
from data import load_images
from models import build_landmark_regressor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("ARASL_REG_OUT",
                     os.path.join(REPO, "reports", "paper", "results", "landmark_regressor_eval.json"))
VAL_SPLIT = 0.1


def _errors(pred, true):
    """MSE/MAE over all 63 values (normalised) and over x,y only in pixels."""
    d = pred - true
    dxy = d.reshape(-1, 21, 3)[:, :, :2] * C.IMG_SIZE      # z has no pixel meaning
    return {"mse_norm": float(np.mean(d ** 2)), "mae_norm": float(np.mean(np.abs(d))),
            "mse_px_xy": float(np.mean(dxy ** 2)), "mae_px_xy": float(np.mean(np.abs(dxy))),
            "mean_euclid_px": float(np.mean(np.linalg.norm(dxy, axis=-1)))}


def main():
    retrain = os.environ.get("ARASL_REG_RETRAIN", "0") == "1"
    wpath = os.environ.get("ARASL_REG_WEIGHTS",
                           os.path.join(C.DRIVE_BASE_B, "checkpoints", "landmark_regressor.weights.h5"))
    cache = os.environ.get("ARASL_LM_CACHE", os.path.join(C.DRIVE_BASE_B, "landmarks_128px.npy"))
    for p in (cache,) if retrain else (wpath, cache):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} (exists only on the training machine)")

    if retrain:
        C.setup_speed()          # same seeds / precision policy as train_model_b.train()
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
    images, _, enc, _ = load_images()
    landmarks = np.load(cache)
    assert landmarks.shape == (len(images), 63), landmarks.shape

    valid = landmarks.any(axis=1)
    x, y = images[valid], np.clip(landmarks[valid], 0.0, 1.0)
    split_at = int(math.floor(len(x) * (1.0 - VAL_SPLIT)))
    x_tr, y_tr, x_va, y_va = x[:split_at], y[:split_at], x[split_at:], y[split_at:]

    train_seconds = None
    if retrain:
        if os.path.exists(wpath):
            raise SystemExit(f"{wpath} already exists - refusing to overwrite; pick a new ARASL_REG_WEIGHTS")
        from models import build_generator, build_discriminator
        from train_model_b import train_regressor
        # train() builds G and D before the regressor; do the same so the seeded
        # initialiser stream reaches the regressor in the same state.
        build_generator(C.Z_DIM, len(enc.classes_)); build_discriminator(len(enc.classes_))
        reg = build_landmark_regressor(C.IMG_SIZE)
        t0 = time.time()
        reg = train_regressor(reg, images, landmarks, wpath)
        train_seconds = round(time.time() - t0, 1)
        print(f"regressor training wall-clock: {train_seconds}s")
    else:
        reg = build_landmark_regressor(C.IMG_SIZE)
        reg.load_weights(wpath)
    pred = reg.predict(x_va, batch_size=C.REG_BATCH_SIZE, verbose=0)
    const = np.repeat(y_tr.mean(axis=0, keepdims=True), len(y_va), axis=0)

    out = {"seed": C.RANDOM_SEED, "img_size": C.IMG_SIZE,
           "reproduced": retrain, "train_seconds": train_seconds,
           "devices": [d.name for d in tf.config.list_physical_devices("GPU")] or ["CPU"],
           "weights": wpath,
           "weights_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(wpath))),
           "n_total": int(len(images)), "n_detected": int(valid.sum()),
           "detection_rate": round(float(valid.mean()), 4),
           "n_train": int(len(x_tr)), "n_val": int(len(x_va)),
           "regressor": _errors(pred, y_va), "constant_baseline": _errors(const, y_va)}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
