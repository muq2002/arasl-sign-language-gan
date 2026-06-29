# -*- coding: utf-8 -*-
"""Step 1 (arasl_venv): deterministic data prep -> saves arrays both other scripts share."""
import io, json, numpy as np, pandas as pd, cv2
from PIL import Image

SUBSET_CLASSES, PER_CLASS, IMG_SIZE, SEED = 10, 500, 64, 42   # ~5,000 images
np.random.seed(SEED)

df = pd.read_parquet("arasl.parquet")
lab_col = "label" if "label" in df.columns else df.columns[-1]
img_col = "image" if "image" in df.columns else df.columns[0]
print(f"dataset rows={len(df)} classes={df[lab_col].nunique()} (same ArASL2018 as Model A/B)")

def decode(v):
    b = v["bytes"] if isinstance(v, dict) else v
    im = Image.open(io.BytesIO(b)).convert("L").resize((IMG_SIZE, IMG_SIZE))
    return (np.asarray(im, np.float32) - 127.5) / 127.5

chosen = sorted(df[lab_col].unique())[:SUBSET_CLASSES]
sub = pd.concat([df[df[lab_col] == c].head(PER_CLASS) for c in chosen]) \
        .sample(frac=1, random_state=SEED).reset_index(drop=True)
X = np.stack([decode(v) for v in sub[img_col]])[..., None].astype(np.float32)
y = sub[lab_col].map({c: i for i, c in enumerate(chosen)}).to_numpy().astype(np.int64)
n_eval = SUBSET_CLASSES * 25
Xtr, ytr, Xev, yev = X[:-n_eval], y[:-n_eval], X[-n_eval:], y[-n_eval:]

def structure_map(img):
    g = ((img[:, :, 0] + 1) * 127.5).clip(0, 255).astype(np.uint8)
    edge = cv2.Canny(g, 60, 160)
    _, sil = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if sil.mean() > 127: sil = 255 - sil
    dist = cv2.normalize(cv2.distanceTransform(sil, cv2.DIST_L2, 3), None, 0, 255, cv2.NORM_MINMAX)
    return (np.stack([edge, sil, dist], -1).astype(np.float32) / 127.5) - 1.0
Ctr = np.stack([structure_map(x) for x in Xtr]).astype(np.float32)

np.save("Xtr.npy", Xtr); np.save("ytr.npy", ytr)
np.save("Xev.npy", Xev); np.save("yev.npy", yev); np.save("Ctr.npy", Ctr)
json.dump({"classes": [int(c) for c in chosen], "img": IMG_SIZE,
           "per_class": PER_CLASS, "n_train": int(len(Xtr)), "n_eval": int(len(Xev))},
          open("prep_meta.json", "w"), indent=2)
print(f"saved Xtr{Xtr.shape} Ctr{Ctr.shape} | classes {chosen} | "
      f"struct coverage {float(np.mean([np.any(c>-0.99) for c in Ctr])):.2f}")
