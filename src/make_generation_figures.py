"""
Paper figures of GENERATED samples from all five exported models (A/B/C/F/G).
GPU (loads the exported generators). Writes to reports/paper/figures/:

  samples_<X>.png        — 8 generations x 32 letters, one grid per model
  real_vs_generated.png  — per letter: real | A | B | C | F | G (same structure
                           ref for the structure-conditioned models, fair compare)

Run:  python src/make_generation_figures.py
"""
import os, json, glob, sys
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from models import SelfAttention2D
from train_model_c import structure_map

OUT = os.path.join(REPO, "outputs")
DATA = os.path.join(REPO, "data", "ArASL_dataset")
FIG = os.path.join(REPO, "reports", "paper", "figures")
os.makedirs(FIG, exist_ok=True)
IMG, Z = 128, 128
K = 8  # generations per letter in the per-model grids
RUNS = [("A", "cgan_A_128"), ("B", "cgan_B_128mp"), ("C", "cgan_C_128struct"),
        ("F", "cgan_F_128fusion"), ("G", "cgan_G_128plus")]


def load(run_dir):
    exp = os.path.join(OUT, run_dir, "checkpoints", "export")
    kp = os.path.join(exp, "generator.keras")
    if not os.path.exists(kp):
        return None
    G = tf.keras.models.load_model(kp, compile=False, custom_objects={"SelfAttention2D": SelfAttention2D})
    labels = json.load(open(os.path.join(exp, "class_labels.json")))
    cfg = json.load(open(os.path.join(exp, "inference_config.json")))
    return {"G": G, "l2i": labels["label_to_idx"],
            "i2l": {int(k): v for k, v in labels["idx_to_label"].items()},
            "structure": cfg.get("conditioned_on_structure", False)}


def real_img(letter, seed):
    files = sorted(glob.glob(os.path.join(DATA, letter, "*.png")))
    if not files:
        return None
    f = files[np.random.default_rng(seed).integers(len(files))]
    im = cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE), (IMG, IMG))
    return (im.astype(np.float32) - 127.5) / 127.5


def gen(m, letter, n, seed, ref=None):
    ci = m["l2i"][letter]
    oh = tf.one_hot([ci] * n, len(m["i2l"]))
    nz = tf.random.normal([n, Z], seed=seed)
    if m["structure"]:
        r = ref if ref is not None else real_img(letter, seed)
        cond = np.stack([structure_map(r[..., None])] * n).astype(np.float32)
        fake = m["G"]([tf.convert_to_tensor(cond, tf.float32), oh, nz], training=False).numpy()
    else:
        fake = m["G"]([nz, oh], training=False).numpy()
    return [(fake[i, :, :, 0] * 127.5 + 127.5).clip(0, 255).astype(np.uint8) for i in range(n)]


def per_model_grids(models, letters):
    for key, m in models.items():
        n_cls = len(letters)
        fig, axes = plt.subplots(n_cls, K, figsize=(K * 1.05, n_cls * 1.05))
        for r, lt in enumerate(letters):
            imgs = gen(m, lt, K, seed=1000 + r)
            for c in range(K):
                ax = axes[r, c]
                ax.imshow(imgs[c], cmap="gray"); ax.axis("off")
            axes[r, 0].set_ylabel(lt, rotation=0, ha="right", va="center", fontsize=8)
            axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        fig.suptitle(f"Model {key} — {K} generations x {n_cls} letters", y=1.001)
        fig.tight_layout()
        out = os.path.join(FIG, f"samples_{key}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        print("wrote", out)


def real_vs_generated(models, letters):
    order = ["A", "B", "C", "F", "G"]
    cols = ["real"] + [k for k in order if k in models]
    n_cls = len(letters)
    fig, axes = plt.subplots(n_cls, len(cols), figsize=(len(cols) * 1.15, n_cls * 1.15))
    for r, lt in enumerate(letters):
        ref = real_img(lt, seed=7)
        axes[r, 0].imshow((ref * 127.5 + 127.5).clip(0, 255).astype(np.uint8), cmap="gray")
        for c, key in enumerate(cols[1:], start=1):
            m = models[key]
            img = gen(m, lt, 1, seed=7, ref=ref if m["structure"] else None)[0]
            axes[r, c].imshow(img, cmap="gray")
        for c in range(len(cols)):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(lt, rotation=0, ha="right", va="center", fontsize=8)
    for c, name in enumerate(cols):
        axes[0, c].set_title(name, fontsize=11)
    fig.suptitle("Real vs generated — same structure per letter (A/B from noise)", y=1.002)
    fig.tight_layout()
    out = os.path.join(FIG, "real_vs_generated.png")
    fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    tf.keras.mixed_precision.set_global_policy("float32")
    models = {k: m for k, d in RUNS if (m := load(d)) is not None}
    print("loaded models:", list(models.keys()))
    letters = [models[next(iter(models))]["i2l"][i] for i in sorted(models[next(iter(models))]["i2l"])]
    per_model_grids(models, letters)
    real_vs_generated(models, letters)
    print("FIGURES_DONE")


if __name__ == "__main__":
    main()
