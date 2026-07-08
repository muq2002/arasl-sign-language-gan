"""
Populate the per-model output sub-folders (samples / plots / history / eval)
for A, B, C so outputs/ is a complete, self-contained artifact set.

Per model:
  samples/  -> samples_grid.png (8 generations x 32 letters) + samples.npy
  plots/    -> loss curve PNG (copied from reports/paper/charts)
  history/  -> progress.json (copied) + loss_history.csv
  eval/     -> metrics.json (this model's recognition/diversity/SSIM + confusions)

Run:  python src/complete_outputs.py
"""
import os, json, glob, shutil, csv
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from config import IMG_SIZE, Z_DIM
from models import SelfAttention2D
from train_model_c import structure_map

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")
DATA = C.DATA_PATH
CHARTS = os.path.join(REPO, "reports", "paper", "charts")
RESULTS = os.path.join(REPO, "reports", "paper", "results")
K = 8  # samples per letter

RUNS = {"A": ("cgan_A_128", False), "B": ("cgan_B_128mp", False), "C": ("cgan_C_128struct", True)}

metrics = json.load(open(os.path.join(RESULTS, "metrics.json"))) if os.path.exists(os.path.join(RESULTS, "metrics.json")) else {"per_model": {}}
conf = json.load(open(os.path.join(RESULTS, "confusion_summary.json"))) if os.path.exists(os.path.join(RESULTS, "confusion_summary.json")) else {}


def load_gen(base):
    kp = os.path.join(base, "checkpoints", "export", "generator.keras")
    return tf.keras.models.load_model(kp, compile=False, custom_objects={"SelfAttention2D": SelfAttention2D})


def real_imgs(letter, n):
    import cv2
    files = glob.glob(os.path.join(DATA, letter, "*.png")) + glob.glob(os.path.join(DATA, letter, "*.jpg"))
    out = []
    for f in np.random.default_rng(0).choice(files, min(n, len(files)), replace=len(files) < n):
        im = cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        out.append(((im.astype(np.float32) - 127.5) / 127.5)[..., None])
    return out


def write_history(base, key):
    prog = json.load(open(os.path.join(base, "checkpoints", "progress.json")))
    hist_dir = os.path.join(base, "history"); os.makedirs(hist_dir, exist_ok=True)
    shutil.copy(os.path.join(base, "checkpoints", "progress.json"), os.path.join(hist_dir, "progress.json"))
    list_keys = [k for k, v in prog.items() if isinstance(v, list)]
    n = max(len(prog[k]) for k in list_keys)
    with open(os.path.join(hist_dir, "loss_history.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch"] + list_keys)
        for i in range(n):
            w.writerow([i + 1] + [round(prog[k][i], 5) if i < len(prog[k]) else "" for k in list_keys])


def main():
    tf.keras.mixed_precision.set_global_policy("float32")
    for key, (dirn, structure) in RUNS.items():
        base = os.path.join(OUT, dirn)
        for sub in ("samples", "plots", "history", "eval"):
            os.makedirs(os.path.join(base, sub), exist_ok=True)
        print(f"[{key}] loading generator...")
        G = load_gen(base)
        labels = json.load(open(os.path.join(base, "checkpoints", "export", "class_labels.json")))
        idx_to_label = {int(k): v for k, v in labels["idx_to_label"].items()}
        letters = [idx_to_label[i] for i in sorted(idx_to_label)]
        n_cls = len(letters)

        # ---- generate K samples per letter ----
        grid = np.zeros((n_cls, K, IMG_SIZE, IMG_SIZE), np.float32)
        for idx, letter in enumerate(letters):
            oh = tf.one_hot([idx] * K, n_cls)
            nz = tf.random.normal([K, Z_DIM], seed=idx)
            if structure:
                refs = real_imgs(letter, K)
                cond = tf.convert_to_tensor(np.stack([structure_map(r) for r in refs]).astype(np.float32))
                fake = G([cond, oh, nz], training=False).numpy()
            else:
                fake = G([nz, oh], training=False).numpy()
            grid[idx] = fake[:, :, :, 0]
        np.save(os.path.join(base, "samples", "samples.npy"), grid.astype(np.float16))

        # ---- labeled grid figure ----
        fig, axes = plt.subplots(n_cls, K, figsize=(K * 1.1, n_cls * 1.1))
        for r in range(n_cls):
            for c in range(K):
                ax = axes[r, c]
                ax.imshow((grid[r, c] * 127.5 + 127.5).clip(0, 255).astype(np.uint8), cmap="gray")
                ax.axis("off")
            axes[r, 0].set_ylabel(letters[r], rotation=0, ha="right", va="center", fontsize=8)
            axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        fig.suptitle(f"Model {key} - 8 generations x {n_cls} letters", y=1.001)
        fig.tight_layout()
        fig.savefig(os.path.join(base, "samples", "samples_grid.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

        # ---- plots: copy loss curve ----
        src_plot = os.path.join(CHARTS, f"loss_model_{key}.png")
        if os.path.exists(src_plot):
            shutil.copy(src_plot, os.path.join(base, "plots", f"loss_model_{key}.png"))

        # ---- history ----
        write_history(base, key)

        # ---- eval: per-model metrics ----
        ev = {"model": key,
              "metrics": metrics.get("per_model", {}).get(key, {}),
              "reference_classifier_on_real": metrics.get("reference_classifier_on_real"),
              "confusion": conf.get(key, {})}
        json.dump(ev, open(os.path.join(base, "eval", "metrics.json"), "w"), indent=2, ensure_ascii=False)
        print(f"[{key}] done -> samples/plots/history/eval populated")

    print("OUTPUTS_COMPLETE")


if __name__ == "__main__":
    main()
