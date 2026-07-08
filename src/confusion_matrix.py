"""
Per-class confusion analysis for the exported generators (A/B/C).

Trains ONE reference classifier on REAL 128px held-out images, then for each
model generates N samples/class, classifies them, and builds a confusion matrix
(true intended class vs predicted). Saves a heatmap PNG per model + a JSON of the
most-confused letter pairs. This reveals WHERE the recognition gap comes from.

Run:  python src/confusion_matrix.py
"""
import os, json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import config as C
from config import IMG_SIZE, Z_DIM
from train_model_c import structure_map, load_images_128
from models import SelfAttention2D
from paper_eval import build_classifier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "reports", "paper", "results")
os.makedirs(RES, exist_ok=True)
RUNS = {"A": C.DRIVE_BASE_A, "B": C.DRIVE_BASE_B, "C": C.DRIVE_BASE_C,
        "F": C.DRIVE_BASE_F, "G": C.DRIVE_BASE_G}
N_PER = 60
BS = 256


def load_gen(base):
    kpath = os.path.join(base, "checkpoints", "export", "generator.keras")
    if not os.path.exists(kpath):
        return None, False
    G = tf.keras.models.load_model(kpath, compile=False,
                                   custom_objects={"SelfAttention2D": SelfAttention2D})
    with open(os.path.join(base, "checkpoints", "export", "inference_config.json")) as f:
        structure = json.load(f).get("conditioned_on_structure", False)
    return G, structure


def gen_for_class(G, structure, c, num_classes, X_tr, y_tr):
    oh = tf.one_hot(np.full(N_PER, c), num_classes)
    nz = tf.random.normal([N_PER, Z_DIM], seed=100 + c)
    if structure:
        pool = np.where(y_tr == c)[0]
        idx = np.random.default_rng(c).choice(pool, N_PER)
        cond = tf.convert_to_tensor(np.stack([structure_map(X_tr[i]) for i in idx]).astype(np.float32))
        out = []
        for s in range(0, N_PER, BS):
            out.append(G([cond[s:s + BS], oh[s:s + BS], nz[s:s + BS]], training=False).numpy())
        return np.concatenate(out)
    return G([nz, oh], training=False).numpy()


def main():
    tf.keras.mixed_precision.set_global_policy("float32")
    X_tr, y_tr, X_ev, y_ev, enc = load_images_128(C.DATA_PATH)
    classes = list(enc.classes_)
    num_classes = len(classes)

    clf = build_classifier(num_classes)
    clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])
    clf.fit(X_tr, y_tr, epochs=12, batch_size=64, validation_data=(X_ev, y_ev), verbose=2)

    summary = {}
    for k, base in RUNS.items():
        G, structure = load_gen(base)
        if G is None:
            print(f"[{k}] not exported - skip"); continue
        y_true, y_pred = [], []
        for c in range(num_classes):
            fakes = gen_for_class(G, structure, c, num_classes, X_tr, y_tr)
            p = clf.predict(fakes, verbose=0).argmax(1)
            y_pred.append(p); y_true.append(np.full(len(p), c))
        y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        acc = float((y_true == y_pred).mean())

        # heatmap
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(num_classes)); ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_yticks(range(num_classes)); ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true (intended)")
        ax.set_title(f"Model {k} - confusion (row-normalized)  |  recognition={acc:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        out_png = os.path.join(RES, f"confusion_{k}.png")
        fig.savefig(out_png, dpi=130); plt.close(fig)

        # most-confused pairs + weakest classes
        pairs = sorted([(classes[i], classes[j], round(float(cmn[i, j]), 3))
                        for i in range(num_classes) for j in range(num_classes) if i != j],
                       key=lambda t: -t[2])[:12]
        per_class = sorted([(classes[i], round(float(cmn[i, i]), 3)) for i in range(num_classes)],
                           key=lambda t: t[1])[:8]
        summary[k] = {"recognition": round(acc, 4), "worst_classes": per_class, "top_confusions": pairs}
        print(f"[{k}] recognition={acc:.3f}  worst: {per_class[:3]}  top confusion: {pairs[0]}")

    with open(os.path.join(RES, "confusion_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("wrote", os.path.join(RES, "confusion_summary.json"))


if __name__ == "__main__":
    main()
