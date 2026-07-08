"""
Unified evaluation for the paper. ONE reference classifier trained on REAL
held-out 128px images, applied to all exported generators so A/B/C are compared
on the same footing.

Metrics:
  - recognition (GAN-test): classifier-on-real accuracy over generated samples
  - diversity: mean intra-class L1 spread of generations
  - (Model C) SSIM to aligned real target + held-out structure test + gap

Writes reports/paper/results/metrics.json and metrics.csv.
Run:  python src/paper_eval.py
"""
import os, json, csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

import config as C
from config import IMG_SIZE, Z_DIM, RANDOM_SEED
from train_model_c import structure_map, load_images_128
from models import SelfAttention2D  # custom layer baked into the exported generators

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(REPO, "reports", "paper", "results")
os.makedirs(RES_DIR, exist_ok=True)

RUNS = {"A": C.DRIVE_BASE_A, "B": C.DRIVE_BASE_B, "C": C.DRIVE_BASE_C,
        "F": C.DRIVE_BASE_F, "G": C.DRIVE_BASE_G}
N_PER_CLASS = 40


def build_classifier(num_classes):
    return tf.keras.Sequential([
        tf.keras.Input((IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization(), layers.LeakyReLU(0.2), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same"), layers.BatchNormalization(), layers.LeakyReLU(0.2), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same"), layers.BatchNormalization(), layers.LeakyReLU(0.2), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same"), layers.BatchNormalization(), layers.LeakyReLU(0.2), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(128), layers.LeakyReLU(0.2), layers.Dropout(0.3),
        layers.Dense(num_classes, dtype="float32")])


def _load_gen(base):
    kpath = os.path.join(base, "checkpoints", "export", "generator.keras")
    if not os.path.exists(kpath):
        return None, False
    G = tf.keras.models.load_model(kpath, compile=False,
                                   custom_objects={"SelfAttention2D": SelfAttention2D})
    cfgp = os.path.join(base, "checkpoints", "export", "inference_config.json")
    structure = json.load(open(cfgp)).get("conditioned_on_structure", False) if os.path.exists(cfgp) else False
    return G, structure


def main():
    # force float32 eval (classifier stability); ignore mixed precision
    tf.keras.mixed_precision.set_global_policy("float32")
    X_tr, y_tr, X_ev, y_ev, enc = load_images_128(C.DATA_PATH)
    num_classes = len(enc.classes_)
    print(f"eval: {len(X_tr)} train / {len(X_ev)} held-out | {num_classes} classes")

    clf = build_classifier(num_classes)
    clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])
    clf.fit(X_tr, y_tr, epochs=12, batch_size=64, validation_data=(X_ev, y_ev), verbose=2)
    ref_acc = float(clf.evaluate(X_ev, y_ev, verbose=0)[1])
    print(f"reference classifier on real held-out = {ref_acc:.4f}")

    # structure maps for C conditioning
    C_tr = None

    def gen_samples(G, structure, n_per=N_PER_CLASS, seed=0):
        nonlocal C_tr
        tf.random.set_seed(seed)
        outs, ys = [], []
        for c in range(num_classes):
            oh = tf.one_hot(np.full(n_per, c), num_classes)
            nz = tf.random.normal([n_per, Z_DIM], seed=seed * 100 + c)
            if structure:
                if C_tr is None:
                    print("building structure maps for eval...")
                    C_tr = np.stack([structure_map(x) for x in tqdm(X_tr)]).astype(np.float32)
                pool = np.where(y_tr == c)[0]
                idx = np.random.default_rng(seed + c).choice(pool, n_per)
                # Keras 3 forbids mixing numpy + tensors in a list input -> make all tensors
                cond = tf.convert_to_tensor(C_tr[idx], dtype=tf.float32)
                f = G([cond, oh, nz], training=False).numpy()
            else:
                f = G([nz, oh], training=False).numpy()
            outs.append(f); ys.append(np.full(n_per, c))
        return np.concatenate(outs), np.concatenate(ys)

    def diversity(fakes, n_per=N_PER_CLASS):
        vals = []
        for c in range(num_classes):
            fl = fakes[c * n_per:(c + 1) * n_per].reshape(n_per, -1)
            m = min(n_per, 12)
            vals += [np.mean(np.abs(fl[i] - fl[j])) for i in range(m) for j in range(i + 1, m)]
        return float(np.mean(vals))

    per_model = {}
    for k, base in RUNS.items():
        G, structure = _load_gen(base)
        if G is None:
            print(f"[{k}] not exported - skipping"); continue
        fakes, ys = gen_samples(G, structure)
        pred = clf.predict(fakes, verbose=0).argmax(1)
        rec = float((pred == ys).mean())
        div = diversity(fakes)
        entry = {"recognition": round(rec, 4), "diversity": round(div, 4), "ssim": None}
        print(f"[{k}] recognition={rec:.4f} diversity={div:.4f}")

        if structure:
            # defensive: never let the structure-only SSIM/held-out block lose the
            # already-computed recognition+diversity numbers.
            try:
                from skimage.metrics import structural_similarity as ssim_fn
                C_ev = np.stack([structure_map(x) for x in X_ev]).astype(np.float32)
                oh = tf.one_hot(y_ev, num_classes)
                nz = tf.random.normal([len(X_ev), Z_DIM], seed=7)
                # generate in batches -> pushing all 5.4k held-out images through the
                # generator at once OOMs the 8 GB GPU.
                BS = 256
                parts = []
                for s in range(0, len(X_ev), BS):
                    parts.append(G([tf.convert_to_tensor(C_ev[s:s + BS], tf.float32),
                                    oh[s:s + BS], nz[s:s + BS]], training=False).numpy())
                fake_ev = np.concatenate(parts)
                rec_ho = float((clf.predict(fake_ev, verbose=0).argmax(1) == y_ev).mean())
                ss = float(np.mean([ssim_fn((X_ev[i, :, :, 0].astype(np.float32) + 1) / 2,
                                            (fake_ev[i, :, :, 0].astype(np.float32) + 1) / 2, data_range=1.0)
                                    for i in range(len(X_ev))]))
                entry["ssim"] = round(ss, 4)
                entry["heldout_recognition"] = round(rec_ho, 4)
                entry["generalization_gap"] = round(rec - rec_ho, 4)
                print(f"[{k}] held-out recog={rec_ho:.4f} gap={rec-rec_ho:.4f} ssim={ss:.4f}")
            except Exception as e:
                print(f"[{k}] SSIM/held-out block failed ({e}); keeping recognition+diversity")
        per_model[k] = entry

    out = {"reference_classifier_on_real": round(ref_acc, 4),
           "num_classes": num_classes, "img_size": IMG_SIZE,
           "n_per_class_eval": N_PER_CLASS, "per_model": per_model}
    with open(os.path.join(RES_DIR, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(RES_DIR, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "recognition", "diversity", "ssim", "heldout_recognition", "generalization_gap"])
        for k, e in per_model.items():
            w.writerow([k, e.get("recognition"), e.get("diversity"), e.get("ssim"),
                        e.get("heldout_recognition"), e.get("generalization_gap")])
    print("wrote", os.path.join(RES_DIR, "metrics.json"))


if __name__ == "__main__":
    main()
