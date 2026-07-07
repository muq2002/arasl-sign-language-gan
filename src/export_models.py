"""
Export trained generators (A, B, C) to inference-ready .keras files + label maps,
so the end-user interface can load them without any training code.

For A/B: rebuild the shared 128px generator (models.build_generator) and load
    <base>/checkpoints/generator.weights.h5.
For C: rebuild the structure-conditioned generator (train_model_c.build_generator)
    and load <base>/checkpoints/generator_C.weights.h5 (or reuse its own export).

Writes into <base>/checkpoints/export/{generator.keras, class_labels.json,
inference_config.json}. Run:  python src/export_models.py
"""
import os, json, sys
import numpy as np
import tensorflow as tf

import config as C
from config import Z_DIM, IMG_SIZE


def _labels(ckpt_dir):
    classes = np.load(os.path.join(ckpt_dir, "classes.npy"), allow_pickle=True)
    idx_to_label = {int(i): str(l) for i, l in enumerate(classes)}
    return idx_to_label, len(classes)


def _write_meta(exp, idx_to_label, num_classes, structure):
    with open(os.path.join(exp, "class_labels.json"), "w") as f:
        json.dump({"idx_to_label": {str(k): v for k, v in idx_to_label.items()},
                   "label_to_idx": {v: k for k, v in idx_to_label.items()}}, f, indent=2)
    with open(os.path.join(exp, "inference_config.json"), "w") as f:
        json.dump({"Z_DIM": Z_DIM, "IMG_SIZE": IMG_SIZE, "num_classes": num_classes,
                   "conditioned_on_structure": structure}, f, indent=2)


def export_ab(kind, base):
    from models import build_generator
    ckpt = os.path.join(base, "checkpoints")
    w = os.path.join(ckpt, "generator.weights.h5")
    if not os.path.exists(w):
        print(f"[{kind}] no weights at {w} - skipping"); return False
    idx_to_label, num_classes = _labels(ckpt)
    G = build_generator(Z_DIM, num_classes)
    G.load_weights(w)
    exp = os.path.join(ckpt, "export"); os.makedirs(exp, exist_ok=True)
    G.save(os.path.join(exp, "generator.keras"))
    _write_meta(exp, idx_to_label, num_classes, structure=False)
    print(f"[{kind}] exported -> {exp}")
    return True


def export_c(base):
    from train_model_c import build_generator as build_c
    ckpt = os.path.join(base, "checkpoints")
    w = os.path.join(ckpt, "generator_C.weights.h5")
    if not os.path.exists(w):
        print(f"[C] no weights at {w} - skipping"); return False
    idx_to_label, num_classes = _labels(ckpt)
    G = build_c(num_classes)
    G.load_weights(w)
    exp = os.path.join(ckpt, "export"); os.makedirs(exp, exist_ok=True)
    G.save(os.path.join(exp, "generator.keras"))
    _write_meta(exp, idx_to_label, num_classes, structure=True)
    print(f"[C] exported -> {exp}")
    return True


if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B", "C"]
    if "A" in which:
        export_ab("A", C.DRIVE_BASE_A)
    if "B" in which:
        export_ab("B", C.DRIVE_BASE_B)
    if "C" in which:
        export_c(C.DRIVE_BASE_C)
