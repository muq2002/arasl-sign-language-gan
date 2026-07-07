"""
Dump the exact 128px training-order image array so the (TF-free) MediaPipe env
can extract landmarks that align with Model B's training data.

load_images() is deterministic (fixed seed, sorted file order), so the array it
produces here is byte-identical to what train_model_b.py will load. We save it as
float16 to halve disk (~1.8 GB) into the Model B run dir; extract_lm_full.py reads
it and writes landmarks_128px.npy next to it, which train_model_b.py then caches.

Run (TF env):  python src/dump_images_for_lm.py
"""
import os
import numpy as np

import config as C
from data import load_images

def main():
    base = C.DRIVE_BASE_B
    os.makedirs(base, exist_ok=True)
    images, labels_int, enc, prototypes = load_images()
    # Large (~1.8 GB) transient handoff -> WSL-native /root (ext4: fast AND
    # persistent across WSL restarts, unlike tmpfs /tmp; /mnt/c can I/O-error on
    # big writes). Not a deliverable.
    out = os.environ.get("ARASL_LM_IMAGES", "/root/arasl_b_images_128.npy")
    np.save(out, images.astype(np.float16))
    np.save(os.path.join(base, "labels_128.npy"), labels_int)
    np.save(os.path.join(base, "classes_dump.npy"), enc.classes_)
    print(f"saved {out}  shape={images.shape}  dtype=float16")
    print(f"n={len(images)}  classes={len(enc.classes_)}")

if __name__ == "__main__":
    main()
