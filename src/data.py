"""
Data loading + a fast tf.data input pipeline shared by both models.

Why this is faster than the original (without changing what the model sees):

  * The original built `tf.constant(slice)` and `tf.one_hot(...)` on the host
    every single step, then blocked on the H->D copy. Here we gather + one-hot
    inside a prefetched tf.data pipeline, so copies overlap with GPU compute.

  * The Phase-2 structural target was selected with a Python list-comprehension
    calling np.random.choice per element. Here the whole epoch's targets are
    chosen with ONE vectorized numpy expression, then handed to the graph via a
    tf.Variable that the pipeline gathers from.

  * Images live in a SINGLE host-side constant that is reused both as the input
    source and the target source (prototypes are concatenated on the end), so
    memory use is lower than the original two-copy approach.
"""
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from config import DATA_PATH, IMG_SIZE, RANDOM_SEED, BATCH_SIZE, TF_DATA_PREFETCH

VALID_EXT = (".png", ".jpg", ".jpeg")
AUTOTUNE = tf.data.AUTOTUNE


def load_images(data_path=DATA_PATH):
    """Load grayscale signs -> (images[-1,1], labels_int, encoder, prototypes, class_index)."""
    imgs, labs, errs = [], [], 0
    print(f"Loading from {data_path} (resize {IMG_SIZE}x{IMG_SIZE})")
    for sub in tqdm(sorted(os.listdir(data_path)), desc="Classes"):
        spath = os.path.join(data_path, sub)
        if not os.path.isdir(spath):
            continue
        for fn in sorted(os.listdir(spath)):
            if not fn.lower().endswith(VALID_EXT):
                continue
            try:
                raw = tf.io.read_file(os.path.join(spath, fn))
                img = tf.image.decode_png(raw, channels=1)
                img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
                img = (tf.cast(img, tf.float32) - 127.5) / 127.5
                imgs.append(img.numpy()); labs.append(sub)
            except Exception:
                errs += 1
    images = np.asarray(imgs, dtype=np.float32)
    enc = LabelEncoder()
    labels_int = enc.fit_transform(np.asarray(labs)).astype(np.int64)
    num_classes = len(enc.classes_)
    print(f"Loaded {len(images)} images, {errs} errors, {num_classes} classes")

    # shuffle once (deterministic)
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(images))
    images, labels_int = images[perm], labels_int[perm]

    prototypes = np.zeros((num_classes, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    for c in range(num_classes):
        m = labels_int == c
        if m.any():
            prototypes[c] = images[m].mean(axis=0)
    return images, labels_int, enc, prototypes


class TargetSelector:
    """Vectorized Phase-1 (prototype) / Phase-2 (random real member) target indexing.

    Targets index into `source = concat([images, prototypes])` of length N + C:
      * prototype row for class c -> index  N + c
      * a real member of class c  -> its own index in [0, N)
    """

    def __init__(self, labels_int, num_classes, n_images):
        self.labels = labels_int
        self.N = n_images
        order = np.argsort(labels_int, kind="stable")
        self.flat_members = order.astype(np.int64)
        counts = np.bincount(labels_int, minlength=num_classes)
        self.counts = counts
        self.offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)

    def epoch_targets(self, epoch, phase2_ep, rng):
        if epoch < phase2_ep:                       # Phase 1: prototypes
            return (self.N + self.labels).astype(np.int32)
        rnd = (rng.random(self.N) * self.counts[self.labels]).astype(np.int64)
        idx = self.flat_members[self.offsets[self.labels] + rnd]
        return idx.astype(np.int32)


def build_pipeline(images, labels_int, prototypes, num_classes,
                   landmarks=None, batch_size=BATCH_SIZE):
    """Return (dataset, target_idx_var, selector).

    dataset yields per batch: (real_img, onehot_label, target_img[, landmark]).
    Reshuffles every epoch. Update `target_idx_var` once per epoch before iterating.
    """
    N = len(images)
    # single host constant reused as input AND target source
    source = np.concatenate([images, prototypes], axis=0)
    source_tf = tf.constant(source)                 # (N+C, H, W, 1) float32
    labels_tf = tf.constant(labels_int, dtype=tf.int32)
    target_idx_var = tf.Variable(np.zeros(N, np.int32), trainable=False)

    has_lm = landmarks is not None
    lm_tf = tf.constant(landmarks, dtype=tf.float32) if has_lm else None

    def map_fn(idx):
        img = tf.gather(source_tf, idx)
        oneh = tf.one_hot(tf.gather(labels_tf, idx), num_classes)
        tgt = tf.gather(source_tf, tf.gather(target_idx_var, idx))
        if has_lm:
            lm = tf.gather(lm_tf, idx)
            return img, oneh, tgt, lm
        return img, oneh, tgt

    ds = (tf.data.Dataset.from_tensor_slices(tf.range(N, dtype=tf.int32))
          .shuffle(N, seed=RANDOM_SEED, reshuffle_each_iteration=True)
          .batch(batch_size, drop_remainder=True)
          .map(map_fn, num_parallel_calls=AUTOTUNE))
    if TF_DATA_PREFETCH:
        ds = ds.prefetch(AUTOTUNE)

    selector = TargetSelector(labels_int, num_classes, N)
    return ds, target_idx_var, selector
