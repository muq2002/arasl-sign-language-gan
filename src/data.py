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


def read_image_folder(data_path, dtype):
    """Decode every class-folder image into ONE preallocated (N, H, W, 1) array.

    Same values and order as appending per-image arrays to a list and calling
    np.asarray at the end, without holding both copies at once (that peak was
    ~2x the final array and got long CPU runs killed on a 16 GB machine).
    Returns (images, labels, n_errors).
    """
    files = []
    for sub in sorted(os.listdir(data_path)):
        spath = os.path.join(data_path, sub)
        if os.path.isdir(spath):
            files += [(sub, os.path.join(spath, fn)) for fn in sorted(os.listdir(spath))
                      if fn.lower().endswith(VALID_EXT)]
    print(f"Loading from {data_path} (resize {IMG_SIZE}x{IMG_SIZE})")
    images = np.empty((len(files), IMG_SIZE, IMG_SIZE, 1), dtype=dtype)
    labs, n = [], 0
    for sub, path in tqdm(files, desc="Images"):
        try:
            raw = tf.io.read_file(path)
            img = tf.image.decode_png(raw, channels=1)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            images[n] = ((tf.cast(img, tf.float32) - 127.5) / 127.5).numpy()
            labs.append(sub); n += 1
        except Exception:
            pass
    return images[:n], labs, len(files) - n


def permute_rows_inplace(arr, perm):
    """arr[:] = arr[perm] without allocating a second full-size array (cycle walk)."""
    seen = np.zeros(len(perm), dtype=bool)
    for start in range(len(perm)):
        if seen[start] or perm[start] == start:
            seen[start] = True
            continue
        tmp = arr[start].copy()
        j = start
        while True:
            seen[j] = True
            k = perm[j]
            if k == start:
                arr[j] = tmp
                break
            arr[j] = arr[k]
            j = k
    return arr


def load_images(data_path=DATA_PATH):
    """Load grayscale signs -> (images[-1,1], labels_int, encoder, prototypes, class_index)."""
    images, labs, errs = read_image_folder(data_path, np.float32)
    # optional subsample for quick smoke tests (per-class cap, keeps all classes)
    _cap = int(os.environ.get("ARASL_MAX_PER_CLASS", 0))
    if _cap > 0:
        labs_arr = np.asarray(labs)
        keep = []
        for c in np.unique(labs_arr):
            idx = np.where(labs_arr == c)[0][:_cap]
            keep.extend(idx.tolist())
        keep = np.array(sorted(keep))
        images = images[keep]
        labs = [labs[i] for i in keep]
        print(f"[smoke] subsampled to {len(images)} images ({_cap}/class)")
    enc = LabelEncoder()
    labels_int = enc.fit_transform(np.asarray(labs)).astype(np.int64)
    num_classes = len(enc.classes_)
    print(f"Loaded {len(images)} images, {errs} errors, {num_classes} classes")

    # shuffle once (deterministic)
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(images))
    images, labels_int = permute_rows_inplace(images, perm), labels_int[perm]

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
