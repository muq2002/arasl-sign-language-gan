"""
Model C — structure-conditioned cGAN, 128x128 (standalone, repo-local).

Faithful port of notebooks/model_C_cgan_128_structure.ipynb into a runnable
script that matches the src/ A & B training style (checkpoint/resume, per-epoch
loss history in progress.json, mixed precision via config, repo-relative paths).

Conditioning: per-image 3-channel structure map (Canny edge + silhouette +
distance transform). Generator: encoder-decoder (structure, label, noise)->image.
Discriminator: paired (image, structure, label). Loss: adversarial + ALIGNED L1
(target == the image the structure came from) -> no regress-to-mean.

Run:  python src/train_model_c.py
"""
import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import config as C
from config import IMG_SIZE, Z_DIM, RANDOM_SEED

COND_CH = 3
LAMBDA_L1 = 5.0
CANNY_LO, CANNY_HI = 60, 160
EVAL_FRACTION, MIN_EVAL_PER_CLS = 0.10, 30
VALID_EXT = (".png", ".jpg", ".jpeg")


# ── data loading (128px, deterministic shuffle, held-out split) ───────────────
def load_images_128(data_path):
    import cv2  # noqa: F401  (imported later for structure maps)
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
    images = np.asarray(imgs, dtype=np.float16)   # float16 to halve RAM (~3.4->1.7 GB)
    cap = int(os.environ.get("ARASL_MAX_PER_CLASS", 0))
    if cap > 0:
        labs_arr = np.asarray(labs); keep = []
        for c in np.unique(labs_arr):
            keep.extend(np.where(labs_arr == c)[0][:cap].tolist())
        keep = np.array(sorted(keep)); images = images[keep]
        labs = [labs[i] for i in keep]
        print(f"[smoke] subsampled to {len(images)} images ({cap}/class)")
    enc = LabelEncoder()
    labels_int = enc.fit_transform(np.asarray(labs)).astype(np.int64)
    num_classes = len(enc.classes_)
    print(f"Loaded {len(images)} images, {errs} errors, {num_classes} classes")
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(images))
    images, labels_int = images[perm], labels_int[perm]
    n_eval = max(num_classes * MIN_EVAL_PER_CLS, int(EVAL_FRACTION * len(images)))
    return images[:-n_eval], labels_int[:-n_eval], images[-n_eval:], labels_int[-n_eval:], enc


def structure_map(img_norm):
    import cv2
    g = ((img_norm[:, :, 0] + 1) * 127.5).clip(0, 255).astype(np.uint8)
    edge = cv2.Canny(g, CANNY_LO, CANNY_HI)
    _, sil = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if sil.mean() > 127:
        sil = 255 - sil
    dist = cv2.normalize(cv2.distanceTransform(sil, cv2.DIST_L2, 3),
                         None, 0, 255, cv2.NORM_MINMAX)
    return (np.stack([edge, sil, dist], -1).astype(np.float32) / 127.5) - 1.0


# ── models (structure-conditioned) ────────────────────────────────────────────
def _gen_block(x, f):
    x = layers.Conv2DTranspose(f, 4, 2, padding="same", use_bias=False)(x)
    return layers.ReLU()(layers.BatchNormalization()(x))


def build_generator(num_classes):
    c = tf.keras.Input((IMG_SIZE, IMG_SIZE, COND_CH))
    l = tf.keras.Input((num_classes,))
    n = tf.keras.Input((Z_DIM,))
    e = c
    for f in [64, 128, 256, 512]:
        e = layers.LeakyReLU(0.2)(layers.BatchNormalization()(
            layers.Conv2D(f, 4, 2, padding="same", use_bias=False)(e)))
    # e is now the 8x8x512 structure bottleneck (kept SPATIAL).
    # The notebook flattened it and did Dense(8*8*512) over [flat_e, noise, label]
    # -> a ~1.1B-parameter layer that needs >10 GB (weights + Adam) and cannot fit
    # in 8 GB. We instead inject noise+label as a small additive projection onto the
    # structure bottleneck. The structure-conditioning method is unchanged; only the
    # (wasteful) fusion layer shrinks from ~1.1B to ~8.4M params. [8 GB VRAM fix]
    le = layers.LeakyReLU(0.2)(layers.Dense(128, use_bias=False)(l))
    z = layers.Dense(8 * 8 * 512, use_bias=False)(layers.Concatenate()([n, le]))
    z = layers.Reshape((8, 8, 512))(z)
    x = layers.Add()([e, z])
    x = layers.ReLU()(layers.BatchNormalization()(x))
    x = _gen_block(x, 256); x = _gen_block(x, 128); x = _gen_block(x, 64); x = _gen_block(x, 32)
    o = layers.Activation("tanh", dtype="float32")(
        layers.Conv2D(1, 3, padding="same", dtype="float32")(x))
    return tf.keras.Model([c, l, n], o, name="generator_C")


def build_discriminator(num_classes):
    SN = layers.SpectralNormalization
    img = tf.keras.Input((IMG_SIZE, IMG_SIZE, 1))
    c = tf.keras.Input((IMG_SIZE, IMG_SIZE, COND_CH))
    l = tf.keras.Input((num_classes,))
    lp = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(layers.Dense(IMG_SIZE * IMG_SIZE)(l))
    x = layers.Concatenate()([img, c, lp])
    for f in [64, 128, 256, 512, 512]:
        x = layers.LeakyReLU(0.2)(SN(layers.Conv2D(f, 4, 2, padding="same"))(x))
    o = layers.Dense(1, dtype="float32")(layers.Flatten()(x))
    return tf.keras.Model([img, c, l], o, name="discriminator_C")


def train():
    C.setup_speed()
    paths = C.make_dirs(C.DRIVE_BASE_C)

    X_tr, y_tr, X_ev, y_ev, enc = load_images_128(C.DATA_PATH)
    num_classes = len(enc.classes_)
    np.save(os.path.join(paths["ckpt"], "classes.npy"), enc.classes_)

    print("Building structure maps for training set...")
    # float16 to halve RAM (~9.5 GB -> ~4.75 GB for the full set); the mixed-precision
    # model casts inputs to float16 anyway, so this is lossless for training.
    C_tr = np.stack([structure_map(x) for x in tqdm(X_tr, desc="structure maps")]).astype(np.float16)
    # held-out structures too (saved for the generalization test)
    C_ev = np.stack([structure_map(x) for x in X_ev]).astype(np.float16)
    np.save(os.path.join(paths["ckpt"], "X_ev.npy"), X_ev)
    np.save(os.path.join(paths["ckpt"], "y_ev.npy"), y_ev)
    np.save(os.path.join(paths["ckpt"], "C_ev.npy"), C_ev)

    G = build_generator(num_classes)
    D = build_discriminator(num_classes)
    print(f"G params {G.count_params():,} | D params {D.count_params():,}")

    from train_utils import make_optimizer, scaled, apply_grads, set_lr
    g_opt = make_optimizer(C.LR_G)
    d_opt = make_optimizer(C.LR_D)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def onehot(y):
        return tf.one_hot(y, num_classes)

    AUTOTUNE = tf.data.AUTOTUNE
    # Pin the ~6.5 GB source tensors to CPU; otherwise from_tensor_slices tries to
    # copy the whole dataset onto the 8 GB GPU and fails ("Dst tensor is not
    # initialized"). tf.data then streams per-batch to the GPU.
    n_tr = len(X_tr)
    with tf.device("/CPU:0"):
        ds = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr, C_tr))
              .shuffle(min(n_tr, 8192), seed=RANDOM_SEED, reshuffle_each_iteration=True)
              .batch(C.BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE))
    # from_tensor_slices copied the arrays into the dataset -> free the numpy
    # originals (and the already-saved eval arrays) to avoid OOM at 16-18 GB.
    import gc
    del X_tr, C_tr, X_ev, C_ev, y_ev
    gc.collect()

    ckpt = tf.train.Checkpoint(generator=G, discriminator=D, g_opt=g_opt, d_opt=d_opt)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, paths["ckpt"], max_to_keep=5)
    start_ep = 0
    hist = {k: [] for k in ["d", "g", "g_adv", "g_l1"]}
    if os.path.exists(paths["progress"]):
        with open(paths["progress"]) as f:
            prog = json.load(f)
        start_ep = prog.get("last_epoch", 0)
        for k in hist:
            hist[k] = prog.get(k, [])
        if ckpt_mgr.latest_checkpoint:
            ckpt.restore(ckpt_mgr.latest_checkpoint)
            print(f"Resumed from epoch {start_ep}")

    @tf.function(jit_compile=C.USE_XLA)
    def train_step(real, y, cond):
        oh = onehot(y)
        nz = tf.random.normal([tf.shape(real)[0], Z_DIM])
        with tf.GradientTape() as t:
            fake = G([cond, oh, nz], training=True)
            d_real = D([real, cond, oh], training=True)
            d_fake = D([fake, cond, oh], training=True)
            d_loss = (bce(tf.ones_like(d_real) * C.LABEL_SMOOTH, d_real)
                      + bce(tf.zeros_like(d_fake), d_fake))
            d_loss_s = scaled(d_opt, d_loss)
        apply_grads(d_opt, t.gradient(d_loss_s, D.trainable_variables), D.trainable_variables)
        nz = tf.random.normal([tf.shape(real)[0], Z_DIM])
        with tf.GradientTape() as t:
            fake = G([cond, oh, nz], training=True)
            f = D([fake, cond, oh], training=True)
            g_adv = bce(tf.ones_like(f), f)
            g_l1 = LAMBDA_L1 * tf.reduce_mean(tf.abs(fake - tf.cast(real, fake.dtype)))
            g_loss = g_adv + g_l1
            g_loss_s = scaled(g_opt, g_loss)
        apply_grads(g_opt, t.gradient(g_loss_s, G.trainable_variables), G.trainable_variables)
        return d_loss, g_loss, g_adv, g_l1

    import time
    hist.setdefault("epoch_seconds", hist.get("epoch_seconds", []))
    for epoch in range(start_ep, C.EPOCHS):
        _t0 = time.time()
        if epoch == C.LR_DECAY_D:
            set_lr(d_opt, C.LR_D / 2)
        if epoch == C.LR_DECAY_G:
            set_lr(g_opt, C.LR_G / 2)
        dl = gl = ga = l1 = k = 0.0
        for real, y, cond in ds:
            d, g, a, p = train_step(real, y, cond)
            dl += float(d); gl += float(g); ga += float(a); l1 += float(p); k += 1
        hist["d"].append(dl / k); hist["g"].append(gl / k)
        hist["g_adv"].append(ga / k); hist["g_l1"].append(l1 / k)
        _el = time.time() - _t0
        hist["epoch_seconds"].append(round(_el, 1))
        print(f"Ep{epoch+1}/{C.EPOCHS}  D={dl/k:.4f}  G={gl/k:.4f}  "
              f"(adv={ga/k:.4f}  L1={l1/k:.4f})  {_el:.1f}s", flush=True)
        ckpt_mgr.save()
        with open(paths["progress"], "w") as f:
            json.dump({"last_epoch": epoch + 1, **hist}, f)
        G.save_weights(os.path.join(paths["ckpt"], "generator_C.weights.h5"))
        D.save_weights(os.path.join(paths["ckpt"], "discriminator_C.weights.h5"))

    # export inference-ready generator + label map
    exp = os.path.join(paths["ckpt"], "export"); os.makedirs(exp, exist_ok=True)
    G.save(os.path.join(exp, "generator.keras"))
    idx_to_label = {int(i): str(l) for i, l in enumerate(enc.classes_)}
    with open(os.path.join(exp, "class_labels.json"), "w") as f:
        json.dump({"idx_to_label": {str(k): v for k, v in idx_to_label.items()},
                   "label_to_idx": {v: k for k, v in idx_to_label.items()}}, f, indent=2)
    with open(os.path.join(exp, "inference_config.json"), "w") as f:
        json.dump({"Z_DIM": Z_DIM, "IMG_SIZE": IMG_SIZE, "COND_CH": COND_CH,
                   "num_classes": num_classes, "conditioned_on_structure": True}, f, indent=2)
    print("Training complete. Exported to", exp)
    return G, D, hist, enc


if __name__ == "__main__":
    train()
