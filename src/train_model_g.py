"""
Model G — Model F + four literature-backed upgrades for higher recognition
          (GAN-test) accuracy, all engineered to fit the 8 GB RTX 3050.

Model F = structure-conditioned cGAN (Canny+silhouette+distance) + aligned L1 +
frozen-landmark-regressor consistency. F reaches ~86% recognition; the ceiling
(a real-trained classifier on real images) is ~97%. The remaining gap is
image-quality / class-fidelity, so Model G adds signals that target exactly that:

  1. AUX-CLASSIFIER recognition loss (AC-GAN, Odena 2017): a classifier trained
     ONLY on REAL images (separate seed + light augmentation, kept independent
     from paper_eval's evaluator) scores each fake; cross-entropy pushes G to
     make class-discriminative images -> directly optimizes what "recognition"
     measures.
  2. DISCRIMINATOR FEATURE-MATCHING loss (pix2pixHD, Wang 2018): L1 between D's
     intermediate activations of fake vs the ALIGNED real target -> kills the
     blur that a real-trained classifier penalizes. Reuses D, no extra model.
  3. LANDMARK loss upgraded: L1 (not MSE, which vanishes as it saturates),
     larger weight, earlier warmup -> the term that already drove C->F.
  4. GENERATOR EMA (Yazici 2019): exponential moving average of G weights,
     exported as the inference generator -> reliable quality/stability gain.

Memory plan for 8 GB: real-side targets (D features, landmarks) are computed
OUTSIDE the GradientTape so their activations are never taped; only the fake
forward passes are retained. Aux classifier + regressor are frozen (no optimizer
state). Batch stays 32 (override with ARASL_BATCH if OOM).

Run:  python src/train_model_g.py
"""
import os, json, time, gc
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import config as C
from config import IMG_SIZE, Z_DIM, RANDOM_SEED
# reuse F's data + structure-map pipeline verbatim (identical conditioning)
from train_model_f import load_images_128, structure_map, COND_CH, LAMBDA_L1
from paper_eval import build_classifier

# ── new-in-G hyperparameters (kept local so A/B/C/F configs are untouched) ────
LAMBDA_FM       = 10.0          # discriminator feature-matching weight
LAMBDA_CLS_END  = 1.0           # aux-classifier recognition weight (ramped)
LAMBDA_LM_END_G = 8.0           # landmark weight (L1 now, so bigger than F's MSE 2.0)
WARMUP_START_G  = 5             # epoch to start ramping cls + landmark
WARMUP_LEN_G    = 10            # ramp length (full strength at START+LEN)
EMA_DECAY       = 0.999
CLS_EPOCHS      = 12            # aux-classifier pretrain epochs (on REAL images)
CLS_SEED        = RANDOM_SEED + 1234   # independent from paper_eval's classifier


# ── models: F's generator verbatim + a discriminator that also emits features ─
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
    le = layers.LeakyReLU(0.2)(layers.Dense(128, use_bias=False)(l))
    z = layers.Dense(8 * 8 * 512, use_bias=False)(layers.Concatenate()([n, le]))
    z = layers.Reshape((8, 8, 512))(z)
    x = layers.Add()([e, z])
    x = layers.ReLU()(layers.BatchNormalization()(x))
    x = _gen_block(x, 256); x = _gen_block(x, 128); x = _gen_block(x, 64); x = _gen_block(x, 32)
    o = layers.Activation("tanh", dtype="float32")(
        layers.Conv2D(1, 3, padding="same", dtype="float32")(x))
    return tf.keras.Model([c, l, n], o, name="generator_G")


def build_discriminator(num_classes):
    """Same as F's D, but also returns two intermediate activations for the
    pix2pixHD feature-matching loss. Weights/architecture are unchanged."""
    SN = layers.SpectralNormalization
    img = tf.keras.Input((IMG_SIZE, IMG_SIZE, 1))
    c = tf.keras.Input((IMG_SIZE, IMG_SIZE, COND_CH))
    l = tf.keras.Input((num_classes,))
    lp = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(layers.Dense(IMG_SIZE * IMG_SIZE)(l))
    x = layers.Concatenate()([img, c, lp])
    feats = []
    for f in [64, 128, 256, 512, 512]:
        x = layers.LeakyReLU(0.2)(SN(layers.Conv2D(f, 4, 2, padding="same"))(x))
        feats.append(x)
    o = layers.Dense(1, dtype="float32")(layers.Flatten()(x))
    # expose the 16x16x256 and 8x8x512 activations as FM targets
    return tf.keras.Model([img, c, l], [o, feats[2], feats[3]], name="discriminator_G")


def train():
    C.setup_speed()
    paths = C.make_dirs(C.DRIVE_BASE_G)

    X_tr, y_tr, X_ev, y_ev, enc = load_images_128(C.DATA_PATH)
    num_classes = len(enc.classes_)
    np.save(os.path.join(paths["ckpt"], "classes.npy"), enc.classes_)

    # ---- (1) aux classifier: train on REAL images, light aug, independent seed ----
    print("Training auxiliary recognition classifier (real images, frozen for GAN)...")
    tf.random.set_seed(CLS_SEED)
    clf = build_classifier(num_classes)
    aug = tf.keras.Sequential([                      # robust to benign GAN texture
        tf.keras.Input((IMG_SIZE, IMG_SIZE, 1)),
        layers.RandomTranslation(0.06, 0.06, fill_mode="constant", fill_value=-1.0),
        layers.GaussianNoise(0.05),
    ], name="clf_aug")
    clf_train = tf.keras.Sequential([aug, clf])
    clf_train.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])
    clf_train.fit(X_tr.astype(np.float32), y_tr, epochs=CLS_EPOCHS, batch_size=64,
                  validation_data=(X_ev.astype(np.float32), y_ev), verbose=2)
    clf.trainable = False
    clf.save_weights(os.path.join(paths["ckpt"], "aux_classifier.weights.h5"))
    tf.random.set_seed(RANDOM_SEED)                  # restore determinism for the GAN

    print("Building structure maps for training set...")
    C_tr = np.stack([structure_map(x) for x in X_tr]).astype(np.float16)
    C_ev = np.stack([structure_map(x) for x in X_ev]).astype(np.float16)
    np.save(os.path.join(paths["ckpt"], "X_ev.npy"), X_ev)
    np.save(os.path.join(paths["ckpt"], "y_ev.npy"), y_ev)
    np.save(os.path.join(paths["ckpt"], "C_ev.npy"), C_ev)

    G = build_generator(num_classes)
    D = build_discriminator(num_classes)
    print(f"G params {G.count_params():,} | D params {D.count_params():,}")

    # ---- (3) Model B's frozen landmark regressor (same ingredient as F) ----
    from models import build_landmark_regressor
    R = build_landmark_regressor(C.IMG_SIZE)
    R.load_weights(os.path.join(C.DRIVE_BASE_B, "checkpoints", "landmark_regressor.weights.h5"))
    R.trainable = False
    print(f"Loaded frozen landmark regressor from Model B ({R.count_params():,} params)")

    # warmup schedules (cls + landmark share START/LEN; ramp 0 -> END)
    def ramp(ep, end):
        if ep < WARMUP_START_G:
            return 0.0
        t = (ep - WARMUP_START_G) / max(WARMUP_LEN_G, 1)
        return float(min(t, 1.0) * end)

    lam_lm_var  = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    lam_cls_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    from train_utils import make_optimizer, scaled, apply_grads, set_lr
    g_opt = make_optimizer(C.LR_G)
    d_opt = make_optimizer(C.LR_D)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def onehot(y):
        return tf.one_hot(y, num_classes)

    AUTOTUNE = tf.data.AUTOTUNE
    n_tr = len(X_tr)
    with tf.device("/CPU:0"):
        ds = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr, C_tr))
              .shuffle(min(n_tr, 8192), seed=RANDOM_SEED, reshuffle_each_iteration=True)
              .batch(C.BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE))
    del X_tr, C_tr, X_ev, C_ev, y_ev
    gc.collect()

    # ---- (4) EMA shadow of G's trainable weights ----
    ema_weights = [tf.Variable(w, trainable=False, name=f"ema_{i}")
                   for i, w in enumerate(G.trainable_variables)]

    @tf.function
    def ema_update():
        for e, w in zip(ema_weights, G.trainable_variables):
            e.assign(EMA_DECAY * e + (1.0 - EMA_DECAY) * tf.cast(w, e.dtype))

    ema_mod = tf.Module(); ema_mod.vars = ema_weights   # tf.Module auto-tracks the list
    ckpt = tf.train.Checkpoint(generator=G, discriminator=D, g_opt=g_opt, d_opt=d_opt, ema=ema_mod)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, paths["ckpt"], max_to_keep=5)
    start_ep = 0
    hist = {k: [] for k in ["d", "g", "g_adv", "g_l1", "g_lm", "g_fm", "g_cls", "epoch_seconds"]}
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
        realf = tf.cast(real, tf.float32)
        # ---- real-side targets computed OUTSIDE the tape (no taped activations) ----
        _, fr2, fr3 = D([real, cond, oh], training=False)
        # D activations are float16 under mixed precision; cast targets to float32
        # so the feature-matching subtraction matches the fake side.
        fr2 = tf.stop_gradient(tf.cast(fr2, tf.float32))
        fr3 = tf.stop_gradient(tf.cast(fr3, tf.float32))
        lm_real = tf.stop_gradient(R(realf, training=False))

        # ---- D step ----
        nz = tf.random.normal([tf.shape(real)[0], Z_DIM])
        with tf.GradientTape() as t:
            fake = G([cond, oh, nz], training=True)
            d_real, _, _ = D([real, cond, oh], training=True)
            d_fake, _, _ = D([fake, cond, oh], training=True)
            d_loss = (bce(tf.ones_like(d_real) * C.LABEL_SMOOTH, d_real)
                      + bce(tf.zeros_like(d_fake), d_fake))
            d_loss_s = scaled(d_opt, d_loss)
        apply_grads(d_opt, t.gradient(d_loss_s, D.trainable_variables), D.trainable_variables)

        # ---- G step ----
        nz = tf.random.normal([tf.shape(real)[0], Z_DIM])
        with tf.GradientTape() as t:
            fake = G([cond, oh, nz], training=True)
            f_logit, ff2, ff3 = D([fake, cond, oh], training=True)
            g_adv = bce(tf.ones_like(f_logit), f_logit)
            g_l1 = LAMBDA_L1 * tf.reduce_mean(tf.abs(fake - realf))
            # (3) landmark consistency, now L1 (persistent gradient vs MSE)
            lm_fake = R(fake, training=False)
            g_lm = tf.reduce_mean(tf.abs(lm_fake - lm_real))
            # (2) feature matching against the aligned real target
            g_fm = LAMBDA_FM * (tf.reduce_mean(tf.abs(tf.cast(ff2, tf.float32) - fr2))
                                + tf.reduce_mean(tf.abs(tf.cast(ff3, tf.float32) - fr3)))
            # (1) auxiliary recognition loss (frozen real-trained classifier)
            g_cls = scce(y, clf(fake, training=False))
            g_loss = g_adv + g_l1 + lam_lm_var * g_lm + g_fm + lam_cls_var * g_cls
            g_loss_s = scaled(g_opt, g_loss)
        apply_grads(g_opt, t.gradient(g_loss_s, G.trainable_variables), G.trainable_variables)
        ema_update()
        return d_loss, g_loss, g_adv, g_l1, g_lm, g_fm, g_cls

    for epoch in range(start_ep, C.EPOCHS):
        _t0 = time.time()
        if epoch == C.LR_DECAY_D:
            set_lr(d_opt, C.LR_D / 2)
        if epoch == C.LR_DECAY_G:
            set_lr(g_opt, C.LR_G / 2)
        lam_lm_var.assign(ramp(epoch, LAMBDA_LM_END_G))
        lam_cls_var.assign(ramp(epoch, LAMBDA_CLS_END))
        dl = gl = ga = l1 = lm = fm = cls = k = 0.0
        for real, y, cond in ds:
            d, g, a, p, ml, fmv, cv = train_step(real, y, cond)
            dl += float(d); gl += float(g); ga += float(a); l1 += float(p)
            lm += float(ml); fm += float(fmv); cls += float(cv); k += 1
        for key, val in [("d", dl), ("g", gl), ("g_adv", ga), ("g_l1", l1),
                         ("g_lm", lm), ("g_fm", fm), ("g_cls", cls)]:
            hist[key].append(val / k)
        _el = time.time() - _t0
        hist["epoch_seconds"].append(round(_el, 1))
        print(f"Ep{epoch+1}/{C.EPOCHS}  D={dl/k:.4f}  G={gl/k:.4f}  (adv={ga/k:.4f}  "
              f"L1={l1/k:.4f}  LM={lm/k:.4f}  FM={fm/k:.4f}  CLS={cls/k:.4f}  "
              f"lam_lm={ramp(epoch,LAMBDA_LM_END_G):.2f} lam_cls={ramp(epoch,LAMBDA_CLS_END):.2f})  {_el:.1f}s", flush=True)
        ckpt_mgr.save()
        with open(paths["progress"], "w") as f:
            json.dump({"last_epoch": epoch + 1, **hist}, f)
        G.save_weights(os.path.join(paths["ckpt"], "generator_G.weights.h5"))
        D.save_weights(os.path.join(paths["ckpt"], "discriminator_G.weights.h5"))

    # ---- export the EMA generator (better than the raw weights) ----
    for w, e in zip(G.trainable_variables, ema_weights):
        w.assign(tf.cast(e, w.dtype))
    exp = os.path.join(paths["ckpt"], "export"); os.makedirs(exp, exist_ok=True)
    G.save(os.path.join(exp, "generator.keras"))
    idx_to_label = {int(i): str(l) for i, l in enumerate(enc.classes_)}
    with open(os.path.join(exp, "class_labels.json"), "w") as f:
        json.dump({"idx_to_label": {str(k): v for k, v in idx_to_label.items()},
                   "label_to_idx": {v: k for k, v in idx_to_label.items()}}, f, indent=2)
    with open(os.path.join(exp, "inference_config.json"), "w") as f:
        json.dump({"Z_DIM": Z_DIM, "IMG_SIZE": IMG_SIZE, "COND_CH": COND_CH,
                   "num_classes": num_classes, "conditioned_on_structure": True,
                   "landmark_loss": True, "feature_matching": True,
                   "aux_classifier": True, "generator_ema": True}, f, indent=2)
    print("Training complete. Exported EMA generator to", exp)
    return G, D, hist, enc


if __name__ == "__main__":
    train()
