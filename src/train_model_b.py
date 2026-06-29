"""
Model B — cGAN + MediaPipe landmark supervision, speed-optimized.

Loss math is IDENTICAL to the original "FIXED" notebook:
    G_total = g_adv
            + lambda_pix(epoch) * mean(|fake - structural_target|)
            + lambda_lm(epoch)  * mean( (regressor(fake) - real_lm)^2 * valid_mask )
            - LAMBDA_DIV * mean(variance(fake))

Speedups (accuracy-neutral): reused MediaPipe instances (huge for extraction),
mixed precision + loss scaling, tf.data prefetch, in-graph one-hot, vectorized
per-epoch target selection, optional XLA.
"""
import os, json
import numpy as np
import tensorflow as tf

import config as C
from data import load_images, build_pipeline
from models import build_generator, build_discriminator, build_landmark_regressor
from train_utils import make_optimizer, apply_loss, set_lr
from mediapipe_utils import compute_landmarks


def get_lambda_pix(epoch):
    if epoch >= C.WARMUP_EP:
        return float(C.LAMBDA_PIX_END)
    t = epoch / max(C.WARMUP_EP, 1)
    return float(C.LAMBDA_PIX_START + t * (C.LAMBDA_PIX_END - C.LAMBDA_PIX_START))


def get_lambda_lm(epoch):
    if epoch < C.WARMUP_LM_EP:
        return 0.0
    t = (epoch - C.WARMUP_LM_EP) / max(C.WARMUP_EP, 1)
    return float(min(C.LAMBDA_LM_START + t * (C.LAMBDA_LM_END - C.LAMBDA_LM_START),
                     C.LAMBDA_LM_END))


def train_regressor(reg, images, landmarks, save_path):
    """Train on images where MediaPipe succeeded, then freeze. (compile auto-handles
    loss scaling under the mixed_float16 policy.)"""
    if os.path.exists(save_path):
        reg.build((None, C.IMG_SIZE, C.IMG_SIZE, 1))
        reg.load_weights(save_path)
        reg.trainable = False
        print(f"Regressor loaded (frozen): {save_path}")
        return reg
    valid = landmarks.any(axis=1)
    x, y = images[valid], np.clip(landmarks[valid], 0.0, 1.0)
    print(f"Training regressor on {len(x)} imgs ({valid.mean()*100:.1f}% of data)")
    reg.compile(optimizer=tf.keras.optimizers.Adam(C.REG_LR), loss="mse", metrics=["mae"])
    reg.fit(x, y, epochs=C.REG_EPOCHS, batch_size=C.REG_BATCH_SIZE, validation_split=0.1,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, verbose=1),
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                 restore_best_weights=True, verbose=1)],
            verbose=1)
    reg.save_weights(save_path)
    reg.trainable = False
    print("Regressor frozen.")
    return reg


def train():
    C.setup_speed()
    paths = C.make_dirs(C.DRIVE_BASE_B)

    images, labels_int, enc, prototypes = load_images()
    num_classes = len(enc.classes_)
    np.save(os.path.join(paths["ckpt"], "classes.npy"), enc.classes_)

    cache = os.path.join(C.DRIVE_BASE_B, "landmarks_128px.npy")
    landmarks = compute_landmarks(images, cache)

    G = build_generator(C.Z_DIM, num_classes)
    D = build_discriminator(num_classes)
    reg = build_landmark_regressor(C.IMG_SIZE)
    reg = train_regressor(reg, images, landmarks, paths["regressor"])
    print(f"G {G.count_params():,} | D {D.count_params():,} | Reg {reg.count_params():,}")

    g_opt = make_optimizer(C.LR_G)
    d_opt = make_optimizer(C.LR_D)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    ds, target_idx_var, selector = build_pipeline(
        images, labels_int, prototypes, num_classes, landmarks=landmarks)

    lam_pix_v = tf.Variable(get_lambda_pix(0), dtype=tf.float32, trainable=False)
    lam_lm_v  = tf.Variable(get_lambda_lm(0),  dtype=tf.float32, trainable=False)

    ckpt = tf.train.Checkpoint(generator=G, discriminator=D, g_opt=g_opt, d_opt=d_opt)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, paths["ckpt"], max_to_keep=5)
    start_ep = 0
    keys = ["d", "g_adv", "g_pix", "g_lm", "g_total", "gd_ratio", "lam_pix", "lam_lm"]
    hist = {k: [] for k in keys}
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
    def step_d(real, lbs):
        noise = tf.random.normal([tf.shape(real)[0], C.Z_DIM])
        with tf.GradientTape() as t:
            fake = G([noise, lbs], training=True)
            r = D([real, lbs], training=True)
            f = D([fake, lbs], training=True)
            d_loss = bce(tf.ones_like(r) * C.LABEL_SMOOTH, r) + bce(tf.zeros_like(f), f)
        apply_loss(d_opt, d_loss, t, D.trainable_variables)
        return d_loss

    @tf.function(jit_compile=C.USE_XLA)
    def step_g(real, lbs, tgt, lm_tgt):
        noise = tf.random.normal([tf.shape(real)[0], C.Z_DIM])
        with tf.GradientTape() as t:
            fake = G([noise, lbs], training=True)
            f = D([fake, lbs], training=True)
            g_adv = bce(tf.ones_like(f), f)
            g_pix = tf.reduce_mean(tf.abs(fake - tgt))
            pred_lm = reg(fake, training=False)
            valid = tf.cast(tf.reduce_any(lm_tgt != 0.0, axis=1, keepdims=True), tf.float32)
            g_lm = tf.reduce_mean(tf.square(pred_lm - lm_tgt) * valid)
            flat = tf.reshape(fake, [tf.shape(fake)[0], -1])
            div = -C.LAMBDA_DIV * tf.reduce_mean(tf.math.reduce_variance(flat, axis=1))
            g_loss = g_adv + lam_pix_v * g_pix + lam_lm_v * g_lm + div
        apply_loss(g_opt, g_loss, t, G.trainable_variables)
        return g_adv, g_pix, g_lm

    g_n = C.G_UPDATES_BASE
    for epoch in range(start_ep, C.EPOCHS):
        if epoch == C.LR_DECAY_D:
            set_lr(d_opt, C.LR_D / 2)
        if epoch == C.LR_DECAY_G:
            set_lr(g_opt, C.LR_G / 2)

        lp, ll = get_lambda_pix(epoch), get_lambda_lm(epoch)
        lam_pix_v.assign(lp); lam_lm_v.assign(ll)
        rng = np.random.default_rng(C.RANDOM_SEED + epoch)
        target_idx_var.assign(selector.epoch_targets(epoch, C.PHASE2_EP, rng))

        ep_d, ep_ga, ep_gp, ep_glm = [], [], [], []
        for real, lbs, tgt, lm_tgt in ds:
            ep_d.append(float(step_d(real, lbs)))
            ga = gp = glm = 0.0
            for _ in range(g_n):
                a, p, l = step_g(real, lbs, tgt, lm_tgt)
                ga += float(a); gp += float(p); glm += float(l)
            ep_ga.append(ga / g_n); ep_gp.append(gp / g_n); ep_glm.append(glm / g_n)

        dm, gam, gpm, glmm = map(lambda v: float(np.mean(v)), (ep_d, ep_ga, ep_gp, ep_glm))
        gtm = gam + lp * gpm + ll * glmm
        gd = gtm / max(dm, 0.01)
        for k, v in zip(keys, [dm, gam, gpm, glmm, gtm, gd, lp, ll]):
            hist[k].append(v)
        phase = "P1" if epoch < C.PHASE2_EP else "P2"
        lmp = "LM-off" if ll == 0 else f"LM-on({ll:.2f})"
        print(f"Ep{epoch+1}/{C.EPOCHS} [{phase}][{lmp}] D={dm:.4f} Ga={gam:.4f} "
              f"Gpix={gpm:.4f} Glm={glmm:.4f} Gt={gtm:.4f} G/D={gd:.2f}x nG={g_n}")

        g_n = (min(C.G_UPDATES_BASE + 2, 5) if gd > C.G_D_RATIO_MAX * 1.5
               else C.G_UPDATES_BASE + 1 if gd > C.G_D_RATIO_MAX
               else C.G_UPDATES_BASE)

        ckpt_mgr.save()
        with open(paths["progress"], "w") as f:
            json.dump({"last_epoch": epoch + 1, **hist}, f)
        G.save_weights(os.path.join(paths["ckpt"], "generator.weights.h5"))
        D.save_weights(os.path.join(paths["ckpt"], "discriminator.weights.h5"))

    print("Training complete.")
    return G, D, reg, hist, enc


if __name__ == "__main__":
    train()
