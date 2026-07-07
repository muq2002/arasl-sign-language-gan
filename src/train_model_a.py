"""
Model A — pure cGAN (no MediaPipe), speed-optimized.

Loss math is IDENTICAL to the original notebook:
    G_total = g_adv + lambda(epoch) * mean(|fake - structural_target|)

Speedups (accuracy-neutral): mixed precision + loss scaling, tf.data prefetch,
in-graph one-hot, vectorized per-epoch target selection, optional XLA.

Run on Colab:
    !python src/train_model_a.py
or import train() from a notebook.
"""
import os, json, time
import numpy as np
import tensorflow as tf

import config as C
from data import load_images, build_pipeline, AUTOTUNE
from models import build_generator, build_discriminator
from train_utils import make_optimizer, scaled, apply_grads, set_lr


def get_lambda(epoch):
    if epoch >= C.WARMUP_EP:
        return float(C.LAMBDA_PIX_END)
    t = epoch / max(C.WARMUP_EP, 1)
    return float(C.LAMBDA_PIX_START + t * (C.LAMBDA_PIX_END - C.LAMBDA_PIX_START))


def train():
    C.setup_speed()
    paths = C.make_dirs(C.DRIVE_BASE_A)

    images, labels_int, enc, prototypes = load_images()
    num_classes = len(enc.classes_)
    np.save(os.path.join(paths["ckpt"], "classes.npy"), enc.classes_)

    G = build_generator(C.Z_DIM, num_classes)
    D = build_discriminator(num_classes)
    print(f"G params {G.count_params():,} | D params {D.count_params():,}")

    g_opt = make_optimizer(C.LR_G)
    d_opt = make_optimizer(C.LR_D)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    ds, target_idx_var, selector = build_pipeline(
        images, labels_int, prototypes, num_classes)

    lam_v = tf.Variable(get_lambda(0), dtype=tf.float32, trainable=False)

    ckpt = tf.train.Checkpoint(generator=G, discriminator=D, g_opt=g_opt, d_opt=d_opt)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, paths["ckpt"], max_to_keep=5)
    start_ep = 0
    hist = {k: [] for k in ["d", "g_adv", "g_pixel", "g_total", "gd_ratio", "lam"]}
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
            d_loss_s = scaled(d_opt, d_loss)
        apply_grads(d_opt, t.gradient(d_loss_s, D.trainable_variables), D.trainable_variables)
        return d_loss

    @tf.function(jit_compile=C.USE_XLA)
    def step_g(real, lbs, tgt):
        noise = tf.random.normal([tf.shape(real)[0], C.Z_DIM])
        with tf.GradientTape() as t:
            fake = G([noise, lbs], training=True)
            f = D([fake, lbs], training=True)
            g_adv = bce(tf.ones_like(f), f)
            g_pix = tf.reduce_mean(tf.abs(fake - tgt))
            g_loss = g_adv + lam_v * g_pix
            g_loss_s = scaled(g_opt, g_loss)
        apply_grads(g_opt, t.gradient(g_loss_s, G.trainable_variables), G.trainable_variables)
        return g_adv, g_pix

    g_n = C.G_UPDATES_BASE
    hist.setdefault("epoch_seconds", hist.get("epoch_seconds", []))
    for epoch in range(start_ep, C.EPOCHS):
        _t0 = time.time()
        if epoch == C.LR_DECAY_D:
            set_lr(d_opt, C.LR_D / 2)
        if epoch == C.LR_DECAY_G:
            set_lr(g_opt, C.LR_G / 2)

        lam = get_lambda(epoch); lam_v.assign(lam)
        rng = np.random.default_rng(C.RANDOM_SEED + epoch)
        target_idx_var.assign(selector.epoch_targets(epoch, C.PHASE2_EP, rng))

        ep_d, ep_ga, ep_gp = [], [], []
        for real, lbs, tgt in ds:
            ep_d.append(float(step_d(real, lbs)))
            ga = gp = 0.0
            for _ in range(g_n):
                a, p = step_g(real, lbs, tgt)
                ga += float(a); gp += float(p)
            ep_ga.append(ga / g_n); ep_gp.append(gp / g_n)

        dm, gam, gpm = map(lambda v: float(np.mean(v)), (ep_d, ep_ga, ep_gp))
        gtm = gam + lam * gpm
        gd = gtm / max(dm, 0.01)
        for k, v in zip(["d", "g_adv", "g_pixel", "g_total", "gd_ratio", "lam"],
                        [dm, gam, gpm, gtm, gd, lam]):
            hist[k].append(v)
        _el = time.time() - _t0
        hist["epoch_seconds"].append(round(_el, 1))
        phase = "P1" if epoch < C.PHASE2_EP else "P2"
        print(f"Ep{epoch+1}/{C.EPOCHS} [{phase}] D={dm:.4f} Ga={gam:.4f} "
              f"Gp={gpm:.4f} Gt={gtm:.4f} G/D={gd:.2f}x lam={lam:.2f} nG={g_n} {_el:.1f}s", flush=True)

        g_n = (min(C.G_UPDATES_BASE + 2, 5) if gd > C.G_D_RATIO_MAX * 1.5
               else C.G_UPDATES_BASE + 1 if gd > C.G_D_RATIO_MAX
               else C.G_UPDATES_BASE)

        ckpt_mgr.save()
        with open(paths["progress"], "w") as f:
            json.dump({"last_epoch": epoch + 1, **hist}, f)
        G.save_weights(os.path.join(paths["ckpt"], "generator.weights.h5"))
        D.save_weights(os.path.join(paths["ckpt"], "discriminator.weights.h5"))

    print("Training complete.")
    return G, D, hist, enc


if __name__ == "__main__":
    train()
