"""
Speed benchmark: OLD (float32) vs OPTIMIZED settings, measured on YOUR hardware.

It uses SYNTHETIC data (no dataset needed), so you can run it in ~1 minute:

    python src/benchmark.py
    # or in a notebook:  import sys; sys.path.insert(0,'src'); import benchmark; benchmark.main()

It reports two things:
  1. Training throughput (images/sec) for: float32  |  mixed_float16  |  mixed_float16+XLA
  2. MediaPipe extraction: per-call Hands() (old)  vs  reused Hands() (optimized)

Nothing here touches the loss math — it measures the execution speed of the
exact same forward/backward work under each setting.
"""
import time
import numpy as np
import tensorflow as tf

from config import Z_DIM, IMG_SIZE, BATCH_SIZE, LR_G, LR_D, LABEL_SMOOTH
from models import build_generator, build_discriminator

NUM_CLASSES = 32
WARMUP = 5
ITERS = 30


def _make_opt(lr, mixed):
    opt = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999, clipnorm=1.0)
    return tf.keras.mixed_precision.LossScaleOptimizer(opt) if mixed else opt


def _apply(opt, loss, tape, vs, mixed):
    if mixed:
        g = opt.get_unscaled_gradients(tape.gradient(opt.get_scaled_loss(loss), vs))
    else:
        g = tape.gradient(loss, vs)
    opt.apply_gradients(zip(g, vs))


def bench_training(mixed=False, xla=False):
    """Time D-step + 2x G-step under one precision/XLA setting. Returns images/sec."""
    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy("mixed_float16" if mixed else "float32")

    G = build_generator(Z_DIM, NUM_CLASSES)
    D = build_discriminator(NUM_CLASSES)
    g_opt, d_opt = _make_opt(LR_G, mixed), _make_opt(LR_D, mixed)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real = tf.constant(np.random.randn(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1).astype("float32"))
    lbs = tf.one_hot(np.random.randint(0, NUM_CLASSES, BATCH_SIZE), NUM_CLASSES)
    tgt = tf.constant(np.random.randn(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1).astype("float32"))

    @tf.function(jit_compile=xla)
    def step_d():
        noise = tf.random.normal([BATCH_SIZE, Z_DIM])
        with tf.GradientTape() as t:
            fake = G([noise, lbs], training=True)
            r = D([real, lbs], training=True); f = D([fake, lbs], training=True)
            loss = bce(tf.ones_like(r) * LABEL_SMOOTH, r) + bce(tf.zeros_like(f), f)
        _apply(d_opt, loss, t, D.trainable_variables, mixed)
        return loss

    @tf.function(jit_compile=xla)
    def step_g():
        noise = tf.random.normal([BATCH_SIZE, Z_DIM])
        with tf.GradientTape() as t:
            fake = G([noise, lbs], training=True)
            f = D([fake, lbs], training=True)
            loss = bce(tf.ones_like(f), f) + 5.0 * tf.reduce_mean(tf.abs(fake - tgt))
        _apply(g_opt, loss, t, G.trainable_variables, mixed)
        return loss

    for _ in range(WARMUP):
        step_d(); step_g(); step_g()
    float(step_g())  # sync

    t0 = time.perf_counter()
    for _ in range(ITERS):
        step_d(); step_g(); step_g()
    float(step_g())  # force sync before stopping clock
    dt = time.perf_counter() - t0

    imgs = ITERS * BATCH_SIZE
    return imgs / dt, dt / ITERS * 1000.0  # images/sec, ms/iter


def bench_mediapipe(n=60):
    """Old (rebuild Hands per call) vs optimized (reuse). Returns (old_s, new_s) or None."""
    try:
        import mediapipe as mp
        from mediapipe_utils import HandLandmarkExtractor, _preprocess, _gray_to_clahe_skintone
    except Exception as e:
        print(f"  (skipped MediaPipe bench: {e})")
        return None

    imgs = [np.random.rand(IMG_SIZE, IMG_SIZE, 1).astype("float32") * 2 - 1 for _ in range(n)]
    H = mp.solutions.hands.Hands

    t0 = time.perf_counter()
    for im in imgs:
        rgb = _preprocess(((im[:, :, 0] + 1) * 127.5).astype("uint8"), 256, _gray_to_clahe_skintone)
        with H(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3, model_complexity=1) as h:
            h.process(rgb)
    old = time.perf_counter() - t0

    t0 = time.perf_counter()
    with HandLandmarkExtractor() as ext:
        for im in imgs:
            ext.extract(im)
    new = time.perf_counter() - t0
    return old, new


def main():
    print("=" * 64)
    print(f"  SPEED BENCHMARK   batch={BATCH_SIZE}  iters={ITERS}  img={IMG_SIZE}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  Device: {'GPU ' + gpus[0].name if gpus else 'CPU (mixed precision will NOT help)'}")
    print("=" * 64)

    print("\n[1] Training throughput (D-step + 2x G-step per iter)")
    base_ips, base_ms = bench_training(mixed=False, xla=False)
    print(f"  OLD  float32           : {base_ips:8.1f} img/s   ({base_ms:6.1f} ms/iter)  1.00x")

    mp_ips, mp_ms = bench_training(mixed=True, xla=False)
    print(f"  OPT  mixed_float16     : {mp_ips:8.1f} img/s   ({mp_ms:6.1f} ms/iter)  {mp_ips/base_ips:.2f}x")

    try:
        x_ips, x_ms = bench_training(mixed=True, xla=True)
        print(f"  OPT  mixed_f16 + XLA   : {x_ips:8.1f} img/s   ({x_ms:6.1f} ms/iter)  {x_ips/base_ips:.2f}x")
    except Exception as e:
        print(f"  OPT  mixed_f16 + XLA   : failed on this GPU ({type(e).__name__}) -> keep USE_XLA=False")

    print("\n[2] MediaPipe extraction (Model B preprocessing)")
    r = bench_mediapipe(n=60)
    if r:
        old, new = r
        print(f"  OLD  rebuild per call  : {old:7.2f} s  for 60 imgs  ({old/60*1000:.1f} ms/img)")
        print(f"  OPT  reused instance   : {new:7.2f} s  for 60 imgs  ({new/60*1000:.1f} ms/img)  {old/new:.1f}x")
        print(f"  -> extrapolated to 54k imgs: OLD ~{old/60*54000/60:.0f} min  vs  OPT ~{new/60*54000/60:.0f} min")

    print("\nNote: numbers are hardware-specific. Mixed precision needs a GPU with")
    print("tensor cores (T4/V100/A100) to show its full speedup.")


if __name__ == "__main__":
    main()
