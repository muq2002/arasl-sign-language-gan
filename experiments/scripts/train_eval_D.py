# -*- coding: utf-8 -*-
"""Model D — structure-conditioned DIFFUSION with classifier-free guidance (CFG).

Different idea from the GANs (A/B/C): instead of a single-shot generator, D learns
an iterative denoiser. It KEEPS Model C's winning insight — condition on a per-image
structure map (Canny+silhouette+distance, aligned target) — but:

  * trains a small time-conditioned U-Net to predict noise (DDPM, simple MSE loss),
  * concatenates the structure map to the noisy image at the input (ControlNet-lite),
  * adds the class via the time embedding,
  * uses classifier-free guidance (CFG): the label is dropped ~10% of steps so the
    model jointly learns conditional + unconditional scores; at sampling we push
    toward the class with a guidance scale `w`. CFG is the accuracy dial.
  * samples with DDIM (few steps) so the reduced CPU run stays tractable.

Reuses the SAME prepared arrays + recognition/diversity metric as train_eval.py so
the numbers sit in the same table as A/B/C. Run in the TF venv on CPU.
"""
import time, json, os, warnings, numpy as np
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image

IMG_SIZE, BATCH, EPOCHS, SEED = 64, 16, 10, 42
T_TRAIN   = 1000          # diffusion steps (training schedule)
DDIM_STEPS = 30           # sampling steps (eval) — kept low for the CPU run
P_UNCOND  = 0.1           # label-dropout prob for classifier-free guidance
CFG_EVAL  = 3.0           # guidance scale used for the headline recognition number
CFG_SWEEP = [0.0, 1.0, 3.0, 5.0]
N_EVAL_PER = 16           # samples/class during eval (directional; raise for full run)
np.random.seed(SEED); tf.random.set_seed(SEED)

# paths: scripts/ -> experiments/{arrays,results,visualizations}
_S = os.path.dirname(os.path.abspath(__file__)); _E = os.path.dirname(_S)
ARR = os.path.join(_E, "arrays"); RES = os.path.join(_E, "results"); VIZ = os.path.join(_E, "visualizations", "assets")
for _d in (RES, VIZ): os.makedirs(_d, exist_ok=True)
A = lambda f: os.path.join(ARR, f); R = lambda f: os.path.join(RES, f); V = lambda f: os.path.join(VIZ, f)

Xtr = np.load(A("Xtr.npy")); ytr = np.load(A("ytr.npy"))
Xev = np.load(A("Xev.npy")); yev = np.load(A("yev.npy"))
Ctr = np.load(A("Ctr.npy"))
meta = json.load(open(A("prep_meta.json")))
num_classes = len(meta["classes"])
NULL_CLASS = num_classes                       # extra "unconditional" embedding row
print(f"TF {tf.__version__} CPU | {num_classes} classes | {len(Xtr)} train images | Model D (diffusion+CFG)")

# ── Diffusion schedule (cosine betas) ───────────────────────────────────────
def cosine_alpha_bar(T):
    s = 0.008
    t = np.linspace(0, T, T + 1) / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    betas = np.clip(1 - ab[1:] / ab[:-1], 1e-4, 0.999)
    return betas.astype(np.float32)

betas = cosine_alpha_bar(T_TRAIN)
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas).astype(np.float32)
ab_tf = tf.constant(alpha_bar)

EMB_DIM = 128
def sinusoidal_emb(t, dim=EMB_DIM):
    half = dim // 2
    freqs = tf.exp(-np.log(10000.0) * tf.range(half, dtype=tf.float32) / (half - 1))
    args = tf.cast(t, tf.float32)[:, None] * freqs[None, :]
    return tf.concat([tf.sin(args), tf.cos(args)], -1)

# ── Time/class-conditioned U-Net (structure map concatenated at input) ──────
def conv(x, f): return layers.Activation("swish")(layers.GroupNormalization(8)(layers.Conv2D(f, 3, padding="same")(x)))

def build_unet():
    img = tf.keras.Input((IMG_SIZE, IMG_SIZE, 1))            # noisy image x_t
    cond = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))           # structure map (ControlNet-lite)
    t = tf.keras.Input((), dtype=tf.int32)                   # timestep
    y = tf.keras.Input((), dtype=tf.int32)                   # class id (num_classes = null)
    temb = layers.Lambda(lambda z: sinusoidal_emb(z, EMB_DIM), output_shape=(EMB_DIM,))(t)
    temb = layers.Dense(EMB_DIM, activation="swish")(temb)
    yemb = layers.Embedding(num_classes + 1, EMB_DIM)(y)
    emb = layers.Dense(EMB_DIM, activation="swish")(layers.Add()([temb, yemb]))

    def add_emb(h, f):
        e = layers.Reshape((1, 1, f))(layers.Dense(f)(emb))
        return layers.Lambda(lambda hw: hw[0] + hw[1])([h, e])

    x = layers.Concatenate()([img, cond])                   # 64x64x4
    x = conv(x, 32)
    # encoder
    d1 = add_emb(conv(x, 32), 32);  p1 = layers.AveragePooling2D(2)(d1)        # 32
    d2 = add_emb(conv(p1, 64), 64); p2 = layers.AveragePooling2D(2)(d2)        # 16
    d3 = add_emb(conv(p2, 128), 128); p3 = layers.AveragePooling2D(2)(d3)      # 8
    # bottleneck
    b = add_emb(conv(p3, 256), 256); b = conv(b, 256)
    # decoder (skip connections)
    u3 = layers.UpSampling2D()(b); u3 = add_emb(conv(layers.Concatenate()([u3, d3]), 128), 128)
    u2 = layers.UpSampling2D()(u3); u2 = add_emb(conv(layers.Concatenate()([u2, d2]), 64), 64)
    u1 = layers.UpSampling2D()(u2); u1 = add_emb(conv(layers.Concatenate()([u1, d1]), 32), 32)
    out = layers.Conv2D(1, 3, padding="same")(u1)           # predicts noise eps
    return tf.keras.Model([img, cond, t, y], out, name="unet_D")

def q_sample(x0, t, noise):
    ab = tf.gather(ab_tf, t)[:, None, None, None]
    return tf.sqrt(ab) * x0 + tf.sqrt(1.0 - ab) * noise

def train_D():
    t0 = time.time()
    net = build_unet()
    opt = tf.keras.optimizers.Adam(2e-4)
    n = len(Xtr); steps = n // BATCH

    @tf.function
    def step(x0, cond, y):
        bs = tf.shape(x0)[0]
        t = tf.random.uniform([bs], 0, T_TRAIN, dtype=tf.int32)
        noise = tf.random.normal(tf.shape(x0))
        xt = q_sample(x0, t, noise)
        # classifier-free guidance: drop the label with prob P_UNCOND -> null class
        drop = tf.cast(tf.random.uniform([bs]) < P_UNCOND, tf.int32)
        y_in = y * (1 - drop) + NULL_CLASS * drop
        with tf.GradientTape() as tape:
            pred = net([xt, cond, t, y_in], training=True)
            loss = tf.reduce_mean(tf.square(pred - noise))
        opt.apply_gradients(zip(tape.gradient(loss, net.trainable_variables), net.trainable_variables))
        return loss

    for ep in range(EPOCHS):
        idx = np.random.permutation(n)
        last = 0.0
        for s in range(steps):
            b = idx[s*BATCH:(s+1)*BATCH]
            last = float(step(tf.constant(Xtr[b]), tf.constant(Ctr[b]), tf.constant(ytr[b], tf.int32)))
        print(f"  [D] ep{ep+1}/{EPOCHS} loss={last:.4f}")
    el = time.time() - t0
    print(f"  [D] trained {el:.1f}s")
    return net, el

# ── DDIM sampling with classifier-free guidance ─────────────────────────────
def ddim_sample(net, cond, y, w=CFG_EVAL, steps=DDIM_STEPS, seed=0):
    bs = cond.shape[0]
    tf.random.set_seed(seed)
    x = tf.random.normal([bs, IMG_SIZE, IMG_SIZE, 1])
    y_cond = tf.constant(y, tf.int32)
    y_null = tf.fill([bs], NULL_CLASS)
    ts = np.linspace(T_TRAIN - 1, 0, steps).astype(np.int32)
    for i, tcur in enumerate(ts):
        tb = tf.fill([bs], int(tcur))
        eps_c = net([x, cond, tb, y_cond], training=False)
        if w != 0.0:
            eps_u = net([x, cond, tb, y_null], training=False)
            eps = eps_u + w * (eps_c - eps_u)       # CFG
        else:
            eps = eps_c
        ab_t = alpha_bar[tcur]
        x0 = (x - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t)
        x0 = tf.clip_by_value(x0, -1.0, 1.0)
        if i < len(ts) - 1:
            ab_n = alpha_bar[ts[i + 1]]
            x = np.sqrt(ab_n) * x0 + np.sqrt(1 - ab_n) * eps
        else:
            x = x0
    return x.numpy()

# ── strong classifier for the recognition metric (trained on REAL) ──────────
tc = time.time()
clf = tf.keras.Sequential([tf.keras.Input((IMG_SIZE, IMG_SIZE, 1)),
    layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization(), layers.LeakyReLU(0.2), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same"), layers.BatchNormalization(), layers.LeakyReLU(0.2), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same"), layers.BatchNormalization(), layers.LeakyReLU(0.2), layers.MaxPooling2D(),
    layers.Flatten(), layers.Dense(128), layers.LeakyReLU(0.2), layers.Dropout(0.3),
    layers.Dense(num_classes)])
clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
clf.fit(Xtr, ytr, epochs=40, batch_size=32, verbose=0)
clf_time = time.time() - tc
real_acc = float(clf.evaluate(Xev, yev, verbose=0)[1])
print(f"classifier real eval-acc = {real_acc:.3f}  ({clf_time:.1f}s)")

def gen_D(net, w, n_per=N_EVAL_PER, seed=0):
    """Generate balanced samples from TRAINING structures at guidance scale w."""
    outs, ys = [], []
    for c in range(num_classes):
        pool = np.where(ytr == c)[0]
        ci = np.random.default_rng(seed + c).choice(pool, n_per)
        f = ddim_sample(net, tf.constant(Ctr[ci]), np.full(n_per, c), w=w, seed=seed*10 + c)
        outs.append(f); ys.append(np.full(n_per, c))
    return np.concatenate(outs), np.concatenate(ys)

def evaluate_D(net, w, n_per=N_EVAL_PER):
    f, ys = gen_D(net, w, n_per, 0)
    acc = float((clf.predict(f, verbose=0).argmax(1) == ys).mean())
    fl = f.reshape(len(f), -1)
    m = min(60, len(fl))
    div = float(np.mean([np.mean(np.abs(fl[i]-fl[j])) for i in range(m) for j in range(i+1, m)]))
    return {"recognition_acc": round(acc, 4), "diversity": round(div, 4)}

# ── sample-grid PNG ─────────────────────────────────────────────────────────
GC = min(5, num_classes); GS = 6
def to_u8(a): return (a[:, :, 0]*127.5+127.5).clip(0, 255).astype(np.uint8)
def save_grid(name, cells):
    H = W = IMG_SIZE; pad = 2
    grid = np.full((GC*(H+pad)+pad, GS*(W+pad)+pad), 30, np.uint8)
    for i, im in enumerate(cells):
        r, c = divmod(i, GS); y = pad+r*(H+pad); x = pad+c*(W+pad)
        grid[y:y+H, x:x+W] = im
    Image.fromarray(grid).save(V(f"{name}.png"))
def gen_cells_D(net, w):
    cells = []
    for c in range(GC):
        pool = np.where(ytr == c)[0]; ci = np.random.default_rng(c).choice(pool, GS)
        f = ddim_sample(net, tf.constant(Ctr[ci]), np.full(GS, c), w=w, seed=c)
        for j in range(GS): cells.append(to_u8(f[j]))
    return cells

# ── run ─────────────────────────────────────────────────────────────────────
print("=== Training Model D (structure-conditioned diffusion + CFG) ===")
net, el = train_D()

print("=== CFG sweep (recognition vs guidance scale) ===")
sweep = {}
for w in CFG_SWEEP:
    m = evaluate_D(net, w)
    sweep[str(w)] = m
    print(f"  w={w:>3}: recognition={m['recognition_acc']:.4f}  diversity={m['diversity']:.4f}")

best_w = max(sweep, key=lambda k: sweep[k]["recognition_acc"])
result_D = {"recognition_acc": sweep[str(CFG_EVAL)]["recognition_acc"],
            "diversity": sweep[str(CFG_EVAL)]["diversity"],
            "cfg_scale": CFG_EVAL, "best_cfg": float(best_w),
            "best_recognition": sweep[best_w]["recognition_acc"], "cfg_sweep": sweep}

save_grid("samples_D", gen_cells_D(net, CFG_EVAL))

print("\n================ MODEL D (diffusion+CFG) — 5K-image CPU run ================")
print(f"recognition @ w={CFG_EVAL}: {result_D['recognition_acc']}  | diversity: {result_D['diversity']}")
print(f"best recognition {result_D['best_recognition']} at CFG w={result_D['best_cfg']}")
print(f"classifier real-acc={real_acc:.3f} | {num_classes} classes | {len(Xtr)} train | {IMG_SIZE}px | {EPOCHS} ep | chance={1/num_classes:.2f}")
out = {"results": {"D": {k: result_D[k] for k in ("recognition_acc", "diversity")}},
       "model_D": result_D, "times": {"D": round(el, 1), "classifier": round(clf_time, 1)},
       "real_acc": real_acc, "meta": meta,
       "config": {"T_TRAIN": T_TRAIN, "DDIM_STEPS": DDIM_STEPS, "P_UNCOND": P_UNCOND,
                  "EPOCHS": EPOCHS, "CFG_EVAL": CFG_EVAL}}
json.dump(out, open(R("results_D.json"), "w"), indent=2)
print("saved results/results_D.json + visualizations/assets/samples_D.png")
