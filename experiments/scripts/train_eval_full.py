# -*- coding: utf-8 -*-
"""Comprehensive A/B/C evaluation + Model-C held-out structure test.
Run in the main (TF) venv after prep_data.py + extract_lm.py.

Adds vs train_eval.py:
  * GAN-test (classifier real -> generated)         [all]
  * GAN-train (classifier generated -> real eval)   [all]
  * diversity, per-class recognition                [all]
  * HELD-OUT STRUCTURE TEST (Model C):              [C]
      - generate from eval-set structure maps the generator never saw
      - recognition on held-out structures vs training structures (generalization gap)
      - paired global-SSIM(C(struct_i), real_i) = fidelity to the unseen target
"""
import time, json, os, warnings, numpy as np
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import cv2

IMG_SIZE, Z_DIM, BATCH, EPOCHS, SEED = 64, 64, 16, 6, 42
np.random.seed(SEED); tf.random.set_seed(SEED)

# paths: scripts/ -> experiments/{arrays,results,visualizations}
_S = os.path.dirname(os.path.abspath(__file__)); _E = os.path.dirname(_S)
ARR = os.path.join(_E, "arrays"); RES = os.path.join(_E, "results"); VIZ = os.path.join(_E, "visualizations", "assets")
for _d in (RES, VIZ): os.makedirs(_d, exist_ok=True)
A = lambda f: os.path.join(ARR, f); R = lambda f: os.path.join(RES, f); V = lambda f: os.path.join(VIZ, f)

Xtr = np.load(A("Xtr.npy")); ytr = np.load(A("ytr.npy"))
Xev = np.load(A("Xev.npy")); yev = np.load(A("yev.npy"))
Ctr = np.load(A("Ctr.npy")); LMtr = np.load(A("LMtr.npy"))
meta = json.load(open(A("prep_meta.json")))
num_classes = len(meta["classes"])

# structure maps for the HELD-OUT eval images (never used in training)
def structure_map(img):
    g = ((img[:, :, 0] + 1) * 127.5).clip(0, 255).astype(np.uint8)
    edge = cv2.Canny(g, 60, 160)
    _, sil = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if sil.mean() > 127: sil = 255 - sil
    dist = cv2.normalize(cv2.distanceTransform(sil, cv2.DIST_L2, 3), None, 0, 255, cv2.NORM_MINMAX)
    return (np.stack([edge, sil, dist], -1).astype(np.float32) / 127.5) - 1.0
Cev = np.stack([structure_map(x) for x in Xev]).astype(np.float32)
print(f"TF {tf.__version__} | {num_classes} classes | {len(Xtr)} train, {len(Xev)} held-out | mp {LMtr.any(1).mean():.2%}")

# ── models (same as train_eval.py) ──────────────────────────────────────────
def G_AB():
    n = tf.keras.Input((Z_DIM,)); l = tf.keras.Input((num_classes,))
    x = layers.Concatenate()([n, layers.LeakyReLU(0.2)(layers.Dense(64)(l))])
    x = layers.ReLU()(layers.BatchNormalization()(layers.Dense(4*4*256, use_bias=False)(x)))
    x = layers.Reshape((4, 4, 256))(x)
    for f in [128, 64, 32, 16]:
        x = layers.ReLU()(layers.BatchNormalization()(layers.Conv2DTranspose(f, 4, 2, padding="same", use_bias=False)(x)))
    return tf.keras.Model([n, l], layers.Conv2D(1, 3, padding="same", activation="tanh")(x))
def D_AB():
    SN = layers.SpectralNormalization
    img = tf.keras.Input((IMG_SIZE, IMG_SIZE, 1)); l = tf.keras.Input((num_classes,))
    lp = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(layers.Dense(IMG_SIZE*IMG_SIZE)(l))
    x = layers.Concatenate()([img, lp])
    for f in [32, 64, 128, 256]:
        x = layers.LeakyReLU(0.2)(SN(layers.Conv2D(f, 4, 2, padding="same"))(x))
    return tf.keras.Model([img, l], layers.Dense(1)(layers.Flatten()(x)))
def G_C():
    c = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3)); l = tf.keras.Input((num_classes,)); n = tf.keras.Input((Z_DIM,))
    e = c
    for f in [32, 64, 128, 256]:
        e = layers.LeakyReLU(0.2)(layers.BatchNormalization()(layers.Conv2D(f, 4, 2, padding="same", use_bias=False)(e)))
    x = layers.Concatenate()([layers.Flatten()(e), n, layers.LeakyReLU(0.2)(layers.Dense(64)(l))])
    x = layers.ReLU()(layers.BatchNormalization()(layers.Dense(4*4*256, use_bias=False)(x)))
    x = layers.Reshape((4, 4, 256))(x)
    for f in [128, 64, 32, 16]:
        x = layers.ReLU()(layers.BatchNormalization()(layers.Conv2DTranspose(f, 4, 2, padding="same", use_bias=False)(x)))
    return tf.keras.Model([c, l, n], layers.Conv2D(1, 3, padding="same", activation="tanh")(x))
def D_C():
    SN = layers.SpectralNormalization
    img = tf.keras.Input((IMG_SIZE, IMG_SIZE, 1)); c = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3)); l = tf.keras.Input((num_classes,))
    lp = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(layers.Dense(IMG_SIZE*IMG_SIZE)(l))
    x = layers.Concatenate()([img, c, lp])
    for f in [32, 64, 128, 256]:
        x = layers.LeakyReLU(0.2)(SN(layers.Conv2D(f, 4, 2, padding="same"))(x))
    return tf.keras.Model([img, c, l], layers.Dense(1)(layers.Flatten()(x)))
def regressor():
    i = tf.keras.Input((IMG_SIZE, IMG_SIZE, 1)); x = i
    for f in [32, 64, 128]:
        x = layers.LeakyReLU(0.2)(layers.Conv2D(f, 3, 2, padding="same")(x))
    x = layers.Dense(128)(layers.GlobalAveragePooling2D()(x))
    return tf.keras.Model(i, layers.Dense(63, activation="sigmoid")(x))

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def oh(v): return tf.one_hot(v, num_classes)

def train(kind):
    t0 = time.time()
    go = tf.keras.optimizers.Adam(2e-4, beta_1=0.5); do = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    reg = None
    if kind == "A": G, D = G_AB(), D_AB()
    elif kind == "B":
        G, D = G_AB(), D_AB(); reg = regressor(); v = LMtr.any(1)
        if v.sum() > 5:
            reg.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
            reg.fit(Xtr[v], np.clip(LMtr[v], 0, 1), epochs=6, batch_size=32, verbose=0)
        reg.trainable = False
    else: G, D = G_C(), D_C()
    n = len(Xtr); steps = n // BATCH
    for ep in range(EPOCHS):
        idx = np.random.permutation(n)
        for s in range(steps):
            b = idx[s*BATCH:(s+1)*BATCH]
            real = tf.constant(Xtr[b]); lbl = oh(ytr[b]); nz = tf.random.normal([len(b), Z_DIM])
            cond = tf.constant(Ctr[b]) if kind == "C" else None
            with tf.GradientTape() as t:
                fake = G([cond, lbl, nz]) if kind == "C" else G([nz, lbl])
                rl = D([real, cond, lbl]) if kind == "C" else D([real, lbl])
                fl = D([fake, cond, lbl]) if kind == "C" else D([fake, lbl])
                dl = bce(tf.ones_like(rl)*0.9, rl) + bce(tf.zeros_like(fl), fl)
            do.apply_gradients(zip(t.gradient(dl, D.trainable_variables), D.trainable_variables))
            nz = tf.random.normal([len(b), Z_DIM])
            with tf.GradientTape() as t:
                fake = G([cond, lbl, nz]) if kind == "C" else G([nz, lbl])
                fl = D([fake, cond, lbl]) if kind == "C" else D([fake, lbl])
                gl = bce(tf.ones_like(fl), fl) + 5.0 * tf.reduce_mean(tf.abs(fake - real))
                if kind == "B":
                    lm = tf.constant(LMtr[b]); pl = reg(fake)
                    val = tf.cast(tf.reduce_any(lm != 0., 1, keepdims=True), tf.float32)
                    gl += 2.0 * tf.reduce_mean(tf.square(pl - lm) * val)
            go.apply_gradients(zip(t.gradient(gl, G.trainable_variables), G.trainable_variables))
        print(f"  [{kind}] ep{ep+1}/{EPOCHS}")
    print(f"  [{kind}] trained {time.time()-t0:.1f}s")
    return G, round(time.time()-t0, 1)

def make_clf():
    return tf.keras.Sequential([tf.keras.Input((IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization(), layers.LeakyReLU(0.2), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same"), layers.BatchNormalization(), layers.LeakyReLU(0.2), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same"), layers.BatchNormalization(), layers.LeakyReLU(0.2), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(128), layers.LeakyReLU(0.2), layers.Dropout(0.3),
        layers.Dense(num_classes)])

# reference classifier trained on REAL (for GAN-test)
clf = make_clf(); clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
clf.fit(Xtr, ytr, epochs=40, batch_size=32, verbose=0)
real_acc = float(clf.evaluate(Xev, yev, verbose=0)[1])
print(f"reference classifier real eval-acc = {real_acc:.3f}")

def gen_class(kind, G, n_per=60, seed=0):
    """Generate a balanced set; for C use TRAINING structures sampled per class."""
    tf.random.set_seed(seed); outs, ys = [], []
    for c in range(num_classes):
        nz = tf.random.normal([n_per, Z_DIM], seed=seed*10+c); lbl = oh(np.full(n_per, c))
        if kind == "C":
            pool = np.where(ytr == c)[0]; ci = np.random.default_rng(seed+c).choice(pool, n_per)
            f = G([tf.constant(Ctr[ci]), lbl, nz]).numpy()
        else:
            f = G([nz, lbl]).numpy()
        outs.append(f); ys.append(np.full(n_per, c))
    return np.concatenate(outs), np.concatenate(ys)

def gssim(a, b):  # global SSIM, inputs HxWx1 in [-1,1]
    x = (a[:, :, 0]+1)/2; y = (b[:, :, 0]+1)/2
    mx, my, vx, vy = x.mean(), y.mean(), x.var(), y.var()
    cov = ((x-mx)*(y-my)).mean(); c1, c2 = 0.01**2, 0.03**2
    return float(((2*mx*my+c1)*(2*cov+c2))/((mx*mx+my*my+c1)*(vx+vy+c2)))

def metrics(kind, G):
    f, ys = gen_class(kind, G, 60, 0)
    pred = clf.predict(f, verbose=0).argmax(1)
    gantest = float((pred == ys).mean())
    per_class = {int(c): round(float((pred[ys == c] == c).mean()), 3) for c in range(num_classes)}
    fl = f.reshape(len(f), -1)
    div = float(np.mean([np.mean(np.abs(fl[i]-fl[j])) for i in range(60) for j in range(i+1, 60)]))
    # GAN-train: classifier trained on generated, tested on real held-out
    c2 = make_clf(); c2.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    c2.fit(f, ys, epochs=15, batch_size=32, verbose=0)
    gantrain = float(c2.evaluate(Xev, yev, verbose=0)[1])
    return {"gan_test": round(gantest, 4), "gan_train": round(gantrain, 4),
            "diversity": round(div, 4), "per_class": per_class}

def heldout_structure_test(G):
    """Generate from EVAL structure maps the generator never saw."""
    nz = tf.random.normal([len(Cev), Z_DIM], seed=7)
    fake = G([tf.constant(Cev), oh(yev), nz]).numpy()
    acc_heldout = float((clf.predict(fake, verbose=0).argmax(1) == yev).mean())
    ssim_heldout = float(np.mean([gssim(fake[i], Xev[i]) for i in range(len(fake))]))
    # recognition from TRAIN structures (same images count) for the gap
    idx = np.random.default_rng(0).choice(len(Xtr), len(Xev))
    nz2 = tf.random.normal([len(idx), Z_DIM], seed=8)
    fake_tr = G([tf.constant(Ctr[idx]), oh(ytr[idx]), nz2]).numpy()
    acc_train = float((clf.predict(fake_tr, verbose=0).argmax(1) == ytr[idx]).mean())
    # qualitative grid: [structure(edge) | C output | real target]
    pad = 2; rows = 6; cell = IMG_SIZE
    grid = np.full((rows*(cell+pad)+pad, 3*(cell+pad)+pad), 30, np.uint8)
    for r in range(rows):
        struct = ((Cev[r, :, :, 0]+1)*127.5).clip(0, 255).astype(np.uint8)
        outimg = (fake[r, :, :, 0]*127.5+127.5).clip(0, 255).astype(np.uint8)
        realim = (Xev[r, :, :, 0]*127.5+127.5).clip(0, 255).astype(np.uint8)
        for cc, im in enumerate([struct, outimg, realim]):
            y0 = pad+r*(cell+pad); x0 = pad+cc*(cell+pad); grid[y0:y0+cell, x0:x0+cell] = im
    Image.fromarray(grid).save(V("heldout_C.png"))
    return {"heldout_recognition": round(acc_heldout, 4), "trainstruct_recognition": round(acc_train, 4),
            "generalization_gap": round(acc_train-acc_heldout, 4), "heldout_ssim": round(ssim_heldout, 4)}

results, times, models = {}, {}, {}
for kind in ["A", "B", "C"]:
    print(f"=== Model {kind} ===")
    G, el = train(kind); models[kind] = G; times[kind] = el
    results[kind] = metrics(kind, G)
    print(f"  {kind}: {results[kind]['gan_test']=} {results[kind]['gan_train']=} div={results[kind]['diversity']}")

print("=== Held-out structure test (Model C) ===")
heldout = heldout_structure_test(models["C"])
print(" ", heldout)

out = {"results": results, "times": times, "real_acc": real_acc,
       "heldout_C": heldout, "meta": meta, "mp_detection_rate": float(LMtr.any(1).mean())}
json.dump(out, open(R("results_full.json"), "w"), indent=2)

print("\n================ FULL RESULTS ================")
print(f"{'Model':<8}{'GAN-test':>10}{'GAN-train':>11}{'Diversity':>11}{'Time(s)':>9}")
for k in ["A", "B", "C"]:
    r = results[k]; print(f"{k:<8}{r['gan_test']:>10}{r['gan_train']:>11}{r['diversity']:>11}{times[k]:>9}")
print(f"\nHELD-OUT STRUCTURE TEST (C): recognition {heldout['heldout_recognition']} "
      f"(train-struct {heldout['trainstruct_recognition']}, gap {heldout['generalization_gap']}) | "
      f"fidelity SSIM {heldout['heldout_ssim']}")
print(f"reference real-acc={real_acc:.3f} | chance={1/num_classes:.2f}")
print("saved results_full.json + viz_full/heldout_C.png")
