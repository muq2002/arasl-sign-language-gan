# -*- coding: utf-8 -*-
"""Step 3 (arasl_venv, TF 2.21): train Model A/B/C on the prepared arrays, evaluate, compare."""
import time, json, warnings, numpy as np
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import layers

IMG_SIZE, Z_DIM, BATCH, EPOCHS, SEED = 64, 64, 16, 6, 42
np.random.seed(SEED); tf.random.set_seed(SEED)

Xtr = np.load("Xtr.npy"); ytr = np.load("ytr.npy")
Xev = np.load("Xev.npy"); yev = np.load("yev.npy")
Ctr = np.load("Ctr.npy"); LMtr = np.load("LMtr.npy")
meta = json.load(open("prep_meta.json"))
num_classes = len(meta["classes"])
print(f"TF {tf.__version__} CPU | classes {meta['classes']} | {len(Xtr)} train | "
      f"mp detect {LMtr.any(1).mean():.2%} | struct cov 100%")

def G_AB():
    n = tf.keras.Input((Z_DIM,)); l = tf.keras.Input((num_classes,))
    x = layers.Concatenate()([n, layers.LeakyReLU(0.2)(layers.Dense(64)(l))])
    x = layers.ReLU()(layers.BatchNormalization()(layers.Dense(4*4*256, use_bias=False)(x)))
    x = layers.Reshape((4, 4, 256))(x)
    for f in [128, 64, 32, 16]:
        x = layers.ReLU()(layers.BatchNormalization()(layers.Conv2DTranspose(f, 4, 2, padding="same", use_bias=False)(x)))
    return tf.keras.Model([n, l], layers.Conv2D(1, 3, padding="same", activation="tanh")(x))

def D_AB(in_ch=1):
    SN = layers.SpectralNormalization
    img = tf.keras.Input((IMG_SIZE, IMG_SIZE, in_ch)); l = tf.keras.Input((num_classes,))
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
        reg.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        reg.fit(Xtr[v], np.clip(LMtr[v], 0, 1), epochs=6, batch_size=32, verbose=0); reg.trainable = False
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
        print(f"  [{kind}] ep{ep+1}/{EPOCHS} D={float(dl):.3f} G={float(gl):.3f}")
    print(f"  [{kind}] trained {time.time()-t0:.1f}s")
    return G

clf = tf.keras.Sequential([tf.keras.Input((IMG_SIZE, IMG_SIZE, 1)),
    layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization(), layers.LeakyReLU(0.2), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same"), layers.BatchNormalization(), layers.LeakyReLU(0.2), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same"), layers.BatchNormalization(), layers.LeakyReLU(0.2), layers.MaxPooling2D(),
    layers.Flatten(), layers.Dense(128), layers.LeakyReLU(0.2), layers.Dropout(0.3),
    layers.Dense(num_classes)])
clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
clf.fit(Xtr, ytr, epochs=40, batch_size=32, verbose=0)
real_acc = float(clf.evaluate(Xev, yev, verbose=0)[1])
print(f"classifier real eval-acc = {real_acc:.3f}")

def gen(kind, G, n_per=40, seed=0):
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

def evaluate(kind, G):
    f, ys = gen(kind, G, 40, 0)
    acc = float((clf.predict(f, verbose=0).argmax(1) == ys).mean())
    fl = f.reshape(len(f), -1)
    div = float(np.mean([np.mean(np.abs(fl[i]-fl[j])) for i in range(60) for j in range(i+1, 60)]))
    return {"recognition_acc": round(acc, 4), "diversity": round(div, 4)}

results = {}
for kind in ["A", "B", "C"]:
    print(f"=== Training Model {kind} ===")
    results[kind] = evaluate(kind, train(kind))
    print(f"  Model {kind}: {results[kind]}")

print("\n================ RESULTS (reduced CPU run) ================")
print(f"{'Model':<26}{'Recognition Acc':>16}{'Diversity':>12}")
nm = {"A": "A (no MediaPipe)", "B": "B (MediaPipe loss)", "C": "C (structure-cond)"}
for k in ["A", "B", "C"]:
    print(f"{nm[k]:<26}{results[k]['recognition_acc']:>16}{results[k]['diversity']:>12}")
print(f"\nclassifier sanity(real)={real_acc:.3f} | classes={meta['classes']} | "
      f"{meta['per_class']}/class | {IMG_SIZE}px | {EPOCHS} epochs | random chance={1/num_classes:.2f}")
json.dump({"results": results, "real_acc": real_acc, "meta": meta}, open("local_results.json", "w"), indent=2)
print("saved local_results.json")
