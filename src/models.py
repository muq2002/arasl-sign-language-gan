"""
Model definitions — identical architecture to the original notebooks, made
mixed-precision safe:

  * output / logit layers are forced to float32 (stable tanh + BCE-from-logits)
  * self-attention softmax is computed in float32 (avoids float16 overflow)

Nothing about the network capacity or layer sizes changed.
"""
import tensorflow as tf
from tensorflow.keras import layers

from config import IMG_SIZE


class SelfAttention2D(layers.Layer):
    """SAGAN-style self-attention. gamma=0 init. Softmax in float32 for fp16 safety."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        ch8 = max(channels // 8, 1)
        self.q = layers.Conv2D(ch8, 1, use_bias=False)
        self.k = layers.Conv2D(ch8, 1, use_bias=False)
        self.v = layers.Conv2D(channels, 1, use_bias=False)
        self.gamma = None

    def build(self, input_shape):
        self.gamma = self.add_weight(name="gamma", shape=(), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        B = tf.shape(x)[0]
        H, W, C = x.shape[1], x.shape[2], x.shape[3]
        ch8 = max(C // 8, 1)
        q = tf.reshape(self.q(x), [B, H * W, ch8])
        k = tf.reshape(self.k(x), [B, H * W, ch8])
        v = tf.reshape(self.v(x), [B, H * W, C])
        # softmax in float32 regardless of compute dtype
        scale = tf.cast(ch8, tf.float32) ** -0.5
        logits = tf.cast(tf.matmul(q, k, transpose_b=True), tf.float32) * scale
        attn = tf.cast(tf.nn.softmax(logits, axis=-1), v.dtype)
        out = tf.reshape(tf.matmul(attn, v), [B, H, W, C])
        return x + self.gamma * out

    def get_config(self):
        cfg = super().get_config()
        cfg["channels"] = self.channels
        return cfg


def build_generator(z_dim, num_classes):
    noise_in = tf.keras.Input(shape=(z_dim,), name="noise")
    label_in = tf.keras.Input(shape=(num_classes,), name="label")

    lbl = layers.Dense(128, use_bias=False)(label_in)
    lbl = layers.LeakyReLU(negative_slope=0.2)(lbl)

    x = layers.Concatenate()([noise_in, lbl])
    x = layers.Dense(4 * 4 * 512, use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Reshape((4, 4, 512))(x)

    x = layers.Conv2DTranspose(256, 4, 2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, 4, 2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, 4, 2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = SelfAttention2D(128, name="self_attn")(x)

    x = layers.Conv2DTranspose(64, 4, 2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, 4, 2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    # output forced to float32 for a stable tanh
    out = layers.Conv2D(1, 3, padding="same", dtype="float32")(x)
    out = layers.Activation("tanh", dtype="float32", name="img_out")(out)
    return tf.keras.Model([noise_in, label_in], out, name="Generator_128")


def build_discriminator(num_classes):
    SN = layers.SpectralNormalization
    image_in = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")
    label_in = tf.keras.Input(shape=(num_classes,), name="label")

    lp = layers.Dense(IMG_SIZE * IMG_SIZE)(label_in)
    lp = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(lp)
    x = layers.Concatenate()([image_in, lp])

    x = SN(layers.Conv2D(64, 4, strides=2, padding="same"))(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = SN(layers.Conv2D(128, 4, strides=2, padding="same"))(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x); x = layers.Dropout(0.2)(x)

    x = SN(layers.Conv2D(256, 4, strides=2, padding="same"))(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x); x = layers.Dropout(0.2)(x)

    x = SN(layers.Conv2D(512, 4, strides=2, padding="same"))(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x); x = layers.Dropout(0.3)(x)

    x = SN(layers.Conv2D(512, 4, strides=2, padding="same"))(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x); x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    out = layers.Dense(1, dtype="float32")(x)   # float32 logits
    return tf.keras.Model([image_in, label_in], out, name="Discriminator_128")


def build_landmark_regressor(img_size=IMG_SIZE):
    inp = tf.keras.Input(shape=(img_size, img_size, 1), name="image")
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inp); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256)(x); x = layers.LeakyReLU(0.2)(x); x = layers.Dropout(0.3)(x)
    out = layers.Dense(63, activation="sigmoid", dtype="float32", name="landmarks")(x)
    return tf.keras.Model(inp, out, name="LandmarkRegressor")
