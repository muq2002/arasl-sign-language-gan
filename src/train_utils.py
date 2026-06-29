"""
Optimizer + gradient helpers that transparently handle mixed-precision loss
scaling. With USE_MIXED_PRECISION=False these are plain Adam + plain gradients,
so the numerical path is identical to the original notebooks.
"""
import tensorflow as tf
from config import USE_MIXED_PRECISION


def make_optimizer(lr):
    opt = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999, clipnorm=1.0)
    if USE_MIXED_PRECISION:
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    return opt


def apply_loss(opt, loss, tape, variables):
    """Compute + apply gradients, with loss scaling when mixed precision is on."""
    if USE_MIXED_PRECISION:
        scaled = opt.get_scaled_loss(loss)
        grads = opt.get_unscaled_gradients(tape.gradient(scaled, variables))
    else:
        grads = tape.gradient(loss, variables)
    opt.apply_gradients(zip(grads, variables))


def set_lr(opt, lr):
    """Assign a new learning rate (handles the LossScaleOptimizer wrapper)."""
    target = opt.inner_optimizer if USE_MIXED_PRECISION else opt
    target.learning_rate.assign(lr)
