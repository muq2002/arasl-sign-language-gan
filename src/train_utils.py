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


def scaled(opt, loss):
    """Scale the loss for mixed precision. MUST be called INSIDE the GradientTape
    (Keras 3 LossScaleOptimizer.scale_loss records a dynamic-scale op the tape
    needs to see). With MP off this is a no-op."""
    return opt.scale_loss(loss) if USE_MIXED_PRECISION else loss


def apply_grads(opt, grads, variables):
    """Apply (already-computed) gradients. Keras 3 LossScaleOptimizer.apply()
    unscales internally; a plain optimizer uses apply_gradients."""
    if USE_MIXED_PRECISION:
        opt.apply(grads, variables)
    else:
        opt.apply_gradients(zip(grads, variables))


def apply_loss(opt, loss, tape, variables):
    """Deprecated: kept for compatibility. Prefer scaled()+apply_grads() so the
    scale op is recorded inside the tape."""
    grads = tape.gradient(scaled(opt, loss), variables)
    apply_grads(opt, grads, variables)


def set_lr(opt, lr):
    """Assign a new learning rate (handles the LossScaleOptimizer wrapper)."""
    target = opt.inner_optimizer if USE_MIXED_PRECISION else opt
    target.learning_rate.assign(lr)
