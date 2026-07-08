"""
Central configuration + speed switches for both models.

Speed optimizations live behind toggles so you can verify they do NOT change
results: flip them off and you get the original numerical behaviour, flip them
on for the fast path. None of them alter the loss math or architecture.
"""
import os
import tensorflow as tf

# ──────────────────────────────────────────────────────────────────────────
#  SPEED SWITCHES  (accuracy-neutral)
# ──────────────────────────────────────────────────────────────────────────
USE_MIXED_PRECISION = True    # float16 compute + loss scaling (tensor-core GPUs)
USE_XLA             = False   # jit_compile train steps. Big speedup; enable if stable on your GPU.
TF_DATA_PREFETCH    = True    # overlap host->device copies with compute

# ──────────────────────────────────────────────────────────────────────────
#  PATHS  (repo-relative so every artifact stays inside the cloned repo)
# ──────────────────────────────────────────────────────────────────────────
# src/config.py -> repo root is one level up
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# allow override from the environment (e.g. a faster WSL-native path)
_OUT = os.environ.get("ARASL_OUT", os.path.join(_REPO_ROOT, "outputs"))

DATA_PATH = os.environ.get("ARASL_DATA",
                           os.path.join(_REPO_ROOT, "data", "ArASL_dataset"))

DRIVE_BASE_A = os.path.join(_OUT, "cgan_A_128")
DRIVE_BASE_B = os.path.join(_OUT, "cgan_B_128mp")
DRIVE_BASE_C = os.path.join(_OUT, "cgan_C_128struct")
DRIVE_BASE_F = os.path.join(_OUT, "cgan_F_128fusion")   # Model F = C structure + B landmark loss
DRIVE_BASE_G = os.path.join(_OUT, "cgan_G_128plus")     # Model G = F + aux-clf + feature-match + EMA

# ──────────────────────────────────────────────────────────────────────────
#  ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────
Z_DIM        = 128
IMG_SIZE     = 128
IMG_CHANNELS = 1
RANDOM_SEED  = 42

# ──────────────────────────────────────────────────────────────────────────
#  TRAINING
# ──────────────────────────────────────────────────────────────────────────
EPOCHS       = int(os.environ.get("ARASL_EPOCHS", 50))   # override for smoke tests
BATCH_SIZE   = int(os.environ.get("ARASL_BATCH", 32))
LR_G         = 2e-4
LR_D         = 1e-4
LR_DECAY_G   = 35
LR_DECAY_D   = 20
LABEL_SMOOTH = 0.9

G_UPDATES_BASE = 2
G_D_RATIO_MAX  = 2.0

# Pixel structural loss schedule (both models)
LAMBDA_PIX_START = 0.5
LAMBDA_PIX_END   = 5.0
WARMUP_EP        = 10
PHASE2_EP        = 10

# Landmark loss schedule (Model B only)
LAMBDA_LM_START = 0.0
LAMBDA_LM_END   = 2.0
WARMUP_LM_EP    = 15
LAMBDA_DIV      = 0.05

# ──────────────────────────────────────────────────────────────────────────
#  MEDIAPIPE  (Model B)
# ──────────────────────────────────────────────────────────────────────────
MP_DETECT_SIZE       = 256
MP_DETECT_SIZE_FB    = 320
MP_CONFIDENCE        = 0.30
MP_CONFIDENCE_LOW    = 0.15
MP_MODEL_COMPLEX     = 1
MIN_VALID_DETECTIONS = 50

# Landmark regressor
REG_EPOCHS     = 20
REG_BATCH_SIZE = 64
REG_LR         = 1e-3

# ──────────────────────────────────────────────────────────────────────────
#  EVALUATION
# ──────────────────────────────────────────────────────────────────────────
N_FID_PER_CLASS  = 60
N_FID_SEEDS      = 5
N_PKLE_PER_CLASS = 15
SAVE_EVERY_N     = 5


def setup_speed():
    """Call ONCE before building any model. Sets seeds, GPU growth, precision policy."""
    import numpy as np
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass

    if USE_MIXED_PRECISION and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: mixed_float16 enabled")
    else:
        print("Mixed precision: OFF (float32)")

    print(f"XLA jit_compile: {'ON' if USE_XLA else 'OFF'}  |  GPUs: {[g.name for g in gpus]}")
    return len(gpus) > 0


def make_dirs(base):
    paths = {
        "ckpt":     os.path.join(base, "checkpoints"),
        "samples":  os.path.join(base, "samples"),
        "eval":     os.path.join(base, "eval"),
        "fid_real": os.path.join(base, "eval", "fid_real"),
        "history":  os.path.join(base, "history"),
        "plots":    os.path.join(base, "plots"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    paths["progress"]  = os.path.join(paths["ckpt"], "progress.json")
    paths["regressor"] = os.path.join(paths["ckpt"], "landmark_regressor.weights.h5")
    return paths
