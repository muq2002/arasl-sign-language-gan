#!/usr/bin/env bash
# Prepare Model B's landmark cache WITHOUT touching the GPU.
#   1) dump the exact training-order 128px image array (CPU-only TF) -> /tmp (fast)
#   2) extract MediaPipe landmarks in the `mp` env (no TensorFlow there)
# Output: outputs/cgan_B_128mp/landmarks_128px.npy  (consumed by train_model_b.py)
# Logs go to /tmp to avoid heavy /mnt/c writes during the run.
REPO=/mnt/c/Users/MustafaJK/arasl-sign-language-gan
LOGS=/tmp/arasl_logs
mkdir -p "$LOGS"

source $REPO/activate_gpu_env.sh
CUDA_VISIBLE_DEVICES="" python -u $REPO/src/dump_images_for_lm.py > "$LOGS/dump_images.log" 2>&1
echo "dump exit=$? ($(date))"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mp
python -u $REPO/experiments/scripts/extract_lm_full.py > "$LOGS/extract_lm_full.log" 2>&1
echo "extract exit=$? ($(date))"
echo "LANDMARK_PREP_DONE"
