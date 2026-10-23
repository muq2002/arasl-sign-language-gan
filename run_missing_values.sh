#!/usr/bin/env bash
# Finish the reviewer-requested runs on the laptop (CPU only; this TensorFlow
# build cannot run convolutions on the RTX 5060). Fully resumable: the reference
# classifier saves every epoch (reports/paper/results/seed42/*.partial.*), each
# stage is skipped once its output exists, and landmark extraction resumes every
# 5,000 images. Just run it again after any interruption:
#
#   wsl -e bash /mnt/c/Users/mktad/OneDrive/Documents/GitHub/arasl-sign-language-gan/run_missing_values.sh
#
# Outputs:  reports/paper/results/seed42/                 (items 15-16)
#           reports/paper/results/landmark_regressor_eval_repro.json  (items 5-6)
# When it prints "regressor exit=0", copy the results into
# missing-values-answers.html (items 5, 6, 15, 16).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$REPO/reports/paper/logs"; W=$HOME/arasl-work/repro_B; mkdir -p "$W"
RES="$REPO/reports/paper/results/seed42"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate arasl
export CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=1 TF_ENABLE_ONEDNN_OPTS=0
export ARASL_DATA=$HOME/arasl-work/data/ArASL_dataset
export ARASL_EVAL_SEED=42 ARASL_EVAL_OUT="$RES"
cd "$REPO/src"
echo "start $(date +%T)"
if [ ! -f "$RES/metrics.json" ]; then
  python -u paper_eval.py >> "$LOGS/paper_eval_seed42.log" 2>&1; echo "eval exit=$? $(date +%T)"
else echo "eval skip (done)"; fi
if [ ! -f "$RES/per_class_accuracy.csv" ]; then
  python -u confusion_matrix.py > "$LOGS/confusion_seed42.log" 2>&1; echo "confusion exit=$? $(date +%T)"
else echo "confusion skip (done)"; fi
export ARASL_LM_IMAGES=$W/images_128.npy
if [ ! -f "$ARASL_LM_IMAGES" ]; then
  ARASL_OUT=$W/outputs python -u dump_images_for_lm.py > "$LOGS/repro_dump_images.log" 2>&1; echo "dump exit=$? $(date +%T)"
else echo "dump skip (done)"; fi
if [ ! -f "$W/lm_full_meta.json" ]; then
  conda activate mp_extract
  python -u "$REPO/experiments/scripts/extract_lm_full.py" "$W" >> "$LOGS/repro_extract_lm.log" 2>&1; echo "extract exit=$? $(date +%T)"
  conda activate arasl
else echo "extract skip (done)"; fi
rm -f "$W/landmark_regressor_repro.weights.h5"   # a partial regressor run never counts
ARASL_LM_CACHE=$W/landmarks_128px.npy ARASL_REG_WEIGHTS=$W/landmark_regressor_repro.weights.h5 ARASL_REG_RETRAIN=1 \
ARASL_REG_OUT="$REPO/reports/paper/results/landmark_regressor_eval_repro.json" \
python -u eval_landmark_regressor.py > "$LOGS/repro_landmark_regressor.log" 2>&1; echo "regressor exit=$? $(date +%T)"
