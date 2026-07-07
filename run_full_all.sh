#!/usr/bin/env bash
# Full 128px GPU training driver: Optimized A -> Optimized B -> Model C.
# Sequential (single GPU). Each script checkpoints every epoch and auto-resumes,
# so re-running this after an interruption continues where it left off.
# Logs -> reports/paper/logs/. Honors ARASL_EPOCHS (default 50 from config).
REPO=/mnt/c/Users/MustafaJK/arasl-sign-language-gan
LOGS=$REPO/reports/paper/logs
mkdir -p "$LOGS"
source $REPO/activate_gpu_env.sh
cd $REPO/src

ts() { date '+%Y-%m-%d %H:%M:%S'; }
run() {  # run <NAME> <script>
  echo "[$(ts)] START $1" | tee -a "$LOGS/driver.log"
  python -u "$2" 2>&1 | tee "$LOGS/train_$1.log"
  echo "[$(ts)] END $1 (exit ${PIPESTATUS[0]})" | tee -a "$LOGS/driver.log"
}

run A train_model_a.py

# Model B needs the pre-extracted landmark cache (built on CPU by prep_b_landmarks.sh).
LM=$REPO/outputs/cgan_B_128mp/landmarks_128px.npy
for i in $(seq 1 90); do [ -f "$LM" ] && break; echo "[$(ts)] waiting for landmark cache..." | tee -a "$LOGS/driver.log"; sleep 60; done
if [ -f "$LM" ]; then
  run B train_model_b.py
else
  echo "[$(ts)] SKIP B - landmark cache missing at $LM" | tee -a "$LOGS/driver.log"
fi

run C train_model_c.py

echo "[$(ts)] ALL TRAINING DONE" | tee -a "$LOGS/driver.log"
# Export + evaluate + charts
python -u export_models.py 2>&1 | tee "$LOGS/export.log"
python -u paper_eval.py    2>&1 | tee "$LOGS/paper_eval.log"
python -u paper_charts.py  2>&1 | tee "$LOGS/paper_charts.log"
echo "[$(ts)] PIPELINE COMPLETE" | tee -a "$LOGS/driver.log"
