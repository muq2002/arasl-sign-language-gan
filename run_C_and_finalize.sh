#!/usr/bin/env bash
# Re-run ONLY Model C (A & B already trained) then export + evaluate + charts.
REPO=/mnt/c/Users/MustafaJK/arasl-sign-language-gan
LOGS=$REPO/reports/paper/logs
mkdir -p "$LOGS"
source $REPO/activate_gpu_env.sh
cd $REPO/src
ts() { date '+%Y-%m-%d %H:%M:%S'; }
run() {
  echo "[$(ts)] START $1" | tee -a "$LOGS/driver.log"
  python -u "$2" 2>&1 | tee "$LOGS/train_$1.log"
  echo "[$(ts)] END $1 (exit ${PIPESTATUS[0]})" | tee -a "$LOGS/driver.log"
}

run C train_model_c.py

echo "[$(ts)] C DONE, finalizing (export/eval/charts)" | tee -a "$LOGS/driver.log"
python -u export_models.py 2>&1 | tee "$LOGS/export.log"
python -u paper_eval.py    2>&1 | tee "$LOGS/paper_eval.log"
python -u paper_charts.py  2>&1 | tee "$LOGS/paper_charts.log"
echo "[$(ts)] PIPELINE COMPLETE (C rerun + finalize)" | tee -a "$LOGS/driver.log"
