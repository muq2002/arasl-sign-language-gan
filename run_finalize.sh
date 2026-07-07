#!/usr/bin/env bash
# Re-run evaluation + charts only (models already trained & exported).
REPO=/mnt/c/Users/MustafaJK/arasl-sign-language-gan
LOGS=$REPO/reports/paper/logs
source $REPO/activate_gpu_env.sh
cd $REPO/src
ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] FINALIZE START" | tee -a "$LOGS/driver.log"
python -u paper_eval.py   2>&1 | tee "$LOGS/paper_eval.log"
echo "[$(ts)] eval exit=${PIPESTATUS[0]}" | tee -a "$LOGS/driver.log"
python -u paper_charts.py 2>&1 | tee "$LOGS/paper_charts.log"
echo "[$(ts)] charts exit=${PIPESTATUS[0]}" | tee -a "$LOGS/driver.log"
echo "[$(ts)] FINALIZE DONE" | tee -a "$LOGS/driver.log"
