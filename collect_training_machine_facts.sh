#!/usr/bin/env bash
# Run ONCE inside WSL on the machine that trained the models (the one holding
# outputs/cgan_*/checkpoints/progress.json). Gathers every value that exists
# only there into reports/paper/results/training_machine/ so it can be committed:
#   - progress.json + loss_history.csv for all five models (gitignored originals)
#   - Model A epoch timing + G-update counts, F/G batch size (steps per epoch)
#   - landmark regressor validation error + constant baseline
#   - CPU / RAM / GPU / driver / CUDA / cuDNN / Keras / WSL / Windows versions
#   - pip freeze of the TensorFlow env and the MediaPipe env
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$REPO/reports/paper/results/training_machine"
mkdir -p "$OUT"
source "$REPO/activate_gpu_env.sh"

for m in cgan_A_128 cgan_B_128mp cgan_C_128struct cgan_F_128fusion cgan_G_128plus; do
  mkdir -p "$OUT/$m"
  cp -v "$REPO/outputs/$m/checkpoints/progress.json" "$OUT/$m/" 2>&1
  cp -v "$REPO/outputs/$m/history/loss_history.csv"  "$OUT/$m/" 2>&1
done

python - "$REPO" > "$OUT/progress_summary.txt" <<'PY'
import json, os, sys
repo = sys.argv[1]
for m in ["cgan_A_128", "cgan_B_128mp", "cgan_C_128struct", "cgan_F_128fusion", "cgan_G_128plus"]:
    p = os.path.join(repo, "outputs", m, "checkpoints", "progress.json")
    if not os.path.exists(p):
        print(f"{m}: progress.json MISSING"); continue
    h = json.load(open(p))
    print(f"== {m}  keys={sorted(h)}")
    s = h.get("epoch_seconds") or []
    if s:
        print(f"  epochs_timed={len(s)} mean={sum(s)/len(s):.1f}s min={min(s)}s max={max(s)}s "
              f"total={sum(s):.1f}s ({sum(s)/3600:.2f} h)")
    for k in ("gd_ratio", "n_g", "g_updates", "steps_per_epoch", "batch_size"):
        if k in h:
            print(f"  {k}={h[k]}")
    steps = h.get("steps_per_epoch")
    if isinstance(steps, list): steps = steps[-1]
    if steps:
        print(f"  batch size from 48645/steps = {48645/steps:.2f}  (54049/steps = {54049/steps:.2f})")
PY
cat "$OUT/progress_summary.txt"

( cd "$REPO/src" && python -u eval_landmark_regressor.py ) 2>&1 | tee "$OUT/landmark_regressor_eval.log"
ls -l --time-style=full-iso "$REPO/outputs/cgan_B_128mp/checkpoints/landmark_regressor.weights.h5" \
      "$REPO/outputs/cgan_B_128mp/landmarks_128px.npy" > "$OUT/regressor_file_times.txt" 2>&1

{
  echo "## CPU / RAM";      lscpu | grep -E 'Model name|^CPU\(s\)'; free -h | head -2
  echo "## Windows";        cmd.exe /c ver 2>/dev/null | tr -d '\r'
  powershell.exe -NoProfile -Command "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB; (Get-CimInstance Win32_Processor).Name" 2>/dev/null | tr -d '\r'
  echo "## WSL";            grep PRETTY_NAME /etc/os-release; uname -r; wsl.exe --version 2>/dev/null | tr -d '\0\r' | head -3
  echo "## GPU";            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo "## nvcc";           nvcc --version 2>&1 | tail -2
  echo "## Python / TF / Keras"
  python -c "import sys, tensorflow as tf, keras; b = tf.sysconfig.get_build_info(); print(sys.version.split()[0], tf.__version__, keras.__version__, 'built-against cuda', b.get('cuda_version'), 'cudnn', b.get('cudnn_version'))"
  echo "## CUDA / cuDNN runtime wheels actually loaded"
  python -m pip list 2>/dev/null | grep -E '^(nvidia-|protobuf|numpy|opencv|scikit)'
  echo "## shell history mentioning ARASL_BATCH"
  grep -h ARASL_BATCH ~/.bash_history 2>/dev/null || echo "(none found)"
} > "$OUT/environment.txt" 2>&1
cat "$OUT/environment.txt"

python -m pip freeze > "$REPO/requirements_tf.txt"
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mp \
  && python -m pip freeze > "$REPO/requirements_mp.txt" \
  && python -c "import mediapipe, google.protobuf as p; print('mediapipe', mediapipe.__version__, 'protobuf', p.__version__)" >> "$OUT/environment.txt"
ls -l --time-style=full-iso "$REPO/data/arasl.parquet" "$REPO/data/ArASL_dataset/_counts.json" >> "$OUT/environment.txt" 2>&1
echo "DONE -> $OUT  (+ requirements_tf.txt, requirements_mp.txt)"
