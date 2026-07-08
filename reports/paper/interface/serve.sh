#!/usr/bin/env bash
# Launch the ArASL Gradio demo fully detached (survives the parent shell exiting),
# wait until it is serving on :7860, then print the URLs. Runs on GPU or CPU.
set -u
ROOT=/mnt/c/Users/MustafaJK/arasl-sign-language-gan
source "$ROOT/activate_gpu_env.sh" 2>/dev/null || conda activate arasl 2>/dev/null
cd "$ROOT/reports/paper/interface"

pkill -f "python app.py" 2>/dev/null
sleep 1
setsid nohup python app.py > /tmp/arasl_gradio.log 2>&1 < /dev/null &
disown || true

# wait up to ~90s for the port to bind (model loading takes a while)
for i in $(seq 1 45); do
  if ss -ltn 2>/dev/null | grep -q ":7860"; then break; fi
  sleep 2
done

if ss -ltn 2>/dev/null | grep -q ":7860"; then
  IP=$(hostname -I 2>/dev/null | awk '{print $1}')
  echo "SERVER_UP"
  echo "windows_url=http://localhost:7860"
  echo "wsl_ip_url=http://${IP}:7860"
else
  echo "SERVER_DOWN"
  tail -n 15 /tmp/arasl_gradio.log 2>/dev/null
fi
