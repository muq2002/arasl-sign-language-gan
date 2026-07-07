"""Generate /root/arasl_env.sh: activates the arasl conda env and puts the
pip-installed CUDA wheel libraries + WSL driver stub on the loader path so
TensorFlow 2.21 can see the RTX 3050. Run once with the arasl python."""
import os, glob, nvidia

base = os.path.dirname(nvidia.__file__)
lib_dirs = sorted(set(os.path.dirname(p)
                      for p in glob.glob(base + "/**/*.so*", recursive=True)))
nvcc_bin = os.path.join(base, "cuda_nvcc", "bin")

ld = ":".join(lib_dirs) + ":/usr/lib/wsl/lib"

script = f"""#!/usr/bin/env bash
# Activate the arasl GPU TensorFlow env (WSL2 + RTX 3050). Auto-generated.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate arasl
export LD_LIBRARY_PATH={ld}:$LD_LIBRARY_PATH
export PATH={nvcc_bin}:$PATH
export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=0
"""

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "activate_gpu_env.sh")
with open(out_path, "w", newline="\n") as f:
    f.write(script)

print("wrote", out_path, "with", len(lib_dirs), "nvidia lib dirs")
print("nvcc bin:", nvcc_bin, "exists:", os.path.isdir(nvcc_bin))
