#!/usr/bin/env bash
# Activate the arasl GPU TensorFlow env (WSL2 + RTX 3050). Auto-generated.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate arasl
export LD_LIBRARY_PATH=/root/miniconda3/envs/arasl/lib/python3.11/site-packages/nvidia/cublas/lib:/root/miniconda3/envs/arasl/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:/root/miniconda3/envs/arasl/lib/python3.11/site-packages/nvidia/cuda_nvcc/nvvm/lib64:/root/miniconda3/envs/arasl/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/root/miniconda3/envs/arasl/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/root/miniconda3/envs/arasl/lib/python3.11/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/arasl/lib/python3.11/site-packages/nvidia/cufft/lib:/root/miniconda3/envs/arasl/lib/python3.11/site-packages/nvidia/curand/lib:/root/miniconda3/envs/arasl/lib/python3.11/site-packages/nvidia/cusolver/lib:/root/miniconda3/envs/arasl/lib/python3.11/site-packages/nvidia/cusparse/lib:/root/miniconda3/envs/arasl/lib/python3.11/site-packages/nvidia/nccl/lib:/root/miniconda3/envs/arasl/lib/python3.11/site-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export PATH=/root/miniconda3/envs/arasl/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:$PATH
export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=0
