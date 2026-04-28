#!/bin/bash
#SBATCH --job-name=speechbrain_env
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=dl_model/speechbrain/env_create_%j.out

# SpeechBrain 环境创建脚本 (服务器)
# RTX 5070 Ti, CUDA 13.0

module load cuda/12.4  # 使用 CUDA 12.4，向下兼容

# 创建环境
conda create -n speechbrain python=3.10 -y
conda activate speechbrain

# 安装 PyTorch (CUDA 12.4)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装 SpeechBrain
pip install speechbrain

# 安装其他依赖
pip install "numpy<2"
pip install soundfile librosa
pip install openai-whisper
pip install ffmpeg-python
conda install -y tqdm

echo "=========================================="
echo "SpeechBrain 环境创建完成！"
echo "=========================================="
echo "激活环境：conda activate speechbrain"
echo "测试 GPU: python -c 'import torch; print(torch.cuda.is_available())'"
echo "=========================================="
