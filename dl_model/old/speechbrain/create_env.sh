#!/bin/bash
# SpeechBrain 环境创建脚本
# RTX 5070 Ti, CUDA 13.0

# 创建环境
conda create -n speechbrain python=3.10 -y
conda activate speechbrain

# 安装 PyTorch (CUDA 12.4 兼容 CUDA 13.0)
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
echo "测试安装：python -c 'import speechbrain; print(speechbrain.__version__)'"
echo "=========================================="
