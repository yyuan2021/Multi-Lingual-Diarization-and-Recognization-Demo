@echo off
REM SpeechBrain 环境创建脚本 (Windows)
REM RTX 5070 Ti, CUDA 13.0

echo ==========================================
echo 创建 SpeechBrain conda 环境...
echo ==========================================

conda create -n speechbrain python=3.10 -y
if errorlevel 1 (
    echo 创建环境失败
    exit /b 1
)

echo ==========================================
echo 激活环境...
echo ==========================================

call conda activate speechbrain

echo ==========================================
echo 安装 PyTorch (CUDA 12.4，兼容 CUDA 13.0)...
echo ==========================================

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ==========================================
echo 安装 SpeechBrain...
echo ==========================================

pip install speechbrain

echo ==========================================
echo 安装其他依赖...
echo ==========================================

pip install "numpy^<2"
pip install soundfile librosa
pip install openai-whisper
pip install ffmpeg-python
conda install -y tqdm

echo ==========================================
echo SpeechBrain 环境创建完成！
echo ==========================================
echo 激活环境：conda activate speechbrain
echo 测试安装：python -c "import speechbrain; print(speechbrain.__version__)"
echo ==========================================
