#!/bin/bash
set -e  # Exit on any error

echo "🚀 Starting Valor Crypto LoRA API deployment..."
echo "📅 $(date)"

# Clone repository
echo "📦 Cloning repository..."
git clone https://github.com/Genfinityllc/valor-crypto-lora-api.git /workspace/app
cd /workspace/app

echo "📁 Current directory: $(pwd)"
echo "📄 Files in directory:"
ls -la

# Install dependencies
echo "🔧 Installing Python dependencies..."
pip install -r requirements.txt

echo "🔍 Checking Python environment..."
python --version
pip list | grep -E "(torch|diffusers|fastapi|uvicorn)"

# Check GPU availability
echo "🖥️ Checking GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU\"}')"

# Start the application
echo "🎯 Starting FastAPI application on port 8080..."

# Try minimal test first, then full app
if [ "${TEST_MODE:-false}" = "true" ]; then
    echo "🧪 Running in TEST MODE - minimal API only"
    python test_minimal.py
else
    echo "🚀 Running FULL AI generation API"
    python app.py
fi