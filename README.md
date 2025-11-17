# 🚀 Valor Crypto LoRA API

**Custom trained LoRA for cryptocurrency cover generation - November 2025**

## Features
- ✅ SDXL base model optimized for crypto covers
- ✅ Optional custom LoRA integration
- ✅ FastAPI production endpoint
- ✅ Valor title system + Genfinity watermarking
- ✅ Ready for RunPod serverless deployment

## Quick Deploy to RunPod

### Container Configuration:
- **Image**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- **Start Command**: `git clone https://github.com/Genfinityllc/valor-crypto-lora-api.git /workspace/app && cd /workspace/app && pip install -r requirements.txt && python app.py`

### Environment Variables:
- `PORT`: 8000 (default)

## API Endpoints

### Generate Image
```bash
POST /generate
{
    "prompt": "bitcoin cryptocurrency professional cover design",
    "title": "Bitcoin News",
    "lora_scale": 1.0,
    "num_inference_steps": 20
}
```

### Health Check
```bash
GET /health
```

## Custom LoRA Integration

To use your trained LoRA:
1. Upload `adapter_model.safetensors` and `adapter_config.json` to `/workspace/app/crypto_lora_trained/`
2. Restart the service
3. The API will automatically detect and load your custom LoRA

## Features
- Professional cryptocurrency cover generation
- Automatic title overlay and watermarking
- GPU-optimized inference
- Error handling and graceful fallbacks

---
Generated with Claude Code - November 2025