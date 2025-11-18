#!/usr/bin/env python3
"""
Minimal test API to verify RunPod container works
"""
from fastapi import FastAPI
import uvicorn
import os
import base64
from PIL import Image
import io

app = FastAPI(title="Minimal Test API")

@app.get("/")
async def root():
    return {"status": "OK", "message": "Minimal test API working"}

@app.get("/health")
async def health():
    return {"status": "healthy", "container": "working"}

@app.post("/run")
async def test_run(request: dict):
    """Minimal test endpoint that doesn't load any models"""
    try:
        input_data = request.get("input", request)
        prompt = input_data.get("prompt", "test")
        
        # Create a simple colored rectangle instead of AI generation
        img = Image.new('RGB', (512, 512), color=(100, 150, 200))
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "image_url": f"data:image/png;base64,{img_str}",
            "image_base64": img_str,
            "metadata": {
                "prompt": prompt,
                "method": "minimal_test",
                "message": "This is a test image, not AI generated"
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🧪 Starting minimal test API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)