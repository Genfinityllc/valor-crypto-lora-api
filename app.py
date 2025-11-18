#!/usr/bin/env python3
"""
🚀 Valor Crypto LoRA API
✅ Custom trained LoRA for cryptocurrency cover generation
✅ FastAPI endpoint optimized for RunPod serverless
"""
import os
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import DiffusionPipeline, AutoencoderKL
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import io
import uvicorn

# FastAPI app
app = FastAPI(title="Valor Crypto LoRA API", version="2.0")

# Global pipeline
_pipeline = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

class GenerationRequest(BaseModel):
    prompt: str
    title: str = ""
    lora_scale: float = 1.0
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    negative_prompt: str = "low quality, blurry, text, watermark, signature"

class GenerationResponse(BaseModel):
    success: bool
    image_url: str = ""
    image_base64: str = ""
    error: str = ""
    metadata: dict = {}

def load_lora_pipeline():
    """Load SDXL pipeline with optional LoRA"""
    global _pipeline
    
    if _pipeline is not None:
        return _pipeline
    
    try:
        print(f"🚀 Loading SDXL pipeline on {_device}")
        
        # Load SDXL base model
        _pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if _device == "cuda" else None
        )
        
        # Try to load custom LoRA if available
        try:
            lora_path = "./crypto_lora_trained/adapter_model.safetensors"
            if os.path.exists(lora_path):
                print(f"📚 Loading custom trained LoRA from {lora_path}")
                _pipeline.load_lora_weights("./crypto_lora_trained/", adapter_name="crypto_lora")
                print("✅ Custom LoRA loaded successfully!")
            else:
                print(f"💡 No custom LoRA found at {lora_path}, using base SDXL model")
                print("📁 Available files:", os.listdir("."))
        except Exception as e:
            print(f"⚠️ LoRA loading failed, continuing with base model: {e}")
            # Don't let LoRA failures crash the pipeline
        
        # Move to device and optimize
        _pipeline = _pipeline.to(_device)
        
        if _device == "cuda":
            try:
                _pipeline.enable_memory_efficient_attention()
                _pipeline.enable_vae_slicing()
            except:
                pass
        
        print("✅ Pipeline loaded successfully!")
        return _pipeline
        
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        raise

def add_valor_title_watermark(image, title):
    """Add VALOR title + Genfinity watermark"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        if image.size != (1800, 900):
            image = image.resize((1800, 900), Image.Resampling.LANCZOS)
        
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        if title and title.strip():
            font_size = 72
            try:
                font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            title = title.upper()
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (1800 - text_width) // 2
            y = (900 - text_height) // 2 - 40
            
            draw.text((x, y), title, font=font, fill=(255, 255, 255, 255))
        
        # Add watermark if available
        watermark_path = "./genfinity-watermark.png"
        if os.path.exists(watermark_path):
            try:
                watermark = Image.open(watermark_path).convert('RGBA')
                wm_width = int(1800 * 0.15)
                wm_ratio = watermark.size[1] / watermark.size[0]
                wm_height = int(wm_width * wm_ratio)
                watermark = watermark.resize((wm_width, wm_height), Image.Resampling.LANCZOS)
                
                wm_x = 1800 - wm_width - 30
                wm_y = 900 - wm_height - 30
                
                overlay.paste(watermark, (wm_x, wm_y), watermark)
            except:
                pass
        
        result = Image.alpha_composite(image, overlay)
        return result.convert('RGB')
        
    except Exception as e:
        print(f"❌ Error adding title/watermark: {e}")
        return image.convert('RGB')

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "OK", "service": "Valor Crypto LoRA API", "version": "2.0"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "OK", 
        "pipeline_loaded": _pipeline is not None,
        "device": _device,
        "cuda_available": torch.cuda.is_available(),
        "service": "Valor Crypto LoRA API",
        "version": "2.0"
    }

@app.post("/run")
async def runpod_generate(request: dict):
    """RunPod serverless endpoint - returns job for async processing"""
    try:
        # Extract input from RunPod format
        input_data = request.get("input", request)
        
        # Convert to our format
        generation_request = GenerationRequest(
            prompt=input_data.get("prompt", ""),
            title=input_data.get("title", ""),
            lora_scale=input_data.get("lora_scale", 1.0),
            num_inference_steps=input_data.get("num_inference_steps", 20),
            guidance_scale=input_data.get("guidance_scale", 7.5),
            width=input_data.get("width", 1024),
            height=input_data.get("height", 1024),
            negative_prompt=input_data.get("negative_prompt", "low quality, blurry, text, watermark, signature")
        )
        
        # Generate image
        result = await generate_image(generation_request)
        
        # Return in RunPod serverless format
        if result.success:
            # Convert base64 to data URL for backend compatibility
            image_data_url = f"data:image/png;base64,{result.image_base64}"
            return {
                "success": True,
                "image_url": image_data_url,
                "image_base64": result.image_base64,
                "metadata": result.metadata
            }
        else:
            raise HTTPException(status_code=500, detail=result.error)
            
    except Exception as e:
        print(f"❌ Error in /run endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_sync")
async def generate_sync(request: dict):
    """Synchronous endpoint for direct backend integration"""
    try:
        # Handle both direct format and RunPod wrapper format
        input_data = request.get("input", request)
        
        # Convert to our format
        generation_request = GenerationRequest(
            prompt=input_data.get("prompt", ""),
            title=input_data.get("title", ""),
            lora_scale=input_data.get("lora_scale", 1.0),
            num_inference_steps=input_data.get("num_inference_steps", 20),
            guidance_scale=input_data.get("guidance_scale", 7.5),
            width=input_data.get("width", 1024),
            height=input_data.get("height", 1024),
            negative_prompt=input_data.get("negative_prompt", "low quality, blurry, text, watermark, signature")
        )
        
        print(f"🎯 Sync generation: {generation_request.prompt}")
        
        # Generate image
        result = await generate_image(generation_request)
        
        # Return direct result
        if result.success:
            return {
                "success": True,
                "image_base64": result.image_base64,
                "metadata": result.metadata
            }
        else:
            return {
                "success": False,
                "error": result.error
            }
            
    except Exception as e:
        print(f"❌ Error in /generate_sync: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """Generate image with optional LoRA"""
    try:
        print(f"🎨 Generating: '{request.prompt}' + title: '{request.title}'")
        
        # Load pipeline
        pipeline = load_lora_pipeline()
        
        # Enhanced prompt for crypto covers
        enhanced_prompt = f"{request.prompt}, professional editorial magazine cover design, high quality cryptocurrency illustration, detailed, trending"
        
        # Generate image
        try:
            with torch.autocast(_device):
                result = pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=request.negative_prompt,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    width=request.width,
                    height=request.height,
                    generator=torch.Generator(_device).manual_seed(42)
                )
        except Exception as generation_error:
            print(f"❌ Image generation failed: {generation_error}")
            # Try with simpler settings
            print("🔄 Retrying with basic settings...")
            with torch.autocast(_device):
                result = pipeline(
                    prompt=request.prompt,  # Use original prompt
                    num_inference_steps=10,  # Reduce steps
                    guidance_scale=5.0,      # Reduce guidance
                    width=512,               # Smaller size
                    height=512,
                    generator=torch.Generator(_device).manual_seed(42)
                )
        
        # Get generated image
        generated_image = result.images[0]
        
        # Resize and add title/watermark
        resized_image = generated_image.resize((1800, 900), Image.Resampling.LANCZOS)
        final_image = add_valor_title_watermark(resized_image, request.title)
        
        # Convert to base64
        image_base64 = image_to_base64(final_image)
        
        print("✅ Generation completed successfully!")
        
        return GenerationResponse(
            success=True,
            image_base64=image_base64,
            metadata={
                "lora_scale": request.lora_scale,
                "steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "prompt": enhanced_prompt,
                "custom_lora": os.path.exists("./crypto_lora_trained/adapter_model.safetensors")
            }
        )
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return GenerationResponse(
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    print("🚀 Starting Valor Crypto LoRA API...")
    print(f"🖥️ Device: {_device}")
    print(f"📍 PyTorch version: {torch.__version__}")
    
    # Pre-load pipeline
    try:
        print("📚 Loading pipeline...")
        load_lora_pipeline()
        print("✅ Pipeline loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        # Continue anyway for debugging
    
    # Start API server
    port = int(os.environ.get("PORT", 8080))
    print(f"🌐 Starting server on 0.0.0.0:{port}")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        raise