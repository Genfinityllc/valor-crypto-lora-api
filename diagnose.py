#!/usr/bin/env python3
"""
RunPod Container Diagnostics
"""
import os
import sys
import subprocess

def run_diagnostics():
    print("🔍 RunPod Container Diagnostics")
    print("=" * 50)
    
    # System info
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Working Dir: {os.getcwd()}")
    print(f"👤 User: {os.getenv('USER', 'unknown')}")
    print(f"🏠 Home: {os.getenv('HOME', 'unknown')}")
    print(f"📦 Port: {os.getenv('PORT', '8080')}")
    print(f"🧪 Test Mode: {os.getenv('TEST_MODE', 'false')}")
    
    # GPU Check
    print("\n🖥️ GPU Information:")
    try:
        import torch
        print(f"🔹 PyTorch: {torch.__version__}")
        print(f"🔹 CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔹 CUDA Version: {torch.version.cuda}")
            print(f"🔹 GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"🔹 GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"🔹 Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        else:
            print("❌ No CUDA GPUs available - will use CPU (VERY SLOW)")
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
    
    # Network test
    print("\n🌐 Network Test:")
    try:
        import requests
        response = requests.get("https://httpbin.org/ip", timeout=5)
        print(f"✅ Internet: {response.json()}")
    except Exception as e:
        print(f"❌ Network failed: {e}")
    
    # Disk space
    print("\n💾 Storage:")
    try:
        result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
        print(result.stdout)
    except:
        pass
    
    # Dependencies
    print("\n📚 Key Dependencies:")
    try:
        import diffusers
        print(f"✅ diffusers: {diffusers.__version__}")
    except ImportError as e:
        print(f"❌ diffusers: {e}")
        
    try:
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ transformers: {e}")
        
    try:
        import fastapi
        print(f"✅ fastapi: {fastapi.__version__}")
    except ImportError as e:
        print(f"❌ fastapi: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Diagnostics Complete")

if __name__ == "__main__":
    run_diagnostics()