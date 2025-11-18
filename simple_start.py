#!/usr/bin/env python3
"""
Simple Python startup script to avoid bash command issues
"""
import os
import sys
import subprocess

def main():
    print("🚀 Python startup script starting...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📦 Python path: {sys.executable}")
    
    # Show environment
    test_mode = os.environ.get('TEST_MODE', 'false')
    port = os.environ.get('PORT', '8080')
    
    print(f"🧪 TEST_MODE: {test_mode}")
    print(f"📡 PORT: {port}")
    
    # Run diagnostics
    print("🔍 Running diagnostics...")
    try:
        subprocess.run([sys.executable, "diagnose.py"], check=True)
    except Exception as e:
        print(f"⚠️ Diagnostics failed: {e}")
    
    # Start the appropriate app
    if test_mode.lower() == 'true':
        print("🧪 Starting TEST MODE (minimal API)")
        app_file = "test_minimal.py"
    else:
        print("🚀 Starting FULL MODE (AI generation)")
        app_file = "app.py"
    
    print(f"▶️ Executing: {app_file}")
    
    # Start the application
    os.execv(sys.executable, [sys.executable, app_file])

if __name__ == "__main__":
    main()