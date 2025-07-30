#!/usr/bin/env python3
"""
Test script to verify all imports work correctly in the container environment.
Run this to check if the deployment will work.
"""

import sys
import os

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import asyncio
        print("✓ asyncio imported successfully")
    except ImportError as e:
        print(f"✗ asyncio import failed: {e}")
        return False
    
    try:
        import json
        print("✓ json imported successfully")
    except ImportError as e:
        print(f"✗ json import failed: {e}")
        return False
    
    try:
        from typing import List, Dict, Any
        print("✓ typing imported successfully")
    except ImportError as e:
        print(f"✗ typing import failed: {e}")
        return False
    
    try:
        from fastapi import FastAPI, HTTPException
        print("✓ fastapi imported successfully")
    except ImportError as e:
        print(f"✗ fastapi import failed: {e}")
        return False
    
    try:
        from fastapi.middleware.cors import CORSMiddleware
        print("✓ CORS middleware imported successfully")
    except ImportError as e:
        print(f"✗ CORS middleware import failed: {e}")
        return False
    
    try:
        from pydantic import BaseModel
        print("✓ pydantic imported successfully")
    except ImportError as e:
        print(f"✗ pydantic import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ opencv imported successfully (version: {cv2.__version__})")
    except ImportError as e:
        print(f"✗ opencv import failed: {e}")
        return False
    
    try:
        import aiohttp
        print("✓ aiohttp imported successfully")
    except ImportError as e:
        print(f"✗ aiohttp import failed: {e}")
        return False
    
    try:
        import requests
        print("✓ requests imported successfully")
    except ImportError as e:
        print(f"✗ requests import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ numpy imported successfully (version: {np.__version__})")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    try:
        import dotenv
        print("✓ python-dotenv imported successfully")
    except ImportError as e:
        print(f"✗ python-dotenv import failed: {e}")
        return False
    
    try:
        from google.cloud import aiplatform
        print("✓ google-cloud-aiplatform imported successfully")
    except ImportError as e:
        print(f"✗ google-cloud-aiplatform import failed: {e}")
        return False
    
    try:
        from google.cloud import speech
        print("✓ google-cloud-speech imported successfully")
    except ImportError as e:
        print(f"✗ google-cloud-speech import failed: {e}")
        return False
    
    try:
        import google.generativeai as genai
        print("✓ google-generativeai imported successfully")
    except ImportError as e:
        print(f"✗ google-generativeai import failed: {e}")
        return False
    
    # Test video_classify_hook imports
    try:
        from video_classify_hook import analyze_video_5sec, download_video
        print("✓ video_classify_hook imported successfully")
    except ImportError as e:
        print(f"✗ video_classify_hook import failed: {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True

def test_ffmpeg():
    """Test if ffmpeg is available."""
    print("\nTesting ffmpeg...")
    
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ FFmpeg is available: {version_line}")
            return True
        else:
            print(f"✗ FFmpeg failed with return code: {result.returncode}")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("✗ FFmpeg test timed out")
        return False
    except Exception as e:
        print(f"✗ FFmpeg test failed: {e}")
        return False

def test_directories():
    """Test if required directories can be created."""
    print("\nTesting directories...")
    
    try:
        os.makedirs("/tmp/json_outputs", exist_ok=True)
        print("✓ /tmp/json_outputs directory created/verified")
    except Exception as e:
        print(f"✗ Failed to create /tmp/json_outputs: {e}")
        return False
    
    try:
        os.makedirs("/app/videos", exist_ok=True)
        print("✓ /app/videos directory created/verified")
    except Exception as e:
        print(f"✗ Failed to create /app/videos: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== Import and Environment Test ===\n")
    
    imports_ok = test_imports()
    ffmpeg_ok = test_ffmpeg()
    dirs_ok = test_directories()
    
    print(f"\n=== Test Results ===")
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"FFmpeg: {'✓ PASS' if ffmpeg_ok else '✗ FAIL'}")
    print(f"Directories: {'✓ PASS' if dirs_ok else '✗ FAIL'}")
    
    if all([imports_ok, ffmpeg_ok, dirs_ok]):
        print("\n🎉 All tests passed! The application should work correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 