#!/bin/bash

# Print environment information for debugging
echo "=== Environment Information ==="
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Port: $PORT"
echo "Environment variables:"
env | grep -E "(GOOGLE|PORT|RAILWAY)" || echo "No relevant env vars found"

# Check if required directories exist
echo "=== Directory Check ==="
mkdir -p /tmp/json_outputs /app/videos
echo "Created/verified directories: /tmp/json_outputs, /app/videos"

# Check if ffmpeg is available
echo "=== FFmpeg Check ==="
if command -v ffmpeg &> /dev/null; then
    echo "FFmpeg is available: $(ffmpeg -version | head -n1)"
else
    echo "WARNING: FFmpeg not found!"
fi

# Check if OpenCV is working
echo "=== OpenCV Check ==="
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" || echo "ERROR: OpenCV import failed"

# Run comprehensive test
echo "=== Running Comprehensive Test ==="
python test_imports.py

# Start the application
echo "=== Starting Application ==="
exec python main.py 