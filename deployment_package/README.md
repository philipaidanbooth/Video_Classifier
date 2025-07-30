# Video Classification API

A FastAPI-based service for analyzing videos using Google's Gemini Vision and Speech-to-Text APIs.

## Features

- Video analysis using Google Gemini Vision API
- Audio transcription using Google Speech-to-Text API
- Support for multiple video formats
- RESTful API with CORS support
- Containerized deployment ready

## API Endpoints

### POST /
Analyze videos from a list of URLs.

**Request Body:**
```json
{
  "bubble_files": [
    {
      "url": "https://example.com/video1.mp4",
      "id": 1
    },
    {
      "url": "https://example.com/video2.mp4", 
      "id": 2
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "id": 1,
      "url": "https://example.com/video1.mp4",
      "success": true,
      "video_summary": {
        "people": "Individual",
        "gender": "Male",
        "age": "Adults",
        "composition": "Video",
        "non_human": "N/A",
        "setting_category": "Indoor",
        "setting_subcategory": "Home",
        "sound": "Dialog_On-Camera_Speech"
      },
      "transcript": "Hello, this is a test video...",
      "has_speech": true
    }
  ],
  "message": "Successfully analyzed 1 videos"
}
```

### GET /health
Health check endpoint.

## Environment Variables

- `GOOGLE_KEY`: Your Google API key for Gemini and Speech-to-Text services
- `PORT`: Port number (default: 8080)

## Deployment

### Railway Deployment

1. **Fork/Clone this repository**
2. **Set up environment variables in Railway:**
   - `GOOGLE_KEY`: Your Google API key
3. **Deploy to Railway**

The application will automatically:
- Install system dependencies (ffmpeg)
- Install Python dependencies
- Run health checks
- Start the FastAPI server

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install ffmpeg:**
   - macOS: `brew install ffmpeg`
   - Ubuntu: `sudo apt-get install ffmpeg`
   - Windows: Download from https://ffmpeg.org/

3. **Set environment variables:**
   ```bash
   export GOOGLE_KEY="your_google_api_key"
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

## Troubleshooting

### Common Issues

1. **OpenCV Import Error (`libGL.so.1`):**
   - **Solution:** Use `opencv-python-headless` instead of `opencv-python`
   - **Status:** ✅ Fixed in this deployment

2. **FFmpeg Not Found:**
   - **Solution:** FFmpeg is automatically installed in the Docker container
   - **Status:** ✅ Fixed in this deployment

3. **Google API Errors:**
   - **Check:** Ensure `GOOGLE_KEY` environment variable is set correctly
   - **Verify:** API key has access to Gemini Vision and Speech-to-Text APIs

4. **Memory Issues:**
   - **Monitor:** Video processing can be memory-intensive
   - **Optimize:** Videos are processed in 5-second chunks to reduce memory usage

### Testing

Run the test script to verify the environment:

```bash
python test_imports.py
```

This will check:
- All Python imports
- FFmpeg availability
- Directory permissions
- OpenCV functionality

## Architecture

- **FastAPI**: Web framework
- **OpenCV**: Video processing and frame extraction
- **FFmpeg**: Audio extraction
- **Google Gemini**: Video analysis
- **Google Speech-to-Text**: Audio transcription
- **aiohttp**: Async HTTP client for video downloads

## File Structure

```
deployment_package/
├── main.py                 # FastAPI application
├── video_classify_hook.py  # Video analysis logic
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── start.sh              # Startup script
├── test_imports.py       # Environment test script
├── .dockerignore         # Docker ignore file
└── README.md            # This file
```

## Performance Notes

- Videos are analyzed using the first 5 seconds only
- Frames are resized to 256x256 for faster processing
- Audio is extracted at 16kHz mono for optimal transcription
- Results are cached in `/tmp/json_outputs/`

## Security

- CORS is enabled for all origins (configure as needed for production)
- Environment variables are used for sensitive data
- Temporary files are cleaned up after processing 