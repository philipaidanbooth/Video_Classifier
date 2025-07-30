import asyncio
import json
import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from video_classify_hook import analyze_video_5sec, download_video

app = FastAPI(title="Video Classification API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BubbleFile(BaseModel):
    url: str
    id: int

class VideoAnalysisRequest(BaseModel):
    bubble_files: List[BubbleFile]

class VideoAnalysisResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    message: str = ""

@app.post("/", response_model=VideoAnalysisResponse)
async def analyze_videos(request: VideoAnalysisRequest):
    """Analyze videos from a list of URLs."""
    try:
        results = []
        
        # Ensure output directory exists in temp
        os.makedirs("/tmp/json_outputs", exist_ok=True)
        
        for bubble_file in request.bubble_files:
            print(f"\n{'='*50}")
            print(f"Processing video ID {bubble_file.id}: {bubble_file.url}")
            
            # Download the video
            video_path = await download_video(bubble_file.url)
            if not video_path:
                print(f"Failed to download: {bubble_file.url}")
                results.append({
                    "id": bubble_file.id,
                    "url": bubble_file.url,
                    "error": "Failed to download video",
                    "success": False
                })
                continue
            
            try:
                # Analyze the video
                result = await analyze_video_5sec(video_path)
                result["id"] = bubble_file.id
                result["url"] = bubble_file.url
                result["success"] = True
                
                results.append(result)
                
                # Save individual result to temp
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_file = os.path.join("/tmp/json_outputs", f"analysis_id_{bubble_file.id}_{video_name}.json")
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Saved analysis to {output_file}")
                
                # Clean up downloaded file
                os.remove(video_path)
                
            except Exception as e:
                print(f"Error analyzing video {bubble_file.id}: {e}")
                results.append({
                    "id": bubble_file.id,
                    "url": bubble_file.url,
                    "error": str(e),
                    "success": False
                })
        
        # Save combined results to temp
        combined_file = os.path.join("/tmp/json_outputs", "api_analysis_results.json")
        with open(combined_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*50}")
        print("Analysis complete!")
        
        return VideoAnalysisResponse(
            success=True,
            results=results,
            message=f"Successfully analyzed {len([r for r in results if r.get('success', False)])} videos"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Video Classification API is running"}

# Cloud Run startup configuration
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080))) 