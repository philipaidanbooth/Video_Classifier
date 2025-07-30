import os
import shutil
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from video_classify_hook import download_video, analyze_video_5sec

class VideoManager:
    """
    Manages video downloads, storage, and cleanup for large-scale processing.
    """
    
    def __init__(self, base_dir: str = "video_storage"):
        self.base_dir = base_dir
        self.videos_dir = os.path.join(base_dir, "videos")
        self.results_dir = os.path.join(base_dir, "results")
        self.temp_dir = os.path.join(base_dir, "temp")
        
        # Create directory structure
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def get_video_path(self, video_id: str, filename: str = None) -> str:
        """Get the path where a video should be stored."""
        video_dir = os.path.join(self.videos_dir, f"video_{video_id}")
        os.makedirs(video_dir, exist_ok=True)
        
        if filename:
            return os.path.join(video_dir, filename)
        else:
            return video_dir
    
    def video_exists(self, video_id: str, filename: str = None) -> bool:
        """Check if a video already exists."""
        video_path = self.get_video_path(video_id, filename)
        if filename:
            return os.path.exists(video_path)
        else:
            # Check if any video file exists in the directory
            video_dir = self.get_video_path(video_id)
            return any(f.endswith(('.mp4', '.avi', '.mov', '.mkv')) for f in os.listdir(video_dir))
    
    def get_stored_video_path(self, video_id: str) -> Optional[str]:
        """Get the path of a stored video file."""
        video_dir = self.get_video_path(video_id)
        if not os.path.exists(video_dir):
            return None
        
        # Find the first video file in the directory
        for file in os.listdir(video_dir):
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return os.path.join(video_dir, file)
        return None
    
    async def download_and_store_video(self, url: str, video_id: str) -> Optional[str]:
        """Download and store a video with proper organization."""
        try:
            # Check if video already exists
            if self.video_exists(video_id):
                print(f"Video {video_id} already exists, skipping download")
                return self.get_stored_video_path(video_id)
            
            # Download video
            video_path = await download_video(url, video_id)
            if video_path:
                print(f"Successfully stored video {video_id} at {video_path}")
                return video_path
            else:
                print(f"Failed to download video {video_id}")
                return None
                
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            return None
    
    async def analyze_video(self, video_id: str, url: str) -> Dict:
        """Analyze a video and return results."""
        try:
            # Download or get existing video
            video_path = await self.download_and_store_video(url, video_id)
            if not video_path:
                return {
                    "id": video_id,
                    "url": url,
                    "success": False,
                    "error": "Failed to download video"
                }
            
            # Analyze the video
            result = await analyze_video_5sec(video_path)
            
            # Add bubble_id and other fields while keeping original structure
            result["bubble_id"] = video_id  # Add bubble_id field
            result["url"] = url
            result["success"] = True
            
            # Save result
            self.save_result(video_id, result)
            
            return result
            
        except Exception as e:
            return {
                "id": video_id,
                "url": url,
                "success": False,
                "error": str(e)
            }
    
    def save_result(self, video_id: str, result: Dict):
        """Save analysis result to organized folder."""
        result_dir = os.path.join(self.results_dir, f"video_{video_id}")
        os.makedirs(result_dir, exist_ok=True)
        
        # Save detailed result
        result_file = os.path.join(result_dir, "analysis_result.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save summary
        if "video_summary" in result:
            summary_file = os.path.join(result_dir, "summary.json")
            with open(summary_file, 'w') as f:
                json.dump({"video_summary": result["video_summary"]}, f, indent=2)
    
    async def process_videos_batch(self, videos: List[Dict]) -> List[Dict]:
        """Process a batch of videos efficiently."""
        results = []
        
        print(f"Processing {len(videos)} videos...")
        
        # Process videos in parallel (limit to avoid overwhelming the system)
        semaphore = asyncio.Semaphore(3)  # Process 3 videos at a time
        
        async def process_single_video(video_data):
            async with semaphore:
                return await self.analyze_video(
                    video_data["id"], 
                    video_data["url"]
                )
        
        # Create tasks for all videos
        tasks = [process_single_video(video) for video in videos]
        
        # Process all videos
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        # Save batch results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = os.path.join(self.results_dir, f"batch_results_{timestamp}.json")
        with open(batch_file, 'w') as f:
            json.dump(processed_results, f, indent=2)
        
        return processed_results
    
    def cleanup_old_videos(self, days_old: int = 7):
        """Clean up videos older than specified days."""
        import time
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        
        cleaned_count = 0
        for video_dir in os.listdir(self.videos_dir):
            video_path = os.path.join(self.videos_dir, video_dir)
            if os.path.isdir(video_path):
                # Check if directory is older than cutoff
                dir_time = os.path.getctime(video_path)
                if dir_time < cutoff_time:
                    shutil.rmtree(video_path)
                    cleaned_count += 1
                    print(f"Cleaned up old video directory: {video_dir}")
        
        print(f"Cleaned up {cleaned_count} old video directories")
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        total_videos = 0
        total_size = 0
        
        for video_dir in os.listdir(self.videos_dir):
            video_path = os.path.join(self.videos_dir, video_dir)
            if os.path.isdir(video_path):
                total_videos += 1
                for file in os.listdir(video_path):
                    file_path = os.path.join(video_path, file)
                    if os.path.isfile(file_path):
                        total_size += os.path.getsize(file_path)
        
        return {
            "total_videos": total_videos,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "videos_directory": self.videos_dir,
            "results_directory": self.results_dir
        }

# Example usage
async def main():
    """Example of using the VideoManager."""
    manager = VideoManager()
    
    # Example videos
    videos = [
        {
            "id": "1",
            "url": "https://407c6e86862b5579c35d29a46ded3103.cdn.bubble.io/f1748870565362x632816124600704600/Jisu%20Trend.mp4"
        },
        {
            "id": "2", 
            "url": "https://407c6e86862b5579c35d29a46ded3103.cdn.bubble.io/f1750087513006x292623507889923100/Kiana%203%20v2-VEED.mp4"
        }
    ]
    
    # Process videos
    results = await manager.process_videos_batch(videos)
    
    # Print results
    for result in results:
        if result.get("success"):
            summary = result.get("video_summary", {})
            print(f"Video {result['id']}: {summary.get('people', 'Unknown')} people, {summary.get('sound', 'Unknown')} sound")
        else:
            print(f"Video {result['id']}: Failed - {result.get('error', 'Unknown error')}")
    
    # Print storage stats
    stats = manager.get_storage_stats()
    print(f"\nStorage Stats: {stats['total_videos']} videos, {stats['total_size_mb']} MB")

if __name__ == "__main__":
    asyncio.run(main()) 