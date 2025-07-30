import os
import json
import base64
from typing import Optional, Dict, Any
import dotenv
from google.cloud import aiplatform
import google.generativeai as genai
import cv2
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import subprocess
import tempfile
from google.cloud import speech
import requests
import urllib.parse

# Load environment variables
dotenv.load_dotenv(override=True)

def rename_property(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Rename input_schema to parameters in the object."""
    new_obj = obj.copy()
    new_obj["parameters"] = new_obj["input_schema"]
    del new_obj["input_schema"]
    return new_obj

def extract_frames_first_5sec(video_path: str) -> list:
    """
    Extract frames from the first 5 seconds of video file.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        List of base64 encoded frame images (optimized for speed)
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return frames
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f} seconds")
    
    # Calculate frames for first 5 seconds
    frames_5sec = int(fps * 5)  # Total frames in 5 seconds
    print(f"Extracting frames from first 5 seconds ({frames_5sec} frames)")
    
    # Use consistent frame count for 5-second analysis
    max_frames = 15  # Consistent number of frames for all videos
    
    # Extract frames at regular intervals from first 5 seconds
    frame_interval = max(1, frames_5sec // max_frames)
    
    for i in range(0, frames_5sec, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i) 
        ret, frame = cap.read()
        
        if ret:
            # Optimize frame for faster processing
            # Resize to 256x256 (good balance between speed and quality) 
            # ~93% reduction in file size
            frame_resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
            
            # Convert to base64 with optimized compression
            # 30-40% reduction in file size on top of the resize
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # 85% quality for good balance
            _, buffer = cv2.imencode('.jpg', frame_resized, encode_param) # encode to jpg
            frame_base64 = base64.b64encode(buffer).decode('utf-8') # convert to base64
            frames.append(frame_base64)
            
            if len(frames) >= max_frames:
                break
    
    cap.release() # close the video file after extracting frames
    print(f"Extracted {len(frames)} optimized frames (256x256) from first 5 seconds of {video_path}")
    return frames

def extract_audio_first_5sec(video_path: str) -> str:
    """
    Extract audio from the first 5 seconds of video and convert to text.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Transcript text from first 5 seconds
    """
    try:
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        # Extract first 5 seconds of audio using ffmpeg
        cmd = [
            'ffmpeg', '-i', video_path,
            '-t', '5',  # Limit to 5 seconds
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM format
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            temp_audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error extracting audio: {result.stderr}")
            return ""
        
        # Read the audio file
        with open(temp_audio_path, 'rb') as audio_file:
            content = audio_file.read()
        
        # Clean up temporary file
        os.unlink(temp_audio_path)
        
        # Use Google Speech-to-Text API
        client = speech.SpeechClient()
        
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )
        
        response = client.recognize(config=config, audio=audio)
        
        # Combine all transcriptions
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + " "
        
        transcript = transcript.strip()
        print(f"Extracted transcript: '{transcript}'")
        return transcript
        
    except Exception as e:
        print(f"Error in audio extraction: {e}")
        return ""

def extract_full_audio(video_path: str) -> dict:
    """
    Extract full audio from video and convert to text with timing information.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Dictionary with full transcript and timing information
    """
    try:
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        # Extract full audio using ffmpeg
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM format
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            temp_audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error extracting full audio: {result.stderr}")
            return {"full_transcript": "", "word_timings": [], "duration": 0}
        
        # Read the audio file
        with open(temp_audio_path, 'rb') as audio_file:
            content = audio_file.read()
        
        # Clean up temporary file
        os.unlink(temp_audio_path)
        
        # Use Google Speech-to-Text API with word-level timing
        client = speech.SpeechClient()
        
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,  # Get word-level timing
        )
        
        response = client.recognize(config=config, audio=audio)
        
        # Combine all transcriptions with timing
        full_transcript = ""
        word_timings = []
        
        for result in response.results:
            full_transcript += result.alternatives[0].transcript + " "
            
            # Extract word-level timing information
            for word_info in result.alternatives[0].words:
                word_timings.append({
                    "word": word_info.word,
                    "start_time": word_info.start_time.total_seconds(),
                    "end_time": word_info.end_time.total_seconds()
                })
        
        full_transcript = full_transcript.strip()
        
        # Calculate video duration
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        print(f"Extracted full transcript ({len(word_timings)} words, {duration:.2f}s): '{full_transcript[:100]}...'")
        
        return {
            "full_transcript": full_transcript,
            "word_timings": word_timings,
            "duration": duration,
            "total_words": len(word_timings)
        }
        
    except Exception as e:
        print(f"Error in full audio extraction: {e}")
        return {"full_transcript": "", "word_timings": [], "duration": 0}

async def analyze_video_5sec(video_path: str) -> Dict[str, Any]:
    """
    Analyze the first 5 seconds of a video using Gemini Vision and transcript analysis.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Dictionary with structured video analysis
    """
    print(f"Analyzing first 5 seconds of video: {video_path}")
    
    # Extract frames and audio from first 5 seconds
    frames = extract_frames_first_5sec(video_path)
    audio_data = extract_full_audio(video_path)
    
    if not frames:
        return {"error": "No frames could be extracted from video"}
    
    print(f"Transcript: '{audio_data['full_transcript']}'")
    
    # Create enhanced prompt with transcript context
    transcript_context = f"TRANSCRIPT: '{audio_data['full_transcript']}'" if audio_data["full_transcript"] else "TRANSCRIPT: No speech detected"
        
    structured_prompt = f"""
    Analyze this video frame, the transcript, and the audio from the first 5 seconds. Identify the *most audibly dominating* sound that a viewer will remember. Return ONLY a JSON object with this exact structure:
    {{  
        "people": "Not_people_focused/Crowd/Group/Couple/Individual/Body_Part_No_Face",
        "gender": "N/A/Male/Female/Both",
        "age": "N/A/Child/Teen/Adults/Seniors",
        "composition": "Video",
        "non_human": "Wild_animals/Tech/Food/Transport/Natural_object/Pets/Man_Made_landmark/N/A",
        "setting_category": "Nature/Urban/Indoor",
        "setting_subcategory": "Skyline/Architecture_Landmark/Street_Scape/Mountains/Forest/Farms_Plains_Fields/Space/Desert/Urban_Park/Water_Ocean_Beach/Outdoor_General/Home/Restaurant/Work/School/Entertainment_Venue/N/A",
        "sound": "Voiceover_Narration/Dialog_On-Camera_Speech/Background_Music/Live_Performance_Audio/Sound_Effects_Foley/Ambient_Environmental_Sound/Text-to-Speech_AI_Voice/Silent_Muted_with_On-Screen_Text"
    }}

    {transcript_context}

    CATEGORY RULES:

    PEOPLE (choose one):
    - "Not_people_focused": No people detected through the video.
    - "Crowd": 10+ people.
    - "Group": 3-10 people.  
    - "Couple": 2 people leading the video. **Note: Do not choose couple if audio detects only one voice**
    - "Individual": 1 person.
    - "Body_Part_No_Face": Only body parts visible (hands, legs, etc.).

    GENDER (choose one):
    - "N/A": Only if people = "Not_people_focused".
    - "Male": More than 75% of the people are males.
    - "Female": More than 75% of the people are females.  
    - "Both": Both males and females visible.

    AGE (choose one):
    - "N/A": Only if people = "Not_people_focused".
    - "Child": Ages 0-12.
    - "Teen": Ages 13-19.
    - "Adults": Ages 20-65.
    - "Seniors": Ages 65+.

    COMPOSITION: Always "Video".

    NON_HUMAN (choose one):
    - "Wild_animals": Wild animals in nature.
    - "Tech": Technology, devices, computers.
    - "Food": Food, cooking, eating.
    - "Transport": Cars, planes, boats, bikes.
    - "Natural_object": Trees, mountains, natural landmarks.
    - "Pets": Domestic animals.
    - "Man_Made_landmark": Buildings, monuments, structures.
    - "N/A": No non-human focus.

    SETTING CATEGORY (choose one):
    - "Nature": Natural landscapes, forests, mountains, etc.
    - "Urban": Urban settings, buildings, streets, etc.
    - "Indoor": Indoor settings, homes, offices, etc.

    SETTING SUBCATEGORY (choose one):
    - "Skyline": Skyscrapers, cityscapes, urban views.
    - "Architecture_Landmark": Monuments, historical buildings, landmarks.
    - "Street_Scape": Streets, sidewalks, urban environments.
    - "Mountains": Mountains, valleys, hiking trails.
    - "Forest": Forests, trees, nature trails.
    - "Farms_Plains_Fields": Farms, plains, fields, agricultural areas.
    - "Space": Space, outer space, galaxies, planets.
    - "Desert": Deserts, sand dunes, arid landscapes.
    - "Urban_Park": Parks, gardens, green spaces.
    - "Water_Ocean_Beach": Oceans, beaches, coastal views.
    - "Outdoor_General": Outdoor settings, nature, parks.
    - "Home": Homes, apartments, residential areas.
    - "Restaurant": Restaurants, cafes, dining establishments.
    - "Work": Offices, workplaces, business environments.

    SOUND (choose the most dominant one):
    - "Voiceover_Narration": A single speaker explaining, storytelling, or guiding the viewer.
    - "Dialog_On-Camera_Speech": Natural conversation or monologue from people visible in the video.
    - "Background_Music": Music that sets a tone or mood but is not the primary focus.
    - "Live_Performance_Audio": Music, singing, or spoken word captured in real-time.
    - "Sound_Effects_Foley": Stylized noises (whooshes, pops) that punctuate visual actions.
    - "Ambient_Environmental_Sound": Natural or location-based audio (street noise, birds, crowd murmur).
    - "Text-to-Speech_AI_Voice": Robotic or synthesized voice narrating written text.
    - "Silent_Muted_with_On-Screen_Text": No audio or completely muted with reliance on captions or text overlays.

    IMPORTANT: Return ONLY the JSON object, no explanations or other text.
    """
        
    # Analyze frames in parallel for faster processing
    async def analyze_single_frame(frame_base64: str, frame_num: int) -> Dict[str, str]:
        """Analyze a single frame asynchronously."""
        try:
            analysis = await get_text_gemini(
                prompt=structured_prompt,
                temperature=0.1,  # Lower temperature for more consistent results
                image_base64=frame_base64,
                file_type="jpg",
                model="gemini-2.0-flash"
            )
            
            # Try to parse JSON from response
            try:
                # Clean the response to extract JSON
                analysis_clean = analysis.strip()
                if analysis_clean.startswith('```json'):
                    analysis_clean = analysis_clean[7:]
                if analysis_clean.endswith('```'):
                    analysis_clean = analysis_clean[:-3]
                analysis_clean = analysis_clean.strip()
                
                frame_data = json.loads(analysis_clean)
                print(f"✓ Frame {frame_num} analyzed successfully")
                return frame_data
            except json.JSONDecodeError:
                print(f"⚠ Could not parse JSON from frame {frame_num}, using default values")
                return {
                    "people": "Undefined",
                    "gender": "Undefined",
                    "age": "Undefined",
                    "composition": "Undefined",
                    "non_human": "Undefined",
                    "setting_category": "Undefined",
                    "setting_subcategory": "Undefined",
                    "sound": "Undefined"
                }
            
        except Exception as e:
            print(f"✗ Error analyzing frame {frame_num}: {e}")
            return {
                "people": "Undefined",
                    "gender": "Undefined",
                    "age": "Undefined",
                    "composition": "Undefined",
                    "non_human": "Undefined",
                    "setting_category": "Undefined",
                    "setting_subcategory": "Undefined",
                    "sound": "Undefined"
            }
    
    # Process frames in parallel (batch size of 5 for faster processing)
    print(f"Analyzing {len(frames)} frames in parallel batches...")
    frame_results = []
    batch_size = 5
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        batch_tasks = []
        
        for j, frame_base64 in enumerate(batch):
            frame_num = i + j + 1
            task = analyze_single_frame(frame_base64, frame_num)
            batch_tasks.append(task)
        
        # Process batch in parallel
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Handle any exceptions in batch results
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"✗ Batch processing error: {result}")
                frame_results.append({
                    "people": "Undefined",
                    "gender": "Undefined",
                    "age": "Undefined",
                    "composition": "Undefined",
                    "non_human": "Undefined",
                    "setting_category": "Undefined",
                    "setting_subcategory": "Undefined",
                    "sound": "Undefined"
                })
            else:
                frame_results.append(result)
        
        # Small delay between batches to avoid rate limits
        if i + batch_size < len(frames):
            await asyncio.sleep(0.5)
    
    # Aggregate results to get most common values
    def get_most_common(values):
        if not values:
            return "Undefined"
        # Count occurrences, excluding "Undefined" and "N/A"
        counts = {}
        for v in values:
            if v not in ["Undefined", "N/A"]:
                counts[v] = counts.get(v, 0) + 1
        if counts:
            return max(counts, key=counts.get)
        return "Undefined"
    
    # Aggregate across all frames
    people_values = [frame["people"] for frame in frame_results]
    gender_values = [frame["gender"] for frame in frame_results]
    age_values = [frame["age"] for frame in frame_results]
    composition_values = [frame["composition"] for frame in frame_results]
    non_human_values = [frame["non_human"] for frame in frame_results]
    setting_category_values = [frame["setting_category"] for frame in frame_results]
    setting_subcategory_values = [frame["setting_subcategory"] for frame in frame_results]
    sound_values = [frame["sound"] for frame in frame_results]
    
    video_summary = {
       "people": get_most_common(people_values),
       "gender": get_most_common(gender_values),
       "age": get_most_common(age_values),
       "composition": get_most_common(composition_values),
       "non_human": get_most_common(non_human_values),
       "setting_category": get_most_common(setting_category_values),
       "setting_subcategory": get_most_common(setting_subcategory_values),
       "sound": get_most_common(sound_values)
    }
    
    return {
        "video_path": video_path,
        "video_summary": video_summary,
        "total_frames_analyzed": len(frames),
        "analysis_duration": "First 5 seconds",
        "transcript": audio_data["full_transcript"],
        "has_speech": bool(audio_data["full_transcript"].strip())
    }

async def get_text_gemini(prompt: str, temperature: float = 0.7, 
                         image_base64: Optional[str] = None, 
                         file_type: Optional[str] = None, 
                         user_id: Optional[str] = None, 
                         model: str = "gemini-2.0-flash",
                         web_tools: bool = True) -> Optional[str]:
    """
    Get text response from Gemini model with optional file input using Google Generative AI.
    
    Args:
        prompt: The text prompt to send to the model
        temperature: Model temperature (0.0 to 1.0)
        image_base64: Base64 encoded file data
        file_type: Type of file (mp4, png, jpg, jpeg, mp3, etc.)
        user_id: User identifier
        model: Model name to use
        web_tools: Whether to enable web tools
    
    Returns:
        Model response text or None if failed
    """
    try:
        # Configure Google Generative AI
        genai.configure(api_key=os.getenv("GOOGLE_KEY"))
        
        # Create model
        model_instance = genai.GenerativeModel(model)
        
        # Prepare content
        content = [prompt]
        
        # Add image if provided
        if file_type and image_base64:
            mime_type_map = {
                "mp4": "video/mp4",
                "png": "image/png", 
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "mp3": "audio/mp3",
                "x-m4a": "audio/mp4",
                "mpeg": "audio/mpeg",
                "ogg": "audio/ogg",
                "wav": "audio/wav"
            }
            
            mime_type = mime_type_map.get(file_type, "application/octet-stream")
            image_data = base64.b64decode(image_base64)
            
            # Add image to content
            content.append({
                "mime_type": mime_type,
                "data": image_data
            })
        
        # Generate content
        iteration_count = 0
        final_response = None
        
        while not final_response and iteration_count < 5:
            try:
                response = model_instance.generate_content(
                    content,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=8192,
                        temperature=temperature
                    )
                )
                
                if response and response.text:
                    final_response = response.text
                else:
                    print("No valid response from the model")
                    break
                    
            except Exception as e:
                print(f"Error generating content: {e}")
                break
                
            iteration_count += 1
        
        return final_response
        
    except Exception as e:
        print(f"Error in get_text_gemini: {e}")
        return None

async def download_video(url: str, video_id: str = None) -> Optional[str]:
    """
    Download a video from a URL to a temporary file.
    
    Args:
        url: The URL of the video to download.
        video_id: Optional ID to use in filename for better organization
    
    Returns:
        Path to the downloaded video file or None if download failed.
    """
    try:
        # Parse URL to get filename
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        filename = os.path.basename(path)
        if not filename:
            filename = "downloaded_video"
        
        # Create structured folder system
        download_dir = "videos"
        if video_id:
            # Create subfolder for this video ID
            video_dir = os.path.join(download_dir, f"video_{video_id}")
            os.makedirs(video_dir, exist_ok=True)
            file_path = os.path.join(video_dir, filename)
        else:
            # Use general videos folder
            os.makedirs(download_dir, exist_ok=True)
            file_path = os.path.join(download_dir, filename)
        
        # Check if file already exists (avoid re-downloading)
        if os.path.exists(file_path):
            print(f"Video already exists at {file_path}, skipping download")
            return file_path
        
        # Use aiohttp to download the file
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(1024):
                            f.write(chunk)
                    print(f"Downloaded {url} to {file_path}")
                    return file_path
                else:
                    print(f"Failed to download {url}: HTTP status {response.status}")
                    return None
    except Exception as e:
        print(f"Error downloading video from {url}: {e}")
        return None

async def process_bubble_files_json(bubble_files_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process bubble files from JSON input and return analysis results.
    
    Args:
        bubble_files_json: JSON with structure {"bubble_files": [{"url": "...", "id": 1}, ...]}
    
    Returns:
        Dictionary with analysis results for all videos
    """
    try:
        # Validate input structure
        if 'bubble_files' not in bubble_files_json:
            return {
                "success": False,
                "error": "Missing 'bubble_files' in request"
            }
        
        bubble_files = bubble_files_json['bubble_files']
        results = []
        
        print(f"Processing {len(bubble_files)} videos...")
        
        # Process each video
        for bubble_file in bubble_files:
            print(f"\n{'='*50}")
            print(f"Processing video ID {bubble_file['id']}: {bubble_file['url']}")
            
            try:
                # Download the video
                video_path = await download_video(bubble_file['url'], str(bubble_file['id']))
                if not video_path:
                    print(f"Failed to download: {bubble_file['url']}")
                    results.append({
                        "id": bubble_file['id'],
                        "url": bubble_file['url'],
                        "error": "Failed to download video",
                        "success": False
                    })
                    continue
                
                # Analyze the video
                result = await analyze_video_5sec(video_path)
                result["id"] = bubble_file['id']
                result["url"] = bubble_file['url']
                result["success"] = True
                
                results.append(result)
                
                # Clean up downloaded file
                if os.path.exists(video_path):
                    os.remove(video_path)
                
            except Exception as e:
                print(f"Error analyzing video {bubble_file['id']}: {e}")
                results.append({
                    "id": bubble_file['id'],
                    "url": bubble_file['url'],
                    "error": str(e),
                    "success": False
                })
        
        print(f"\n{'='*50}")
        print("Analysis complete!")
        
        return {
            "success": True,
            "results": results,
            "message": f"Successfully analyzed {len([r for r in results if r.get('success', False)])} videos"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }

async def test_bubble_files_processing():
    """Test function to demonstrate bubble files JSON processing."""
    
    # Sample input JSON
    test_input = {
        "bubble_files": [
            {
                "url": "https://407c6e86862b5579c35d29a46ded3103.cdn.bubble.io/f1748870565362x632816124600704600/Jisu%20Trend.mp4",
                "id": 1
            },
            {
                "url": "https://407c6e86862b5579c35d29a46ded3103.cdn.bubble.io/f1750087513006x292623507889923100/Kiana%203%20v2-VEED.mp4",
                "id": 2
            }
        ]
    }
    
    print("Testing bubble files JSON processing...")
    result = await process_bubble_files_json(test_input)
    
    # Save result to file
    with open("bubble_files_test_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Result saved to bubble_files_test_result.json")
    print(f"Success: {result.get('success', False)}")
    print(f"Message: {result.get('message', 'No message')}")
    
    return result

async def main():
    """Main function to process videos from URLs or local files."""
    
    # Example URLs - you can modify this list or pass URLs as arguments
    video_urls = [
        # Add your video URLs here
        # "https://example.com/video1.mp4",
        # "https://example.com/video2.mp4",
    ]
    
    # Also check for local files in videos directory
    videos_dir = "videos"
    local_video_files = []
    
    if os.path.exists(videos_dir):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        for file in os.listdir(videos_dir):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                local_video_files.append(os.path.join(videos_dir, file))
    
    # Combine URLs and local files
    all_videos = []
    
    # Download videos from URLs
    for url in video_urls:
        print(f"Downloading video from: {url}")
        downloaded_path = await download_video(url)
        if downloaded_path:
            all_videos.append(downloaded_path)
        else:
            print(f"Failed to download: {url}")
    
    # Add local files
    all_videos.extend(local_video_files)
    
    if not all_videos:
        print("No videos found! Add URLs to video_urls list or place video files in 'videos' directory.")
        return
    
    print(f"Found {len(all_videos)} videos to analyze (first 5 seconds only):")
    for video in all_videos:
        print(f"  - {os.path.basename(video)}")
    
    # Analyze each video
    results = []
    total_videos = len(all_videos)
    
    for idx, video_path in enumerate(all_videos, 1):
        print(f"\n{'='*50}")
        print(f"Processing video {idx}/{total_videos}: {os.path.basename(video_path)}")
        
        start_time = asyncio.get_event_loop().time()
        result = await analyze_video_5sec(video_path)
        end_time = asyncio.get_event_loop().time()
        
        processing_time = end_time - start_time
        print(f"✓ Video processed in {processing_time:.2f} seconds")
        
        results.append(result)
        
        # Save individual result
        output_file = os.path.join("json_outputs", f"analysis_hook_{os.path.splitext(os.path.basename(video_path))[0]}.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved analysis to {output_file}")
    
    # Save combined results
    combined_file = os.path.join("json_outputs", "video_analysis_hook_results.json")
    with open(combined_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save individual video summaries
    for result in results:
        if "video_summary" in result:
            video_name = os.path.splitext(os.path.basename(result["video_path"]))[0]
            summary_file = os.path.join("json_outputs", f"{video_name}_hook_summary.json")
            with open(summary_file, 'w') as f:
                json.dump({"video_summary": result["video_summary"]}, f, indent=2)
            print(f"  - {summary_file}")
    
    print(f"\n{'='*50}")
    print("Analysis complete! Results saved to:")
    print(f"  - {combined_file} (combined results)")
    for video in all_videos:
        video_name = os.path.splitext(os.path.basename(video))[0]
        summary_file = os.path.join("json_outputs", f"{video_name}_hook_summary.json")
        print(f"  - {summary_file}")

if __name__ == "__main__":
    asyncio.run(main()) 