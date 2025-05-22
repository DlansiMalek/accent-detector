import os
import tempfile
import uuid
from typing import Optional

import requests
import yt_dlp
from pydub import AudioSegment


class VideoProcessor:
    """
    Handles video processing operations including downloading videos from URLs
    and extracting audio from video files.
    """
    
    async def extract_audio_from_url(self, url: str) -> str:
        """
        Downloads a video from a URL and extracts its audio.
        
        Args:
            url: The URL of the video
            
        Returns:
            Path to the extracted audio file
        """
        # Create a temporary directory for our files
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
        
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
                'quiet': True,
            }
            
            # Convert Pydantic URL object to string if needed
            url_str = str(url)
            
            # For direct MP4 or other video file links
            if url_str.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                response = requests.get(url_str, stream=True)
                response.raise_for_status()
                
                video_path = os.path.join(temp_dir, f"{uuid.uuid4()}.mp4")
                with open(video_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract audio using pydub
                video = AudioSegment.from_file(video_path)
                video.export(audio_path, format="wav")
                
                # Clean up the video file
                os.remove(video_path)
            else:
                # Use yt-dlp for platforms like YouTube, Loom, etc.
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url_str, download=True)
                    downloaded_file = ydl.prepare_filename(info).replace(f".{info['ext']}", ".wav")
                    os.rename(downloaded_file, audio_path)
            
            return audio_path
        except Exception as e:
            raise Exception(f"Failed to extract audio from URL: {str(e)}")
    
    async def extract_audio_from_file(self, file_path: str) -> str:
        """
        Extracts audio from a video file.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            Path to the extracted audio file
        """
        try:
            # Create a temporary file for the audio
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
            
            # Extract audio using pydub
            video = AudioSegment.from_file(file_path)
            video.export(audio_path, format="wav")
            
            return audio_path
        except Exception as e:
            raise Exception(f"Failed to extract audio from file: {str(e)}")
