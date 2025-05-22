import os
import tempfile
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

from accent_detector import AccentDetector
from video_processor import VideoProcessor

app = FastAPI(title="Accent Detector API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://accent-detector-app.netlify.app",  # Production frontend URL
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize our services
video_processor = VideoProcessor()
accent_detector = AccentDetector()


class VideoUrlRequest(BaseModel):
    url: HttpUrl


class AccentResponse(BaseModel):
    accent: str
    confidence: float
    explanation: Optional[str] = None
    transcription: Optional[str] = None
    probabilities: Optional[dict] = None


@app.get("/")
async def root():
    """Root endpoint for health checks and API information"""
    return {
        "status": "healthy",
        "message": "Welcome to the Accent Detector API",
        "version": "1.0.0",
        "endpoints": [
            "/analyze/url",
            "/analyze/file",
            "/analyze/audio"
        ]
    }


@app.post("/analyze/url", response_model=AccentResponse)
async def analyze_from_url(request: VideoUrlRequest):
    """
    Analyze accent from a video URL
    """
    try:
        # Extract audio from the video URL
        audio_path = await video_processor.extract_audio_from_url(request.url)
        
        # Analyze the accent
        result = accent_detector.analyze_accent(audio_path)
        
        # Clean up the temporary file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/file", response_model=AccentResponse)
async def analyze_from_file(file: UploadFile = File(...)):
    """
    Analyze accent from an uploaded video file
    """
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}")
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Write the uploaded file to the temporary file
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        
        # Extract audio from the video file
        audio_path = await video_processor.extract_audio_from_file(temp_file_path)
        
        # Analyze the accent
        result = accent_detector.analyze_accent(audio_path)
        
        # Clean up the temporary files
        for path in [temp_file_path, audio_path]:
            if os.path.exists(path):
                os.remove(path)
                
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/audio", response_model=AccentResponse)
async def analyze_from_audio(file: UploadFile = File(...)):
    """
    Analyze accent directly from an uploaded audio file (WAV, MP3, etc.)
    """
    try:
        # Create a temporary file
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else 'wav'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Write the uploaded file to the temporary file
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        
        # For audio files, we can analyze directly without extraction
        result = accent_detector.analyze_accent(temp_file_path)
        
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
                
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
