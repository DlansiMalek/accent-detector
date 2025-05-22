# Accent Detector API

A FastAPI backend service that analyzes English accents from video URLs or uploaded files.

## Features

- Extract audio from video URLs (supports YouTube, Loom, direct MP4 links, etc.)
- Detect and classify English accents
- Provide confidence scores and explanations for accent detection

## Setup

### Prerequisites

- Python 3.11
- FFmpeg (for audio extraction)

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the API

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

## API Endpoints

### GET /
- Welcome message

### POST /analyze/url
- Analyze accent from a video URL
- Request body: `{"url": "https://example.com/video.mp4"}`

### POST /analyze/file
- Analyze accent from an uploaded video file
- Form data: `file` (video file)

## Response Format

```json
{
  "accent": "American",
  "confidence": 85.7,
  "explanation": "High confidence detection of American accent based on distinctive phonetic patterns."
}
```
