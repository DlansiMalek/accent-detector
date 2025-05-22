# Accent Detector Application

A full-stack application that analyzes English accents from video content. The application accepts a public video URL or uploaded video file, extracts the audio, and analyzes the speaker's accent to detect and classify English language speaking candidates.

## Deployed Application

- **Frontend**: [https://accent-detector-app.netlify.app](https://accent-detector-app.netlify.app)
- **Backend API**: [https://accent-detector-api.onrender.com](https://accent-detector-api.onrender.com)

## Features

- Accept video input via URL (YouTube, Loom, direct MP4 links, etc.) or file upload
- Extract audio from video content
- Analyze and classify English accents
- Provide confidence scores and explanations for accent detection
- Modern, responsive UI

## Project Structure

The project is organized into two main components:

- **Frontend**: React application with TypeScript and Material UI
- **Backend**: FastAPI (Python 3.11) application for video processing and accent analysis

## Deployment Instructions

### Frontend Deployment (Netlify)

1. The frontend is already deployed at [https://accent-detector-app.netlify.app](https://accent-detector-app.netlify.app)
2. To manage the deployment, claim the site using the link provided during deployment
3. For future deployments, you can connect your GitHub repository to Netlify for continuous deployment

### Backend Deployment (Render)

1. Create a free account on [Render](https://render.com/)
2. Create a new Web Service and connect your GitHub repository
3. Use the following settings:
   - Environment: Python
   - Build Command: `pip install -r backend/requirements.txt`
   - Start Command: `cd backend && gunicorn -c gunicorn.conf.py main:app`
   - Advanced Settings: Add environment variable `PORT` with value `10000`
4. Deploy the service
5. Update the frontend's `.env.production` file with your new backend URL if needed

Alternatively, you can use the `render.yaml` file included in this repository for automatic deployment configuration.

## Setup and Installation

### Prerequisites

- Node.js (v14+)
- Python 3.11
- FFmpeg (for audio extraction)

### Backend Setup

1. Navigate to the backend directory:
```bash
cd accent-detector/backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The backend API will be available at http://localhost:8000

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd accent-detector/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The frontend application will be available at http://localhost:3000

## Usage

1. Open the application in your browser
2. Choose between entering a video URL or uploading a video file
3. Submit the video for analysis
4. View the accent detection results, including:
   - Classification of the accent (e.g., British, American, Australian)
   - Confidence score (0-100%)
   - Explanation of the analysis

## Technical Implementation

### Frontend

- React with TypeScript for a type-safe frontend
- Material UI for responsive, modern UI components
- React Router for navigation
- Axios for API communication

### Backend

- FastAPI for a high-performance API
- yt-dlp for video downloading from various platforms
- pydub and librosa for audio processing
- Transformers (Wav2Vec2) for speech analysis
- Scikit-learn for machine learning components

## Accent Classification

The system can detect and classify the following English accents:
- American
- British
- Australian
- Indian
- Canadian
- Irish
- Scottish
- South African
- New Zealand
- Non-native
