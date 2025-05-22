# Accent Detector API

This is the backend API for the Accent Detector application. It analyzes audio from video files to detect and classify English accents.

## Features

- Extract audio from video URLs or uploaded files
- Analyze and classify English accents
- Provide confidence scores and explanations for accent detection
- Return transcription of the audio

## API Endpoints

- `/` - Health check and API information
- `/analyze/url` - Analyze accent from a video URL
- `/analyze/file` - Analyze accent from an uploaded video file
- `/analyze/audio` - Analyze accent from an uploaded audio file

## Frontend

The frontend for this application is deployed at [https://accent-detector-app.netlify.app](https://accent-detector-app.netlify.app)
