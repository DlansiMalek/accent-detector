import axios from 'axios';

// Get the API URL from environment variables or use the ngrok URL for production
const API_URL = process.env.REACT_APP_API_URL || 'https://c37d-196-226-134-164.ngrok-free.app';

// Create an axios instance with the appropriate base URL
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  // No timeout to allow for full video processing
});

// Define the response types
export interface AccentAnalysisResponse {
  accent: string;
  confidence: number;
  explanation?: string;
  transcription?: string;
  probabilities?: { [key: string]: number };
  session_id?: string; // Session ID for WebSocket connection
}

// API methods
export const analyzeVideoUrl = async (url: string): Promise<AccentAnalysisResponse> => {
  try {
    const response = await api.post<AccentAnalysisResponse>('/analyze/url', { url });
    return response.data;
  } catch (error) {
    console.error('Error analyzing video URL:', error);
    throw error;
  }
};

export const analyzeVideoFile = async (file: File): Promise<AccentAnalysisResponse> => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post<AccentAnalysisResponse>('/analyze/file', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing video file:', error);
    throw error;
  }
};

export const analyzeAudioRecording = async (audioBlob: Blob): Promise<AccentAnalysisResponse> => {
  try {
    // Convert blob to File object with .wav extension for proper handling on the server
    const audioFile = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
    
    const formData = new FormData();
    formData.append('file', audioFile);
    
    const response = await api.post<AccentAnalysisResponse>('/analyze/audio', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing audio recording:', error);
    throw error;
  }
};

export default api;
