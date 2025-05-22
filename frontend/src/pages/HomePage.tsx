import React, { useState } from 'react';
import { Box, Typography, Divider, Alert, Paper, Tab, Tabs } from '@mui/material';
import VideoUrlInput from '../components/VideoUrlInput';
import AccentResults from '../components/AccentResults';
import ProgressBar from '../components/ProgressBar';
import FileUploadIcon from '@mui/icons-material/FileUpload';
import LinkIcon from '@mui/icons-material/Link';
import MicIcon from '@mui/icons-material/Mic';
import { analyzeVideoUrl, analyzeVideoFile, analyzeAudioRecording, AccentAnalysisResponse } from '../services/api';
import VideoFileUpload from '../components/VideoFileUpload';
import VoiceRecorder from '../components/VoiceRecorder';

const HomePage: React.FC = () => {
  const [results, setResults] = useState<AccentAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
    // Reset results when changing tabs
    setResults(null);
    setError(null);
  };

  const handleUrlSubmit = async (url: string) => {
    setLoading(true);
    setError(null);
    setSessionId(undefined);
    
    try {
      const response = await analyzeVideoUrl(url);
      setResults(response);
      // Store the session ID for WebSocket connection
      if (response.session_id) {
        setSessionId(response.session_id);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to analyze video. Please try again.');
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    setSessionId(undefined);
    
    try {
      const response = await analyzeVideoFile(file);
      setResults(response);
      // Store the session ID for WebSocket connection
      if (response.session_id) {
        setSessionId(response.session_id);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to analyze video. Please try again.');
      setResults(null);
    } finally {
      setLoading(false);
    }
  };
  
  const handleAudioRecording = async (audioBlob: Blob) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await analyzeAudioRecording(audioBlob);
      setResults(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to analyze audio recording. Please try again.');
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          English Accent Detector
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Analyze speaker accents from video content
        </Typography>
      </Box>

      <Divider sx={{ mb: 4 }} />

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Paper sx={{ mb: 4 }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          variant="fullWidth"
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab icon={<LinkIcon />} label="VIDEO URL" />
          <Tab icon={<FileUploadIcon />} label="UPLOAD VIDEO" />
          <Tab icon={<MicIcon />} label="RECORD VOICE" />
        </Tabs>
      </Paper>

      {tabValue === 0 && (
        <VideoUrlInput onSubmit={handleUrlSubmit} isLoading={loading} />
      )}

      {tabValue === 1 && (
        <VideoFileUpload onUpload={handleFileUpload} isLoading={loading} />
      )}
      
      {tabValue === 2 && (
        <VoiceRecorder onRecordingComplete={handleAudioRecording} isLoading={loading} />
      )}
      
      <ProgressBar isLoading={loading} />

      {results && !loading && (
        <AccentResults
          accent={results.accent}
          confidence={results.confidence}
          explanation={results.explanation}
          transcription={results.transcription}
          probabilities={results.probabilities}
        />
      )}
    </Box>
  );
};

export default HomePage;
