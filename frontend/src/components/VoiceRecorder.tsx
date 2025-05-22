import React, { useState, useRef } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  Paper,
  CircularProgress,
  Alert,
  IconButton,
  LinearProgress
} from '@mui/material';
import MicIcon from '@mui/icons-material/Mic';
import StopIcon from '@mui/icons-material/Stop';
import DeleteIcon from '@mui/icons-material/Delete';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';

interface VoiceRecorderProps {
  onRecordingComplete: (audioBlob: Blob) => Promise<void>;
  isLoading: boolean;
}

const VoiceRecorder: React.FC<VoiceRecorderProps> = ({ onRecordingComplete, isLoading }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioURL, setAudioURL] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  
  const startRecording = async () => {
    try {
      setError(null);
      
      // Reset previous recording if exists
      if (audioURL) {
        URL.revokeObjectURL(audioURL);
        setAudioURL(null);
        setAudioBlob(null);
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        setAudioURL(audioUrl);
        setAudioBlob(audioBlob);
        
        // Stop all tracks in the stream
        stream.getTracks().forEach(track => track.stop());
      };
      
      // Start recording
      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      
      // Start timer
      timerRef.current = window.setInterval(() => {
        setRecordingTime(prevTime => prevTime + 1);
      }, 1000);
      
    } catch (err: any) {
      setError(err.message || 'Failed to start recording');
      console.error('Error starting recording:', err);
    }
  };
  
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      // Clear timer
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };
  
  const handleDelete = () => {
    if (audioURL) {
      URL.revokeObjectURL(audioURL);
      setAudioURL(null);
      setAudioBlob(null);
      setRecordingTime(0);
    }
  };
  
  const handlePlayPause = () => {
    if (!audioRef.current || !audioURL) return;
    
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    
    setIsPlaying(!isPlaying);
  };
  
  const handleAudioEnded = () => {
    setIsPlaying(false);
  };
  
  const handleAnalyze = async () => {
    if (audioBlob) {
      try {
        await onRecordingComplete(audioBlob);
      } catch (err: any) {
        setError(err.message || 'Failed to analyze recording');
      }
    }
  };
  
  // Format seconds to MM:SS
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };
  
  return (
    <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
      <Typography variant="h6" gutterBottom>
        Record Your Voice
      </Typography>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Record your voice to analyze your accent. Speak clearly for at least 10 seconds for best results.
      </Typography>
      
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        {/* Hidden audio element for playback */}
        <audio 
          ref={audioRef} 
          src={audioURL || ''} 
          onEnded={handleAudioEnded} 
          style={{ display: 'none' }} 
        />
        
        {/* Recording timer and progress */}
        {(isRecording || audioURL) && (
          <Box sx={{ width: '100%', mb: 2, textAlign: 'center' }}>
            <Typography variant="h5" color={isRecording ? 'error' : 'primary'}>
              {formatTime(recordingTime)}
            </Typography>
            {isRecording && (
              <LinearProgress 
                color="error" 
                sx={{ mt: 1, height: 8, borderRadius: 4 }} 
              />
            )}
          </Box>
        )}
        
        {/* Recording controls */}
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          {!isRecording && !audioURL ? (
            <Button
              variant="contained"
              color="primary"
              startIcon={<MicIcon />}
              onClick={startRecording}
              disabled={isLoading}
              sx={{ borderRadius: 28, px: 3 }}
            >
              Start Recording
            </Button>
          ) : isRecording ? (
            <Button
              variant="contained"
              color="error"
              startIcon={<StopIcon />}
              onClick={stopRecording}
              sx={{ borderRadius: 28, px: 3 }}
            >
              Stop Recording
            </Button>
          ) : (
            <>
              <IconButton 
                color="primary" 
                onClick={handlePlayPause}
                disabled={isLoading}
                sx={{ border: '1px solid', borderColor: 'primary.main' }}
              >
                {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
              </IconButton>
              
              <IconButton 
                color="error" 
                onClick={handleDelete}
                disabled={isLoading}
                sx={{ border: '1px solid', borderColor: 'error.main' }}
              >
                <DeleteIcon />
              </IconButton>
              
              <Button
                variant="contained"
                color="primary"
                onClick={handleAnalyze}
                disabled={isLoading || !audioBlob}
                sx={{ ml: 1 }}
                startIcon={isLoading ? <CircularProgress size={20} /> : null}
              >
                {isLoading ? 'Analyzing...' : 'Analyze Accent'}
              </Button>
            </>
          )}
        </Box>
      </Box>
    </Paper>
  );
};

export default VoiceRecorder;
