import React, { useState, useEffect } from 'react';
import { Box, LinearProgress, Typography, Paper } from '@mui/material';
import progressSocket, { ProgressUpdate } from '../services/progressSocket';

interface ProgressBarProps {
  isLoading: boolean;
  sessionId?: string;
  onComplete?: () => void;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ isLoading, sessionId, onComplete }) => {
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('Initializing...');
  const [usingRealUpdates, setUsingRealUpdates] = useState(false);

  useEffect(() => {
    if (!isLoading) {
      setProgress(0);
      setUsingRealUpdates(false);
      return;
    }

    // If we have a session ID, connect to WebSocket for real-time updates
    if (sessionId) {
      setUsingRealUpdates(true);
      
      // Connect to the WebSocket for this session
      progressSocket.connect(sessionId, {
        onProgress: (update: ProgressUpdate) => {
          setProgress(update.progress);
          setStage(update.stage);
        },
        onComplete: () => {
          setProgress(100);
          if (onComplete) {
            onComplete();
          }
        },
        onError: (error) => {
          console.error('Progress socket error:', error);
          // Fall back to simulated progress if WebSocket fails
          setUsingRealUpdates(false);
        }
      });
      
      // Disconnect when component unmounts or loading stops
      return () => {
        progressSocket.disconnect();
      };
    }
    
    // If no session ID, use simulated progress
    if (!usingRealUpdates) {
      // Define the stages of processing
      const stages = [
        'Downloading audio...',
        'Processing audio...',
        'Extracting acoustic features...',
        'Analyzing accent patterns...',
        'Generating transcription...',
        'Calculating accent probabilities...',
        'Finalizing results...'
      ];

      let currentStage = 0;
      const totalDuration = 30000; // 30 seconds total estimated time
      const stageTime = totalDuration / stages.length;
      const intervalTime = 100; // Update every 100ms
      const stepsPerStage = stageTime / intervalTime;
      const progressPerStep = 100 / (stages.length * stepsPerStage);

      // Start progress animation
      const timer = setInterval(() => {
        setProgress((prevProgress) => {
          // Calculate new progress
          const newProgress = prevProgress + progressPerStep;
          
          // Check if we should move to next stage
          if (newProgress >= ((currentStage + 1) / stages.length) * 100) {
            currentStage = Math.min(currentStage + 1, stages.length - 1);
            setStage(stages[currentStage]);
          }
          
          // If we're at 100%, clear the interval
          if (newProgress >= 99) {
            clearInterval(timer);
            if (onComplete) {
              onComplete();
            }
            return 100;
          }
          
          return newProgress;
        });
      }, intervalTime);

      return () => {
        clearInterval(timer);
      };
    }
  }, [isLoading, sessionId, onComplete, usingRealUpdates]);

  if (!isLoading) {
    return null;
  }

  return (
    <Paper elevation={3} sx={{ p: 3, mb: 4, mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Analyzing Accent
      </Typography>
      
      <Box sx={{ width: '100%', mb: 2 }}>
        <LinearProgress 
          variant="determinate" 
          value={progress} 
          sx={{ 
            height: 10, 
            borderRadius: 5,
            '& .MuiLinearProgress-bar': {
              transition: 'transform 0.1s linear'
            }
          }} 
        />
      </Box>
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Typography variant="body2" color="text.secondary">
          {stage}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {Math.round(progress)}%
        </Typography>
      </Box>
    </Paper>
  );
};

export default ProgressBar;
