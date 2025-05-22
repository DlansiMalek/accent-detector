import React, { useState } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  CircularProgress, 
  Typography, 
  Paper,
  Alert
} from '@mui/material';
import LinkIcon from '@mui/icons-material/Link';
import SendIcon from '@mui/icons-material/Send';

interface VideoUrlInputProps {
  onSubmit: (url: string) => Promise<void>;
  isLoading: boolean;
}

const VideoUrlInput: React.FC<VideoUrlInputProps> = ({ onSubmit, isLoading }) => {
  const [url, setUrl] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Basic URL validation
    if (!url) {
      setError('Please enter a video URL');
      return;
    }
    
    try {
      new URL(url);
      setError(null);
      await onSubmit(url);
    } catch (err) {
      setError('Please enter a valid URL');
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
      <Typography variant="h6" gutterBottom>
        Enter Video URL
      </Typography>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Paste a link to a video (YouTube, Loom, direct MP4, etc.) to analyze the speaker's accent.
      </Typography>
      
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      
      <Box component="form" onSubmit={handleSubmit} noValidate>
        <TextField
          fullWidth
          label="Video URL"
          variant="outlined"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com/video.mp4"
          InputProps={{
            startAdornment: <LinkIcon color="action" sx={{ mr: 1 }} />,
          }}
          disabled={isLoading}
          sx={{ mb: 2 }}
        />
        
        <Button
          type="submit"
          variant="contained"
          color="primary"
          disabled={isLoading}
          endIcon={isLoading ? <CircularProgress size={20} /> : <SendIcon />}
        >
          {isLoading ? 'Analyzing...' : 'Analyze Accent'}
        </Button>
      </Box>
    </Paper>
  );
};

export default VideoUrlInput;
