import React, { useState, useRef } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  Paper,
  CircularProgress,
  Alert,
  IconButton
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import MovieIcon from '@mui/icons-material/Movie';

interface VideoFileUploadProps {
  onUpload: (file: File) => Promise<void>;
  isLoading: boolean;
}

const VideoFileUpload: React.FC<VideoFileUploadProps> = ({ onUpload, isLoading }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const file = event.target.files[0];
      
      // Check if the file is a video
      if (!file.type.startsWith('video/')) {
        setError('Please select a valid video file');
        setSelectedFile(null);
        return;
      }
      
      // Check file size (limit to 100MB)
      if (file.size > 100 * 1024 * 1024) {
        setError('File size exceeds 100MB limit');
        setSelectedFile(null);
        return;
      }
      
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a video file');
      return;
    }
    
    try {
      await onUpload(selectedFile);
    } catch (err) {
      setError('Failed to upload video');
    }
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
      <Typography variant="h6" gutterBottom>
        Upload Video File
      </Typography>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Upload a video file to analyze the speaker's accent. Supported formats: MP4, AVI, MOV, etc.
      </Typography>
      
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
          id="video-file-upload"
        />
        
        {!selectedFile ? (
          <label htmlFor="video-file-upload">
            <Button
              variant="outlined"
              component="span"
              startIcon={<CloudUploadIcon />}
              sx={{ mb: 2 }}
              disabled={isLoading}
            >
              Select Video File
            </Button>
          </label>
        ) : (
          <Box 
            sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              p: 2, 
              border: '1px dashed grey',
              borderRadius: 1,
              mb: 2,
              width: '100%'
            }}
          >
            <MovieIcon color="primary" sx={{ mr: 1 }} />
            <Typography 
              variant="body2" 
              sx={{ 
                flexGrow: 1,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap'
              }}
            >
              {selectedFile.name} ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
            </Typography>
            <IconButton 
              size="small" 
              onClick={handleClearFile}
              disabled={isLoading}
            >
              <DeleteIcon />
            </IconButton>
          </Box>
        )}
        
        <Button
          variant="contained"
          color="primary"
          onClick={handleUpload}
          disabled={!selectedFile || isLoading}
          startIcon={isLoading ? <CircularProgress size={20} /> : <CloudUploadIcon />}
        >
          {isLoading ? 'Analyzing...' : 'Analyze Accent'}
        </Button>
      </Box>
    </Paper>
  );
};

export default VideoFileUpload;
