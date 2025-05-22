import React from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  LinearProgress,
  Chip,
  Divider
} from '@mui/material';
import RecordVoiceOverIcon from '@mui/icons-material/RecordVoiceOver';

interface AccentResultsProps {
  accent: string;
  confidence: number;
  explanation?: string;
  transcription?: string;
  probabilities?: { [key: string]: number };
}

const AccentResults: React.FC<AccentResultsProps> = ({ 
  accent, 
  confidence, 
  explanation,
  transcription,
  probabilities
}) => {
  
  // Determine color based on confidence
  const getConfidenceColor = () => {
    if (confidence >= 80) return 'success';
    if (confidence >= 60) return 'primary';
    if (confidence >= 40) return 'warning';
    return 'error';
  };
  
  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <RecordVoiceOverIcon color="primary" sx={{ mr: 1, fontSize: 28 }} />
        <Typography variant="h5" component="h2">
          Accent Analysis Results
        </Typography>
      </Box>
      
      <Divider sx={{ mb: 3 }} />
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle1" gutterBottom>
          Detected Accent:
        </Typography>
        <Chip 
          label={accent} 
          color="primary" 
          size="medium"
          sx={{ 
            fontSize: '1.2rem', 
            py: 2.5,
            height: 'auto',
            fontWeight: 'bold' 
          }} 
        />
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle1" gutterBottom>
          Confidence Score: {confidence.toFixed(1)}%
        </Typography>
        <LinearProgress 
          variant="determinate" 
          value={confidence} 
          color={getConfidenceColor() as any}
          sx={{ height: 10, borderRadius: 5 }}
        />
      </Box>
      
      {explanation && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Explanation:
          </Typography>
          <Typography variant="body1" sx={{ fontStyle: 'italic' }}>
            {explanation}
          </Typography>
        </Box>
      )}
      
      {transcription && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Transcription:
          </Typography>
          <Paper 
            elevation={1} 
            sx={{ 
              p: 2, 
              backgroundColor: '#f5f5f5', 
              borderLeft: '4px solid #3f51b5',
              fontFamily: 'monospace'
            }}
          >
            <Typography variant="body1">
              {transcription}
            </Typography>
          </Paper>
        </Box>
      )}
      
      {probabilities && Object.keys(probabilities).length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Accent Probabilities:
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {Object.entries(probabilities)
              .sort((a, b) => b[1] - a[1]) // Sort by probability (highest first)
              // Show all probabilities
              .map(([accentName, probability]) => (
                <Box key={accentName} sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Typography variant="body2" sx={{ width: '120px', fontWeight: accentName === accent ? 'bold' : 'normal' }}>
                    {accentName}:
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={probability * 100} 
                    sx={{ 
                      flexGrow: 1, 
                      height: 8, 
                      borderRadius: 4,
                      backgroundColor: '#e0e0e0',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: accentName === accent ? '#3f51b5' : '#9fa8da'
                      }
                    }} 
                  />
                  <Typography variant="body2" sx={{ minWidth: '50px', textAlign: 'right' }}>
                    {(probability * 100).toFixed(1)}%
                  </Typography>
                </Box>
              ))}
          </Box>
        </Box>
      )}
    </Paper>
  );
};

export default AccentResults;
