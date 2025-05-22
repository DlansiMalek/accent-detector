import React from 'react';
import { Box, Typography, Paper, Grid, Divider, List, ListItem, ListItemIcon, ListItemText } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import InfoIcon from '@mui/icons-material/Info';

const AboutPage: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        About the Accent Detector
      </Typography>
      
      <Divider sx={{ mb: 4 }} />
      
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <InfoIcon color="primary" sx={{ mr: 1 }} />
              <Typography variant="h5" component="h2">
                How It Works
              </Typography>
            </Box>
            
            <Typography variant="body1" paragraph>
              The Accent Detector is an application that analyzes English accents from video content. It works in several steps:
            </Typography>
            
            <List>
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Video Processing" 
                  secondary="We extract audio from your video using advanced processing techniques."
                />
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Feature Extraction" 
                  secondary="We analyze acoustic features like pitch, rhythm, and phonetic patterns."
                />
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Accent Classification" 
                  secondary="Our machine learning model identifies the accent and provides a confidence score."
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <InfoIcon color="primary" sx={{ mr: 1 }} />
              <Typography variant="h5" component="h2">
                Supported Accents
              </Typography>
            </Box>
            
            <Typography variant="body1" paragraph>
              Our system can detect and classify the following English accents:
            </Typography>
            
            <Grid container spacing={2}>
              {[
                "American", "British", "Australian", "Indian", 
                "Canadian", "Irish", "Scottish", "South African",
                "New Zealand", "Non-native"
              ].map((accent) => (
                <Grid item xs={6} key={accent}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <CheckCircleIcon color="primary" fontSize="small" sx={{ mr: 1 }} />
                    <Typography variant="body2">{accent}</Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
            
            <Typography variant="body2" color="text.secondary" sx={{ mt: 3 }}>
              The system provides a confidence score (0-100%) indicating how certain it is about the detected accent.
            </Typography>
          </Paper>
        </Grid>
        
        <Grid item xs={12}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <InfoIcon color="primary" sx={{ mr: 1 }} />
              <Typography variant="h5" component="h2">
                Technology Stack
              </Typography>
            </Box>
            
            <Typography variant="body1" paragraph>
              This application is built using modern technologies:
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>Frontend:</Typography>
                <List dense>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircleIcon color="primary" fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="React with TypeScript" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircleIcon color="primary" fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="Material UI for responsive design" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircleIcon color="primary" fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="Axios for API communication" />
                  </ListItem>
                </List>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>Backend:</Typography>
                <List dense>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircleIcon color="primary" fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="FastAPI (Python 3.11)" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircleIcon color="primary" fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="Librosa for audio processing" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircleIcon color="primary" fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary="Transformers for speech analysis" />
                  </ListItem>
                </List>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AboutPage;
