import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Button, 
  Box 
} from '@mui/material';
import MicIcon from '@mui/icons-material/Mic';

const Header: React.FC = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        <MicIcon sx={{ mr: 2 }} />
        <Typography
          variant="h6"
          component="div"
          sx={{ flexGrow: 1, fontWeight: 'bold' }}
        >
          Accent Detector
        </Typography>
        <Box>
          <Button 
            color="inherit" 
            component={RouterLink} 
            to="/"
          >
            Home
          </Button>
          <Button 
            color="inherit" 
            component={RouterLink} 
            to="/about"
          >
            About
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
