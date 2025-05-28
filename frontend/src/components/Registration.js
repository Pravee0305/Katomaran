import React, { useState, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
  Alert,
  CircularProgress
} from '@mui/material';
import { styled } from '@mui/material/styles';

const WebcamContainer = styled(Paper)(({ theme }) => ({
  width: '100%',
  maxWidth: 640,
  marginBottom: theme.spacing(2),
  overflow: 'hidden',
  position: 'relative'
}));

const Registration = () => {
  const [name, setName] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);
  const [alert, setAlert] = useState(null);
  const webcamRef = useRef(null);
  const ws = useRef(null);

  // Initialize WebSocket connection
  const connectWebSocket = useCallback(() => {
    ws.current = new WebSocket('ws://localhost:3001');
    
    ws.current.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.current.onclose = () => {
      console.log('WebSocket disconnected');
      setTimeout(connectWebSocket, 3000);
    };

    ws.current.onmessage = (event) => {
      const response = JSON.parse(event.data);
      setIsRegistering(false);

      if (response.success) {
        setAlert({ severity: 'success', message: response.message });
        setName('');
      } else {
        setAlert({ severity: 'error', message: response.message });
      }
    };
  }, []);

  // Connect WebSocket on component mount
  React.useEffect(() => {
    connectWebSocket();
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [connectWebSocket]);

  const handleRegister = useCallback(() => {
    if (!name.trim()) {
      setAlert({ severity: 'error', message: 'Please enter a name' });
      return;
    }

    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) {
      setAlert({ severity: 'error', message: 'Could not capture image' });
      return;
    }

    setIsRegistering(true);
    setAlert(null);

    // Remove data:image/jpeg;base64, prefix
    const imageData = imageSrc.split(',')[1];

    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        service: 'face_recognition',
        action: 'register',
        name: name.trim(),
        frame: imageData
      }));
    } else {
      setIsRegistering(false);
      setAlert({ severity: 'error', message: 'WebSocket connection lost' });
    }
  }, [name]);

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user"
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <Typography variant="h5" gutterBottom>
        Face Registration
      </Typography>
      
      <WebcamContainer elevation={3}>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          mirrored={true}
          style={{ width: '100%', height: 'auto' }}
        />
      </WebcamContainer>

      <Box sx={{ width: '100%', maxWidth: 640 }}>
        <TextField
          fullWidth
          label="Name"
          variant="outlined"
          value={name}
          onChange={(e) => setName(e.target.value)}
          disabled={isRegistering}
          sx={{ mb: 2 }}
        />

        <Button
          fullWidth
          variant="contained"
          color="primary"
          onClick={handleRegister}
          disabled={isRegistering || !name.trim()}
        >
          {isRegistering ? (
            <CircularProgress size={24} color="inherit" />
          ) : (
            'Register Face'
          )}
        </Button>

        {alert && (
          <Alert 
            severity={alert.severity}
            sx={{ mt: 2 }}
            onClose={() => setAlert(null)}
          >
            {alert.message}
          </Alert>
        )}
      </Box>
    </Box>
  );
};

export default Registration; 