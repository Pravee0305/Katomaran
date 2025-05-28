import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import {
  Box,
  Typography,
  Paper,
  Alert,
  Switch,
  FormControlLabel
} from '@mui/material';
import { styled } from '@mui/material/styles';

const WebcamContainer = styled(Paper)(({ theme }) => ({
  width: '100%',
  maxWidth: 640,
  marginBottom: theme.spacing(2),
  overflow: 'hidden',
  position: 'relative'
}));

const Canvas = styled('canvas')({
  position: 'absolute',
  top: 0,
  left: 0,
  width: '100%',
  height: '100%'
});

const Recognition = () => {
  const [isActive, setIsActive] = useState(false);
  const [alert, setAlert] = useState(null);
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const ws = useRef(null);
  const frameInterval = useRef(null);

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
      
      if (response.success) {
        drawFaceBoxes(response.faces);
      } else {
        setAlert({ severity: 'error', message: response.message });
      }
    };
  }, []);

  // Connect WebSocket on component mount
  useEffect(() => {
    connectWebSocket();
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [connectWebSocket]);

  const drawFaceBoxes = useCallback((faces) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const video = webcamRef.current.video;

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Set drawing styles
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.font = '16px Arial';
    ctx.fillStyle = '#00ff00';

    faces.forEach(face => {
      const { top, right, bottom, left } = face.location;
      const name = face.name;

      // Draw bounding box
      ctx.beginPath();
      ctx.rect(left, top, right - left, bottom - top);
      ctx.stroke();

      // Draw name label
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(left, top - 20, ctx.measureText(name).width + 10, 20);
      ctx.fillStyle = '#00ff00';
      ctx.fillText(name, left + 5, top - 5);
    });
  }, []);

  const captureFrame = useCallback(() => {
    if (webcamRef.current && ws.current?.readyState === WebSocket.OPEN) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        // Remove data:image/jpeg;base64, prefix
        const imageData = imageSrc.split(',')[1];
        
        ws.current.send(JSON.stringify({
          service: 'face_recognition',
          action: 'recognize',
          frame: imageData
        }));
      }
    }
  }, []);

  // Toggle recognition
  const handleToggle = useCallback((event) => {
    const isChecked = event.target.checked;
    setIsActive(isChecked);

    if (isChecked) {
      // Start recognition loop
      frameInterval.current = setInterval(captureFrame, 500); // 2 FPS
    } else {
      // Stop recognition loop
      if (frameInterval.current) {
        clearInterval(frameInterval.current);
        frameInterval.current = null;
      }
      // Clear canvas
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [captureFrame]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (frameInterval.current) {
        clearInterval(frameInterval.current);
      }
    };
  }, []);

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user"
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <Typography variant="h5" gutterBottom>
        Live Recognition
      </Typography>

      <FormControlLabel
        control={
          <Switch
            checked={isActive}
            onChange={handleToggle}
            color="primary"
          />
        }
        label="Enable Recognition"
        sx={{ mb: 2 }}
      />
      
      <WebcamContainer elevation={3}>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          mirrored={true}
          style={{ width: '100%', height: 'auto' }}
        />
        <Canvas ref={canvasRef} />
      </WebcamContainer>

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
  );
};

export default Recognition; 