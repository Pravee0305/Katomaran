const express = require('express');
const cors = require('cors');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS
app.use(cors());

// Proxy middleware options
const faceRecognitionProxy = createProxyMiddleware({
  target: 'http://localhost:8000',
  changeOrigin: true,
  ws: true,
});

const ragEngineProxy = createProxyMiddleware({
  target: 'http://localhost:8001',
  changeOrigin: true,
  ws: true,
});

// Proxy routes
app.use('/face-recognition', faceRecognitionProxy);
app.use('/rag', ragEngineProxy);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Start server
const server = app.listen(PORT, () => {
  console.log(`Backend server running on port ${PORT}`);
});

// WebSocket proxy
server.on('upgrade', (req, socket, head) => {
  if (req.url.startsWith('/face-recognition')) {
    faceRecognitionProxy.upgrade(req, socket, head);
  } else if (req.url.startsWith('/rag')) {
    ragEngineProxy.upgrade(req, socket, head);
  }
}); 