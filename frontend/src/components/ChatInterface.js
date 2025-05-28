import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  List,
  ListItem,
  ListItemText,
  CircularProgress
} from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';
import { styled } from '@mui/material/styles';

const ChatContainer = styled(Paper)(({ theme }) => ({
  width: '100%',
  maxWidth: 800,
  height: 500,
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden'
}));

const MessageList = styled(List)(({ theme }) => ({
  flexGrow: 1,
  overflow: 'auto',
  padding: theme.spacing(2),
  backgroundColor: theme.palette.grey[100]
}));

const MessageItem = styled(ListItem)(({ theme, align }) => ({
  flexDirection: 'column',
  alignItems: align === 'right' ? 'flex-end' : 'flex-start',
  padding: theme.spacing(1)
}));

const MessageBubble = styled(Paper)(({ theme, variant }) => ({
  padding: theme.spacing(1, 2),
  backgroundColor: variant === 'user' ? theme.palette.primary.main : theme.palette.background.paper,
  color: variant === 'user' ? theme.palette.primary.contrastText : theme.palette.text.primary,
  maxWidth: '70%'
}));

const InputContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.paper,
  borderTop: `1px solid ${theme.palette.divider}`
}));

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const ws = useRef(null);
  const messageListRef = useRef(null);

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
      setIsLoading(false);

      if (response.success) {
        setMessages(prev => [...prev, {
          text: response.answer,
          type: 'assistant'
        }]);
      } else {
        setMessages(prev => [...prev, {
          text: 'Sorry, I encountered an error processing your question.',
          type: 'assistant'
        }]);
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

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop = messageListRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = useCallback(() => {
    if (!input.trim()) return;

    // Add user message
    setMessages(prev => [...prev, {
      text: input,
      type: 'user'
    }]);

    // Send to RAG service
    if (ws.current?.readyState === WebSocket.OPEN) {
      setIsLoading(true);
      ws.current.send(JSON.stringify({
        service: 'rag',
        action: 'query',
        query: input.trim()
      }));
      setInput('');
    } else {
      setMessages(prev => [...prev, {
        text: 'Sorry, the connection to the server was lost. Please try again.',
        type: 'assistant'
      }]);
    }
  }, [input]);

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  return (
    <Box sx={{ display: 'flex', justifyContent: 'center' }}>
      <ChatContainer elevation={3}>
        <MessageList ref={messageListRef}>
          <MessageItem>
            <MessageBubble variant="assistant">
              <Typography>
                Hello! I can help you with questions about face registrations. For example:
                <br />• "Who was the last person registered?"
                <br />• "At what time was [name] registered?"
                <br />• "How many people are currently registered?"
              </Typography>
            </MessageBubble>
          </MessageItem>

          {messages.map((message, index) => (
            <MessageItem key={index} align={message.type === 'user' ? 'right' : 'left'}>
              <MessageBubble variant={message.type}>
                <Typography>{message.text}</Typography>
              </MessageBubble>
            </MessageItem>
          ))}

          {isLoading && (
            <MessageItem align="left">
              <CircularProgress size={24} />
            </MessageItem>
          )}
        </MessageList>

        <InputContainer>
          <TextField
            fullWidth
            multiline
            maxRows={4}
            placeholder="Type your question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
            sx={{ mr: 1 }}
          />
          <IconButton
            color="primary"
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
          >
            <SendIcon />
          </IconButton>
        </InputContainer>
      </ChatContainer>
    </Box>
  );
};

export default ChatInterface; 