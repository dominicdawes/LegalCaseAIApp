// websocket-server.js (Node.js + Socket.io)
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const redis = require('redis');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "https://your-weweb-app.com",
    methods: ["GET", "POST"]
  }
});

// Redis subscriber
const subscriber = redis.createClient({
  url: process.env.REDIS_URL
});

subscriber.connect();

io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  // Join specific chat session room
  socket.on('join_chat', ({ sessionId, userId }) => {
    socket.join(`chat:${sessionId}`);
    console.log(`User ${userId} joined chat:${sessionId}`);
  });

  // Handle disconnect
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Bridge Redis pub/sub to WebSocket
subscriber.subscribe('chat:*', (message, channel) => {
  try {
    const data = JSON.parse(message);
    const sessionId = channel.split(':')[1];
    
    // Broadcast to specific chat room
    io.to(`chat:${sessionId}`).emit('streaming_event', data);
  } catch (error) {
    console.error('Error parsing Redis message:', error);
  }
});

server.listen(3001, () => {
  console.log('WebSocket server running on port 3001');
});