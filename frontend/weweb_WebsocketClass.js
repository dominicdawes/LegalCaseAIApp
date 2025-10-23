// Chat WebSocket for WeWeb — page-scoped singleton with health checks + etc...

// --- IDs from WeWeb ---
const WS_VAR_ID = '6f50f54e-9341-48cd-ae17-2f0402067d68';   // websocket_message variable id
const CITATIONS_VAR_ID = '31d36af3-5270-417b-9398-671748331583'; // websocketCitations var id
const RENDERED_HTML_VAR_ID = 'b83788fa-f60d-44b4-ae7b-fb46580151ad'; // websocketMessageHTMLReRender
const current_user_id = pluginVariables[/* Supabase Auth user */ '1fa0dd68-5069-436c-9a7d-3b54c340f1fa']['user']?.['id']
const session_id = variables['85f9b7d7-12bc-4b4f-8d73-991b5532d113']?.id;
const HTML_STYLE_ID_TAG = 'citation-preview-style'  // also seen in CSS injector class function onLoad [Depreacated]
const HTML_CONTAINER_SELECTOR = variables[/* citationCssId */'1a1d9dd4-7046-4866-a782-9edef5b000f3'];

// --- Derived ---
const WS_URL = `wss://law-school-study-ws.onrender.com/ws/chat/${session_id}?user_id=${current_user_id}`;

// --- Global slots (prevent dupes per page) ---
const sessionKey = `__chat_ws_${session_id}`;
const userKey = `__chat_user_${current_user_id}`;
const g = globalThis;

// Initialize user connection registry if it doesn't exist
if (!g[userKey]) {
  g[userKey] = new Set();
}

// === SMART CLEANUP & DUPLICATE PREVENTION ===

// 1. Close any OTHER active sockets for this user before creating a new one.
const connections = g[userKey];
if (connections && connections.size > 0) {
  const connectionsToClose = [...connections]; // Create a copy to iterate over safely
  
  var count_log = `🗑️ Closing : ${connectionsToClose.length} stale connections`;
  console.log(count_log);
  variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = count_log;

  connectionsToClose.forEach(existingSessionKey => {
    // IMPORTANT: Only close connections that are NOT for the current session.
    if (existingSessionKey !== sessionKey) {
      const connection = g[existingSessionKey];
      
      if (connection?.ws && (connection.ws.readyState === WebSocket.OPEN || connection.ws.readyState === WebSocket.CONNECTING)) {
        var close_log = `🧹 Closing stale connection for a different session: ${existingSessionKey.replace('__chat_ws_', '')}`;
        console.log(close_log);
        variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = close_log;

        try {
          // Use the connection's own cleanup function if it exists, then close.
          connection.cleanup?.();
          connection.ws.close(4000, 'new session started');
        } catch (err) {
          console.warn(`⚠️ Error closing stale connection ${existingSessionKey}:`, err);
        }
      }
      // Ensure stale entries are removed from the registry and global scope
      connections.delete(existingSessionKey);
      delete g[existingSessionKey];
    }
  });
}

// 2. After cleanup, check if a connection for THIS specific session already exists and preserve it.
if (g[sessionKey]?.ws?.readyState === WebSocket.OPEN) {
  var debug_log = `✅ Same WebSocket connection preserved for session ${session_id}.`;
  console.log(debug_log);
  variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = debug_log;
  return; // Exit the script; the correct socket is already running.
}

// === HELPER FUNCTIONS (PRESERVED FOR DEBUGGING) ===
function closeAllUserConnections(userId, reason = 'user cleanup') {
  const userConnKey = `__chat_user_${userId}`;
  const connections = g[userConnKey];

  if (!connections || connections.size === 0) {
    console.log(`[in main JS func]🧹 No existing connections found for user ${userId}`);
    return;
  }

  var debug_log = `[in main JS func]🧹 Closing ${connections.size} existing connections for user ${userId}`
  console.log(debug_log);
  variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = debug_log;

  const connectionsToClose = [...connections];

  connectionsToClose.forEach(sessionKey => {
    const connection = g[sessionKey];
    if (connection?.ws &&
      (connection.ws.readyState === WebSocket.OPEN ||
        connection.ws.readyState === WebSocket.CONNECTING)) {

      console.log(`🔌 Closing connection: ${sessionKey}`);
      try {
        connection.cleanup?.();
        connection.ws.close(4000, reason);
      } catch (err) {
        console.warn(`⚠️ Error closing ${sessionKey}:`, err);
      }
    }
    connections.delete(sessionKey);
    delete g[sessionKey];
  });
  console.log(`[in main JS func] ✅ User cleanup completed for ${userId}`);
}

function closeSessionConnection(sessionKey, reason = 'replaced') {
  if (g[sessionKey]?.ws &&
    (g[sessionKey].ws.readyState === WebSocket.OPEN ||
      g[sessionKey].ws.readyState === WebSocket.CONNECTING)) {

    const debug_log = `🔌 Closing existing WebSocket for session: ${sessionKey}`
    variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = debug_log;
    console.log(debug_log);

    try {
      g[sessionKey].cleanup?.();
      g[sessionKey].ws.close(4000, reason);
    } catch (err) {
      const warning_log = `⚠️ Error closing session ${sessionKey}:`
      variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = warning_log;
      console.warn(warning_log, err);
    }
  }
  g[userKey]?.delete(sessionKey);
  delete g[sessionKey];
}

// --- State ---
let streaming = '';
let isNew = true;
let hbTimer = null;
let stallTimer = null;
let pongTimer = null;
let backoff = 1000;
let reconnectAttempts = 0;
const MAX_BACKOFF = 30000;
const MAX_RECONNECT_ATTEMPTS = 10;
const PING_INTERVAL = 25000;
const PONG_TTL = 10000;
const STALL_TTL = 60000;

function setStatus(s) {
  console.log('WS status →', s);
  // Optional: update a WeWeb status variable
  // variables['status_var_id'] = s;
}

function clearResponse() {
  // clean up both websocketText and websocketCitation vars
  streaming = '';
  variables['6f50f54e-9341-48cd-ae17-2f0402067d68'] = "";
  variables[/* append websocketCitations*/'31d36af3-5270-417b-9398-671748331583'] = [];
  variables[RENDERED_HTML_VAR_ID] = ""; // Clear rendered HTML
}

// This function now *only* updates the raw markdown variable
function appendText(chunk) {
  if (isNew) {
    clearResponse();
    isNew = false;
  }
  streaming += chunk ?? '';
  variables[/* append Websocket_message*/'6f50f54e-9341-48cd-ae17-2f0402067d68'] = streaming;
}

// This function now *only* updates the citation array variable
function appendCitations(newCitations) {
  // Get the current list of citations from the WeWeb variable, default to an empty array if null/undefined
  const existingCitations = variables[CITATIONS_VAR_ID] || [];

  // --- 🐞 ADD DEBUG LOGS HERE ---
  // console.log('--- Debug: appendCitations ---');
  // console.log('Existing Citations (Before Append):', JSON.stringify(existingCitations)); // See what the variable holds
  // console.log('New Citations Received:', JSON.stringify(newCitations)); // See what just arrived
  // -----------------------------

  // Combine the existing citations with the new ones
  const updatedCitations = [...existingCitations, ...newCitations];

  // --- 🐞 ADD DEBUG LOG HERE ---
  // console.log('Updated Citations (After Append):', JSON.stringify(updatedCitations)); // See the result before saving
  // ----------------------------

  // Update the WeWeb variable with the combined list
  variables[CITATIONS_VAR_ID] = updatedCitations;

  console.log(`📚 Appended ${newCitations.length} citations. Total: ${updatedCitations.length}`);
}

function complete(msgId) {
  isNew = true;

  // Get the chat history array
  const chatHistory = variables['75a6503a-173c-43e9-b12a-a5c875cb58dc'];

  // Check if the array has items before trying to access the last one
  if (chatHistory && chatHistory.length > 0) {
    // Get the last item and set its property
    const lastMessage = chatHistory[chatHistory.length - 1];
    lastMessage['streaming_complete'] = true;
    // Or using dot notation: lastMessage.streaming_complete = true;
  }

  variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = 'Stream complete 📡';
  console.log('✅ stream_complete', msgId, `len=${streaming.length}`);
}

// other util methods 

function heartbeat(ws) {
  let waitingForPong = false;

  function sendPing() {
    if (ws.readyState !== WebSocket.OPEN) return;

    if (waitingForPong) {
      console.warn('⚠️ Still waiting for previous pong, skipping ping');
      return;
    }

    waitingForPong = true;
    const pingData = { type: 'ping', ts: Date.now() };

    try {
      ws.send(JSON.stringify(pingData));
      console.log('🏓 Sent ping');

      pongTimer = setTimeout(() => {
        if (waitingForPong) {
          console.warn('⏳ Pong timeout → reconnect');
          variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = '⏳ Pong timeout → reconnect';
          safeReconnect();
        }
      }, PONG_TTL);

    } catch (error) {
      console.error('❌ Failed to send ping:', error);
      safeReconnect();
    }
  }

  function handlePong() {
    waitingForPong = false;
    clearTimeout(pongTimer);
    console.log('💘 Got pong response');
    variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = '💘 Got pong response';
  }

  hbTimer = setInterval(sendPing, PING_INTERVAL);

  const cleanup = () => {
    clearInterval(hbTimer);
    clearTimeout(pongTimer);
    hbTimer = null;
    pongTimer = null;
    waitingForPong = false;
  };

  cleanup.handlePong = handlePong;
  return cleanup;
}

function startStallTimer() {
  clearTimeout(stallTimer);
  stallTimer = setTimeout(() => {
    console.warn('🧊 Stall detected → reconnect');
    variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = '🧊 Stall detected → reconnect';
    safeReconnect();
  }, STALL_TTL);
}

function safeReconnect() {
  const ws = g[sessionKey]?.ws;
  if (ws && ws.readyState !== WebSocket.CLOSED) {
    try {
      g[sessionKey]?.cleanup?.();
      ws.close(4001, 'reconnect');
    } catch { }
  }
}

// --- Main logic for connection 

function connect() {
  setStatus('connecting');

  if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
    console.error('💥 Max reconnection attempts reached');
    setStatus('failed');
    variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = '💥 Max reconnection attempts reached';
    return;
  }

  const ws = new WebSocket(WS_URL);
  let stopHB = () => { };
  let reconnecting = false;

  // Store in global scope and register with user
  g[sessionKey] = { ws };
  g[userKey].add(sessionKey);

  ws.onopen = () => {
    console.log('🔗 WebSocket connected');
    setStatus('connected');
    reconnectAttempts = 0;
    backoff = 1000;

    try {
      ws.send(JSON.stringify({
        type: 'hello',
        session_id,
        user_id: current_user_id
      }));
    } catch (error) {
      console.error('❌ Failed to send handshake:', error);
    }

    stopHB = heartbeat(ws);
    startStallTimer();
  };

  ws.onmessage = (evt) => {
    startStallTimer();

    let data;
    try {
      data = JSON.parse(evt.data);
    } catch (e) {
      console.error('💥 Bad JSON:', e, evt.data);
      return;
    }

    console.log('📨 Received:', data.type, data);

    // --- Helper function to re-render HTML ---
    const reRenderContent = () => {
      // 1. Get the latest raw markdown text
      const markdownText = variables[WS_VAR_ID] || '';
      if (!markdownText) return;

      // 2. Render markdown to HTML (window.md was created by your startup script)
      const html = window.md ? window.md.render(markdownText) : markdownText;

      // 3. Update the HTML variable in WeWeb
      variables[RENDERED_HTML_VAR_ID] = html;

      // 4. Re-initialize link previews on the newly rendered content
      // We use event delegation now, so we only need to do this ONCE
      // But for safety, we can call it. The new initializeLinks is smart.
      setTimeout(() => {
        if (window.linkPreviewManager) {
          // This function is now very fast because it just adds/removes listeners
          // from the *container*, not every single link.
          window.linkPreviewManager.initializeLinks(HTML_CONTAINER_SELECTOR);
        }
      }, 50);
    };

    switch (data.type) {
      case 'hello_ack':
        console.log('🤝 Handshake acknowledged');
        variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = '🤝 Handshake acknowledged';
        break;

      case 'pong':
        if (stopHB && stopHB.handlePong) {
          stopHB.handlePong();
        }
        break;

      case 'content_delta':
        appendText(data.chunk); // 1. Append raw markdown
        reRenderContent();      // 2. Re-render HTML from raw markdown during stream
        break;

      case 'stream_complete':
        complete(data.message_id);
        reRenderContent(); // Final render (end of stream)
        break;

      case 'citations_found':
        console.log('📚 Citations:', data.citations);

        // 1. Append to citations array
        appendCitations(data.citations);

        // 2. Register new citations with the link manager
        if (window.linkPreviewManager && data.citations) {
          data.citations.forEach(citation => {
            // The backend sends 'title' AND 'document_title', let's prioritize
            const title = citation.document_title || citation.title || 'Source';

            window.linkPreviewManager.registerCitation(citation.id, {
              source: title, // Use the real title from the backend
              text: citation.relevant_excerpt || `Details for ${title}, Page ${citation.page_number || 'N/A'}`,
              // You can add more fields if you send them from the backend
            });
          });
        }
        break;

      case 'message_received':
        console.log('✅ Message received:', data.message);
        variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = `✅ ${data.message}`;
        break;

      case 'task_started':
        console.log('🚀 Task started:', data.task_id);
        variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = `🚀 Task started: ${data.task_id}`;
        break;

      case 'stream_cancelled':
        console.log('🛑 Stream was cancelled');
        variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = '🛑 Stream cancelled by user';
        // Reset streaming state
        isNew = true;
        // Optional: Add cancellation notice to content
        streaming += '\n\n[Response cancelled by user]';
        variables[WS_VAR_ID] = streaming;
        break;

      case 'error':
        console.error('❌ Server error:', data.message);
        variables['e70e3b39-8cd4-4696-ba05-49843187f822'] = `❌ Error: ${data.message}`;
        break;

      default:
        console.log('ℹ️ Unhandled message type:', data.type, data);
    }
  };

  function scheduleReconnect(code, reason) {
    if (reconnecting) return;
    reconnecting = true;

    stopHB();
    clearTimeout(stallTimer);
    clearTimeout(pongTimer);

    setStatus('reconnecting');
    reconnectAttempts++;

    const delay = Math.min(backoff + Math.random() * 500, MAX_BACKOFF);
    console.log(`🔁 Reconnect attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS} in ${Math.round(delay)}ms (code: ${code}, reason: ${reason})`);

    setTimeout(() => {
      if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        connect();
      }
    }, delay);

    backoff = Math.min(backoff * 2, MAX_BACKOFF);
  }

  ws.onclose = (evt) => {
    console.log(`🔌 WebSocket closed: ${evt.code} - ${evt.reason}`);
    stopHB();
    clearTimeout(stallTimer);
    clearTimeout(pongTimer);
    setStatus('closed');

    // Remove from user registry
    g[userKey]?.delete(sessionKey);

    // Don't reconnect if manually closed
    if (evt.code !== 4000 && evt.code !== 4001 && evt.code !== 4002) {
      scheduleReconnect(evt.code, evt.reason);
    }
  };

  ws.onerror = (err) => {
    console.error('💥 WebSocket error:', err);
  };

  // Enhanced cleanup that also removes from user registry
  g[sessionKey].cleanup = () => {
    console.log('🧹 Cleaning up WebSocket');
    stopHB();
    clearTimeout(stallTimer);
    clearTimeout(pongTimer);

    // Remove from user registry
    g[userKey]?.delete(sessionKey);

    try {
      ws.close(4002, 'page unmounted');
    } catch { }
    delete g[sessionKey];
  };

  g[sessionKey].send = (message) => {
    if (ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify(message));
        return true;
      } catch (error) {
        console.error('❌ Failed to send message:', error);
        return false;
      }
    } else {
      console.warn('⚠️ WebSocket not ready, state:', ws.readyState);
      return false;
    }
  };
}

// Start the connection
connect();

// Optional: Expose global reference for debugging
if (typeof window !== 'undefined') {
  window.chatWS = g[sessionKey];
  window.cleanupUser = (userId) => closeAllUserConnections(userId);
}

// Verify connection was created successfully
setTimeout(() => {
  console.log(`🔍 POST-CREATION VERIFICATION:`);
  console.log(`Session key ${sessionKey} exists:`, !!g[sessionKey]);
  console.log(`User registry ${userKey} exists:`, !!g[userKey]);
  console.log(`WebSocket state:`, g[sessionKey]?.ws?.readyState);
  console.log(`Registry size:`, g[userKey]?.size);
}, 1000);