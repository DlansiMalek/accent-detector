/**
 * WebSocket service for real-time progress updates
 */

// Get the API URL from environment variables or use the proxy in development
const API_URL = process.env.REACT_APP_API_URL || '';
const WS_URL = API_URL.replace(/^http/, 'ws');

export interface ProgressUpdate {
  progress: number;
  stage: string;
}

export interface ProgressSocketCallbacks {
  onProgress?: (update: ProgressUpdate) => void;
  onComplete?: (result: any) => void;
  onError?: (error: any) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

class ProgressSocket {
  private socket: WebSocket | null = null;
  private sessionId: string | null = null;
  private callbacks: ProgressSocketCallbacks = {};
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimeout: NodeJS.Timeout | null = null;

  /**
   * Connect to the WebSocket server for a specific session
   * @param sessionId The session ID to connect to
   * @param callbacks Callbacks for different socket events
   */
  connect(sessionId: string, callbacks: ProgressSocketCallbacks = {}) {
    this.sessionId = sessionId;
    this.callbacks = callbacks;
    
    // Close existing connection if any
    if (this.socket) {
      this.socket.close();
    }
    
    try {
      // Create a new WebSocket connection
      this.socket = new WebSocket(`${WS_URL}/ws/${sessionId}`);
      
      // Set up event handlers
      this.socket.onopen = this.handleOpen.bind(this);
      this.socket.onmessage = this.handleMessage.bind(this);
      this.socket.onclose = this.handleClose.bind(this);
      this.socket.onerror = this.handleError.bind(this);
    } catch (error) {
      console.error('Error connecting to WebSocket:', error);
      if (this.callbacks.onError) {
        this.callbacks.onError(error);
      }
    }
  }
  
  /**
   * Disconnect from the WebSocket server
   */
  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    this.sessionId = null;
    this.reconnectAttempts = 0;
  }
  
  /**
   * Handle WebSocket open event
   */
  private handleOpen() {
    console.log('WebSocket connected');
    this.reconnectAttempts = 0;
    
    if (this.callbacks.onConnect) {
      this.callbacks.onConnect();
    }
  }
  
  /**
   * Handle WebSocket message event
   */
  private handleMessage(event: MessageEvent) {
    try {
      const data = JSON.parse(event.data);
      
      if (data.type === 'progress' && this.callbacks.onProgress) {
        this.callbacks.onProgress(data.data);
      } else if (data.type === 'complete' && this.callbacks.onComplete) {
        this.callbacks.onComplete(data.data);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }
  
  /**
   * Handle WebSocket close event
   */
  private handleClose(event: CloseEvent) {
    console.log('WebSocket disconnected:', event.code, event.reason);
    
    if (this.callbacks.onDisconnect) {
      this.callbacks.onDisconnect();
    }
    
    // Attempt to reconnect if not a normal closure
    if (event.code !== 1000 && event.code !== 1001) {
      this.attemptReconnect();
    }
  }
  
  /**
   * Handle WebSocket error event
   */
  private handleError(event: Event) {
    console.error('WebSocket error:', event);
    
    if (this.callbacks.onError) {
      this.callbacks.onError(event);
    }
  }
  
  /**
   * Attempt to reconnect to the WebSocket server
   */
  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts || !this.sessionId) {
      console.log('Max reconnect attempts reached or no session ID');
      return;
    }
    
    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    
    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    this.reconnectTimeout = setTimeout(() => {
      if (this.sessionId) {
        this.connect(this.sessionId, this.callbacks);
      }
    }, delay);
  }
}

// Export a singleton instance
export default new ProgressSocket();
