import React, { useState, useRef, useEffect } from 'react';
import '../home.css';

const Home = () => {
  const [prompt, setPrompt] = useState('');
  const [chatLog, setChatLog] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatLog]);

  const handleFileUpload = (e) => {
    setSelectedFile(e.target.files[0]);
    // Auto-submit file
    if (e.target.files[0]) {
      setIsLoading(true);
      setTimeout(() => {
        setChatLog([...chatLog, 
          { text: `Uploaded file: ${e.target.files[0].name}`, isUser: true },
          { text: "I've received your file. How can I help with it?", isUser: false }
        ]);
        setSelectedFile(null);
        setIsLoading(false);
      }, 1500);
    }
  };

  const sendMessage = async () => {
    if (!prompt.trim()) return;
    
    setIsLoading(true);
    const userMessage = { text: prompt, isUser: true };
    setChatLog([...chatLog, userMessage]);
    setPrompt('');
    
    try {
      // Call your backend API here
      // Get token from cookies or auth context
      const token = document.cookie.split('; ')
        .find(row => row.startsWith('token='))
        ?.split('=')[1] || localStorage.getItem('token');
      
      if (!token) {
        window.location.href = '/login';
        return;
      }

      const response = await fetch('http://127.0.0.1:8000/api/get_ai_response/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        credentials: 'include',
        body: JSON.stringify({ prompt })
      });
      
      const data = await response.json();
      if (data.status === 'success') {
          // Clean up name remembering response
          let responseText = data.response;
          // Format name remembering responses
          if (data.response.includes("I'll remember your name is ")) {
              const name = data.response.replace("I'll remember your name is ", "").trim();
              responseText = `Got it! I'll call you ${name}`;
          } 
          // Format name recall responses
          else if (data.response.startsWith("Your name is ")) {
              const name = data.response.replace("Your name is ", "").trim();
              responseText = `Yes, your name is ${name}`;
          }
              
          setChatLog(prev => [...prev, { 
              text: responseText,
              isUser: false 
          }]);
      } else {
          throw new Error(data.error || 'Unknown error');
      }
    } catch (error) {
        console.error("API Error:", error);
        let errorMessage = "Sorry, I'm having trouble responding. Please try again.";
        if (error.response) {
            if (error.response.status === 405) {
                errorMessage = "Invalid request method. Please try again.";
            } else if (error.response.status === 401) {
                errorMessage = "Please login again.";
                window.location.href = '/login';
            } else if (error.response.data?.error) {
                errorMessage = error.response.data.error;
            } else if (error.response.data?.response) {
                errorMessage = error.response.data.response;
            }
        } else if (error.message) {
            errorMessage = error.message;
        }
        setChatLog(prev => [...prev, { 
            text: `Error: ${errorMessage}`,
            isUser: false,
            isError: true 
        }]);
    } finally {
      setIsLoading(false);
    }
  };

  const [conversations, setConversations] = useState([
    { id: 1, name: 'Project Discussion', date: new Date().toLocaleDateString() },
    { id: 2, name: 'Data Analysis', date: new Date().toLocaleDateString() }
  ]);

  return (
    <div className="chat-gpt-container">
      <div className="sidebar">
        <button className="new-chat-btn">
          <span>+</span> New Chat
        </button>
        <div className="conversations-list">
          {conversations.map(conv => (
            <div key={conv.id} className="conversation-item">
              <div className="conversation-name">{conv.name}</div>
              <div className="conversation-date">{conv.date}</div>
            </div>
          ))}
        </div>
      </div>
      <div className="main-content">
        {chatLog.length === 0 && (
          <div className="empty-state">
            <h2>BrainBot</h2>
            <p>How can I help you today?</p>
          </div>
        )}

        {chatLog.length > 0 && (
          <div className="chat-log">
            {chatLog.map((message, index) => (
              <div key={index} className={`chat-message ${message.isUser ? 'user' : 'bot'}`}>
                <div className="message-content">
                  <p>{message.text}</p>
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="chat-input-container">
          <div className="input-wrapper">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Message BrainBot..."
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              className="send-button"
              disabled={isLoading || !prompt.trim()}
            >
              {isLoading ? (
                <div className="loading-spinner" />
              ) : (
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M1 1L15 8L1 15" stroke="currentColor" strokeWidth="2" />
                </svg>
              )}
            </button>
          </div>
          <div className="input-actions">
            <label className="file-upload-button">
              <input
                type="file"
                className="file-input"
                onChange={handleFileUpload}
                accept=".pdf,.doc,.docx,.txt"
                ref={fileInputRef}
              />
              <span>Upload</span>
            </label>
            <div className="disclaimer">
              BrainBot can make mistakes. Consider checking important information.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
