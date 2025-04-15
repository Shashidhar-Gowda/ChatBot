import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import '../home.css'; // Make sure this path is correct

const Home = () => {
  const navigate = useNavigate();

  // --- Authentication useEffect (keep as is) ---
  useEffect(() => {
    const token = localStorage.getItem('token') ||
                 document.cookie.split('; ')
                   .find(row => row.startsWith('token='))
                   ?.split('=')[1];

    if (!token) {
      navigate('/login');
      return;
    }

    const verifyToken = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/api/verify_token/', {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        if (!response.ok) {
          throw new Error('Invalid token');
        }
      } catch (error) {
        localStorage.removeItem('token');
        // Clear cookie if necessary
        document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
        navigate('/login');
      }
    };

    verifyToken();
  }, [navigate]);

  const [prompt, setPrompt] = useState('');
  const [chatLog, setChatLog] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null); // Keep for scrolling

  // --- Auto-scroll useEffect (keep as is) ---
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatLog]);

  // --- handleFileUpload (keep as is, maybe update response format) ---
   const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSelectedFile(file);
    setIsLoading(true);
    // Simulate upload processing and response
    setTimeout(() => {
      setChatLog(prev => [
        ...prev,
        { text: `Uploaded file: ${file.name}`, isUser: true },
        { text: "I've received your file. How can I help with it?", isUser: false } // Standard bot response object
      ]);
      setSelectedFile(null); // Clear selection after pseudo-upload
      setIsLoading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = ""; // Reset file input
      }
    }, 1500);
  };

  // --- sendMessage (keep as is, response structure is already handled) ---
  const sendMessage = async () => {
    console.log("sendMessage called with prompt:", prompt);
    if (!prompt.trim() && !selectedFile) {
      console.log("Empty prompt and no file selected, aborting send.");
      return; // Prevent empty send
    }

    setIsLoading(true);
    const userMessage = { text: prompt, isUser: true };
    setPrompt('');
    setChatLog(prev => [...prev, userMessage]);

    try {
      const token = document.cookie.split('; ')
        .find(row => row.startsWith('token='))
        ?.split('=')[1] || localStorage.getItem('token');

      if (!token) {
        navigate('/login');
        return;
      }

      const response = await fetch('http://127.0.0.1:8000/api/get_ai_response/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ prompt })
      });

      console.log("API response status:", response.status);

      if (!response.ok) {
          if (response.status === 401) {
             navigate('/login');
             throw new Error("Authentication failed. Please login again.");
          }
          let errorData;
          try {
              errorData = await response.json();
          } catch (jsonError) {
              throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
          }
          throw new Error(errorData?.error || errorData?.detail || `Request failed with status ${response.status}`);
      }

      const data = await response.json();
      console.log("API response data:", data);

      if (data.status === 'success') {
          const responseObj = typeof data.response === 'object' ? data.response : {};
          let responseText = responseObj.response || (typeof data.response === 'string' ? data.response : 'Received response');
          const intent = responseObj.intent || null;
          const followUp = Array.isArray(responseObj.follow_up) ? responseObj.follow_up : [];

          if (typeof responseText === 'string') {
            responseText = responseText.replace(/<think>[\s\S]*?<\/think>\n*/, '').trim();
          }

          let formattedText = responseText;
          if (typeof responseText === 'string') {
              if (responseText.includes("I'll remember your name is ")) {
                  const name = responseText.replace("I'll remember your name is ", "").trim();
                  formattedText = `Got it! I'll call you ${name}`;
              }
              else if (responseText.startsWith("Your name is ")) {
                  const name = responseText.replace("Your name is ", "").trim();
                  formattedText = `Yes, your name is ${name}`;
              }
          }

          const botResponse = {
            text: formattedText,
            isUser: false,
            intent: intent && intent.trim() ? intent : null,
            followUp: followUp
          };

          setChatLog(prev => [...prev, botResponse]);
      } else {
          throw new Error(data.error || data.detail || 'API returned an error');
      }
    } catch (error) {
        console.error("API Error:", error);
        setChatLog(prev => [...prev, {
            text: `Error: ${error.message || "Sorry, I couldn't get a response."}`,
            isUser: false,
            isError: true
        }]);
    } finally {
      setIsLoading(false);
    }
  };

  // --- Conversations state (keep as is) ---
  const [conversations, setConversations] = useState([
    { id: 1, name: 'Project Discussion', date: new Date().toLocaleDateString() },
    { id: 2, name: 'Data Analysis', date: new Date().toLocaleDateString() }
  ]);

  // --- Return JSX ---
  return (
    <div className="chat-gpt-container">
      {/* --- Sidebar (keep as is) --- */}
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
        {/* --- Empty State (keep as is) --- */}
        {chatLog.length === 0 && (
          <div className="empty-state">
            <h2>BrainBot</h2>
            <p>How can I help you today?</p>
          </div>
        )}

        {/* --- Chat Log Area (Modified) --- */}
        {chatLog.length > 0 && (
          <div className="chat-log">
            {chatLog.map((message, index) => (
              <div
                key={`message-${index}`}
                // Add error class styling if message.isError is true
                className={`chat-message ${message.isUser ? 'user' : 'bot'} ${message.isError ? 'error' : ''}`}
              >
                {/* Combine intent and message content within one div */}
                <div className="message-content">
                  {/* Conditionally render intent text FIRST inside message-content */}
                  {!message.isUser && message.intent && !message.isError && ( // Don't show intent for errors
                    <div className="intent-text-integrated"> {/* Use this new class */}
                      ✨ Detected Intent: {message.intent}
                    </div>
                  )}

                  {/* Render the main message text */}
                  <p>{message.text}</p>

                  {/* Render follow-up questions AFTER main text */}
                   {!message.isUser && !message.isError && message.followUp?.length > 0 && (
                     <div className="follow-up-questions">
                       <p>Follow-up:</p>
                       {message.followUp.map((q, i) => (
                         <button
                           key={`follow-${i}`}
                           className="follow-up-btn"
                           // Set prompt and maybe trigger send automatically? Or just set prompt.
                           onClick={() => { setPrompt(q); /* Optionally call sendMessage() here */ }}
                         >
                           {q}
                         </button>
                       ))}
                     </div>
                   )}
                </div>
              </div>
            ))}
             {/* Add ref for auto-scrolling */}
             <div ref={messagesEndRef} />
          </div>
        )}

        {/* --- Chat Input Area (keep as is, maybe add error handling for file upload) --- */}
        <div className="chat-input-container">
          <div className="input-wrapper">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !isLoading && sendMessage()} // Prevent send while loading
              placeholder="Message BrainBot..."
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              className="send-button"
              disabled={isLoading || (!prompt.trim() && !selectedFile)} // Disable if loading or no input/file
            >
              {isLoading ? (
                <div className="loading-spinner" />
              ) : (
                // Send icon SVG
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                   <path d="M1.49992 14.5001L14.4999 8.00008L1.49992 1.50008L1.49992 6.83341L10.6666 8.00008L1.49992 9.16675L1.49992 14.5001Z" fill="currentColor"/>
               </svg>

              )}
            </button>
          </div>
          {/* Input Actions */}
          <div className="input-actions">
             <label className="file-upload-button">
               <input
                 type="file"
                 className="file-input"
                 onChange={handleFileUpload}
                 accept=".pdf,.doc,.docx,.txt,.csv" // Added CSV example
                 ref={fileInputRef}
                 disabled={isLoading} // Disable while loading
               />
                {/* Upload icon (example) */}
               <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M14 11V14H2V11H0V14C0 15.1 0.9 16 2 16H14C15.1 16 16 15.1 16 14V11H14ZM3 7L4.41 8.41L7 5.83V12H9V5.83L11.59 8.41L13 7L8 2L3 7Z" fill="currentColor"/>
               </svg>
               <span>{selectedFile ? selectedFile.name : 'Upload'}</span>
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