import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, Title, Tooltip, Legend, LineElement, PointElement, CategoryScale, LinearScale } from 'chart.js';
import '../home.css'; // Path to your CSS
import axios from 'axios';
import { authFetch } from './utils';

ChartJS.register(Title, Tooltip, Legend, LineElement, PointElement, CategoryScale, LinearScale);

const Home = () => {
    const navigate = useNavigate();
    const [prompt, setPrompt] = useState('');
    const [chatLog, setChatLog] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [currentSessionId, setCurrentSessionId] = useState(null);
    const [selectedFile, setSelectedFile] = useState(null);
    const [conversations, setConversations] = useState([]); // For sidebar
    const fileInputRef = useRef(null); // For file input
    const messagesEndRef = useRef(null);
    const [chartData, setChartData] = useState(null); // Data for visualization

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatLog]);

    useEffect(() => {
        const fetchChatHistory = async () => {
            try {
                const token = localStorage.getItem('token');
                if (!token) {
                    console.error('No token found');
                    return;
                }
        
                const response = await fetch('/api/get_grouped_chat_history/', {
                    method: 'GET',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                    }
                });
                
                if (response.ok) {
                    const data = await response.json();
                    console.log('Chat History:', data);
                } else {
                    const errorText = await response.text();
                    console.error('Failed to load chat history:', errorText);
                }
            } catch (error) {
                console.error('Error fetching chat history:', error);
            }
        };
        fetchChatHistory();
    }, []);

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        setSelectedFile(file);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const token = document.cookie.split('; ').find(row => row.startsWith('token='))
                ?.split('=')[1] || localStorage.getItem('token');

            let response = await authFetch('http://127.0.0.1:8000/upload-file/', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}: ${await response.text()}`);
            }

            const data = await response.json();
            console.log("File upload response:", data);
            setChatLog(prev => [...prev, { text: `File "${file.name}" uploaded successfully.`, isUser: false }]);
            setCurrentSessionId(data.file_id);

        } catch (error) {
            console.error("File upload error:", error);
            setChatLog(prev => [...prev, { text: `Error uploading file: ${error.message}`, isUser: false, isError: true }]);
        } finally {
            setIsLoading(false);
            setSelectedFile(null);
            if (fileInputRef.current) {
                fileInputRef.current.value = null;
            }
        }
    };

    const sendMessage = async () => {
        if (!prompt.trim()) return;
        setIsLoading(true);
      
        const userMessage = { text: prompt, isUser: true };
        setChatLog(prev => [...prev, userMessage]);
      
        try {
          const token = document.cookie.split('; ').find(row => row.startsWith('token='))
            ?.split('=')[1] || localStorage.getItem('token');
      
          let response;
          if (currentSessionId) {
            response = await authFetch('http://127.0.0.1:8000/api/get_ai_response/', {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ prompt, file_id: currentSessionId })
            });
          } else {
            response = await fetch('http://127.0.0.1:8000/api/get_ai_response/', {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ prompt })
            });
          }
      
          if (!response.ok) {
            throw new Error(`HTTP error ${response.status}: ${await response.text()}`);
          }
      
          const data = await response.json();
          console.log("Backend Data:", data);
          
          let botResponse;
          
          if (data.type === 'visualization') {
            console.log("Full visualization response:", data);
            const visualizationData = processVisualizationData(data);
            console.log("Processed visualization data:", visualizationData);
            
            botResponse = {
                text: visualizationData ? 'Here is your visualization:' : 'Could not generate visualization',
                isUser: false,
                visualizationData: visualizationData,
                intent: data.response?.intent || 'VISUALIZATION'
            };
          } else {
            const { bot_reply, intent, matched_columns, query_used } = data.response || {};
            botResponse = {
              text: bot_reply || 'No response.',
              isUser: false,
              intent: intent || null,
              matchedColumns: currentSessionId ? matched_columns || [] : [],
              queryUsed: currentSessionId ? query_used || null : null,
            };
          }
          
          setChatLog(prev => [...prev, botResponse]);
          

      
        } catch (error) {
          console.error("Fetch error:", error);
          setChatLog(prev => [...prev, { text: `Error: ${error.message}`, isUser: false, isError: true }]);
        } finally {
          setIsLoading(false);
          setPrompt('');
        }
    };

    const startNewChat = () => {
        setChatLog([]);
        setCurrentSessionId(null); // Reset session id
    };

    // Chart Data Processing (using Chart.js)
    const processVisualizationData = (response) => {
        try {
            // Handle both direct response and nested response structures
            const visualizationData = response.response?.response || response;
            
            if (!visualizationData) {
                console.error("No visualization data found in response:", response);
                return null;
            }
    
            const imageUrl = visualizationData.image_url || visualizationData.response?.image_url;
            
            if (!imageUrl) {
                console.error("No image_url found in visualization data:", visualizationData);
                return null;
            }
            
            return {
                imageUrl: imageUrl,
                type: visualizationData.type || 'scatter',
                title: visualizationData.title || 'Data Visualization',
                xField: visualizationData.x_field,
                yField: visualizationData.y_field
            };
        } catch (error) {
            console.error("Error processing visualization data:", error);
            return null;
        }
    };
    
    // Add this right before your return statement
    const VisualizationComponent = ({ data }) => {
        if (!data || !data.imageUrl) {
            console.error("Invalid visualization data:", data);
            return null;
        }
      
        // Construct proper URL - handle cases where imageUrl might already have the base URL
        const fullImageUrl = data.imageUrl.startsWith('http') 
            ? data.imageUrl 
            : `http://127.0.0.1:8000${data.imageUrl}`;
    
        return (
          <div className="visualization-container">
            <h3>{data.title}</h3>
            <div className="visualization-image">
              <img 
                src={fullImageUrl}
                alt={data.title}
                style={{ maxWidth: '100%', height: 'auto' }}
                onError={(e) => {
                    console.error("Failed to load image:", fullImageUrl);
                    e.target.style.display = 'none';
                }}
              />
            </div>
            {data.xField && data.yField && (
                <div className="visualization-details">
                  <p>X-axis: {data.xField}</p>
                  <p>Y-axis: {data.yField}</p>
                </div>
            )}
          </div>
        );
    };
  
    return (
        <div className="chat-gpt-container">
            <div className="sidebar">
                <button
                    className="new-chat-btn"
                    onClick={startNewChat} // Add new chat functionality
                >
                    <span>+</span> New Chat
                </button>
                <div className="conversations-list">
                    {Object.keys(conversations).map(section => (
                        <div key={section}>
                            <div className="conversation-section-header">{section}</div>
                            {conversations[section].map((chat, i) => (
                                <div key={`${section}-${i}`} className="conversation-item">
                                    <div className="conversation-name">{chat.prompt.slice(0, 30)}...</div>
                                    <div className="conversation-date">{new Date(chat.timestamp).toLocaleString()}</div>
                                </div>
                            ))}
                        </div>
                    ))}
                </div>
            </div>

            <div className="main-content">
                {/* Empty State (as is) */}
                {chatLog.length === 0 && (
                    <div className="empty-state">
                        <h2>BrainBot</h2>
                        <p>How can I help you today?</p>
                    </div>
                )}

                {/* Chat Log Area */}
                {chatLog.length > 0 && (
                    <div className="chat-log">
                        {chatLog.map((message, index) => (
                            <div key={`message-${index}`} className={`chat-message ${message.isUser ? 'user' : 'bot'} ${message.isError ? 'error' : ''}`}>
                                <div className="message-content">
                                    {!message.isUser && message.intent && !message.isError && (
                                        <div className="intent-text-integrated">
                                            ✨ Detected Intent: {message.intent}
                                        </div>
                                    )}
                                    {!message.isUser && message.matchedColumns?.length > 0 && (
                                        <div className="intent-text-integrated">📊 Matched Columns: {message.matchedColumns.join(", ")}</div>
                                    )}
                                    {!message.isUser && message.queryUsed && (
                                        <div className="intent-text-integrated">🧠 Rewritten Prompt: {message.queryUsed}</div>
                                    )}
                                    <p>{message.text}</p>
                                    {message.visualizationData && (
                                        <VisualizationComponent data={message.visualizationData} />
                                        )}
                                </div>
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>
                )}

                {/* Data Visualization */}
                {chartData && (
                    <div className="visualization-container">
                        <h3>Data Visualization</h3>
                        <Line data={processChartData()} />
                    </div>
                )}

                {/* Chat Input */}
                <div className="chat-input-container">
                    <div className="input-wrapper">
                        <input
                            type="text"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && !isLoading && sendMessage()}
                            placeholder="Message BrainBot..."
                            disabled={isLoading}
                        />
                        <button
                            onClick={sendMessage}
                            className="send-button"
                            disabled={isLoading || (!prompt.trim() && !selectedFile)}
                        >
                            {isLoading ? (
                                <div className="loading-spinner" />
                            ) : (
                                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M1.49992 14.5001L14.4999 8.00008L1.49992 1.50008L1.49992 6.83341L10.6666 8.00008L1.49992 9.16675L1.49992 14.5001Z" fill="currentColor"/>
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
                                accept=".pdf,.doc,.docx,.txt,.csv"
                                ref={fileInputRef}
                                disabled={isLoading}
                            />
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
