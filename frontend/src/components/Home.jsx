import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import '../home.css'; // Path to your CSS
import axios from 'axios';
import { authFetch } from './utils';

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
    const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatLog]);

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    
    useEffect(() => {
        const fetchChatHistory = async () => {
            try {
                // Check if token is available in localStorage (or any other source)
                const token = localStorage.getItem('token'); // Replace with the correct key or source
        
                if (!token) {
                    console.error('No token found');
                    return; // Early exit if no token is available
                }
        
                const response = await fetch(`${API_BASE_URL}/api/get_grouped_chat_history/`, {
                    method: 'GET',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                    }
                });
                
                if (response.ok) {
                    const data = await response.json();
                    console.log('Chat History:', data);
                    // Handle the data
                } else {
                    const errorText = await response.text();
                    console.error('Failed to load chat history:', errorText);
                    // Handle the error appropriately (show error message to user)
                }
            } catch (error) {
                console.error('Error fetching chat history:', error);
                // Handle the network error
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

        let response = await authFetch(`${API_BASE_URL}/upload-file/`, {
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
        setCurrentSessionId(data.file_id); // Store the file_id as currentSessionId temporarily

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
            // Send file_id if available
            response = await authFetch(`${API_BASE_URL}/api/get_ai_response/`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt, file_id: currentSessionId })
            });
        } else {
            // Send only prompt if no file_id
            response = await fetch(`${API_BASE_URL}/api/get_ai_response/`, {
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
        const { bot_reply, intent, matched_columns, query_used } = data.response || {};
        const isFileQuery = !!currentSessionId;

        const botResponse = {
            text: bot_reply || 'No response.',
            isUser: false,
            intent: intent || null,
            matchedColumns: isFileQuery ? matched_columns || [] : [],
            queryUsed: isFileQuery ? query_used || null : null,
            followUp: []
        };


        

        setChatLog(prev => [...prev, botResponse]);
        console.log("Response from server:", data);

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


    // --- Return JSX ---
    return (
        <div className="chat-gpt-container">
            {/* --- Sidebar (keep as is) --- */}
            <div className="sidebar">
                <button
                    className="new-chat-btn"
                    onClick={startNewChat} // Add new chat functionality
                >
                    <span>+</span> New Chat
                </button>
                <div className="conversations-list">
                    {/* Insert the following code here */}
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
                                    {!message.isUser && message.intent && !message.isError && (
                                        <div className="intent-text-integrated">
                                            ✨ Detected Intent: {message.intent}
                                        </div>
                                    )}
                                    {/*Only show if matchedColumns exist (i.e., file was used) */}
                                        {!message.isUser && message.matchedColumns?.length > 0 && (
                                            <div className="intent-text-integrated">📊 Matched Columns: {message.matchedColumns.join(", ")}</div>
                                    )}

                                        {!message.isUser && message.queryUsed && (
                                            <div className="intent-text-integrated">🧠 Rewritten Prompt: {message.queryUsed}</div>
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
                                                    onClick={() => { setPrompt(q); }}
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
