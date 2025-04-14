// components/AskFile.js
import React, { useState, useRef } from "react";
import axios from "axios";
import "./AskFile.css";

const AskFile = () => {
  const [file, setFile] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [currentAnswer, setCurrentAnswer] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const answerBoxRef = useRef(null);

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file first");
      return;
    }

    setIsLoading(true);
    setError("");
    setSuccess("");
    
    const token = localStorage.getItem("accessToken");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("prompt", prompt);

    try {
      const res = await axios.post("http://127.0.0.1:8001/api/ask-file", formData, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "multipart/form-data",
        },
      });

      const newEntry = {
        prompt: prompt,
        answer: res.data.answer,
        timestamp: new Date().toISOString(),
        file: file.name
      };
      setChatHistory([...chatHistory, newEntry]);
      setCurrentAnswer(res.data.answer);
      setSuccess("File uploaded successfully!");
      
      // Auto-scroll to bottom when new answer arrives
      setTimeout(() => {
        if (answerBoxRef.current) {
          answerBoxRef.current.scrollIntoView({ 
            behavior: 'smooth',
            block: 'nearest'
          });
        }
      }, 100);
    } catch (err) {
      setError(err.response?.data?.message || "Failed to upload file");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.size > 10 * 1024 * 1024) { // 10MB limit
        setError("File size exceeds 10MB limit");
        return;
      }
      if (!selectedFile.name.match(/\.(csv|xlsx?|json)$/i)) {
        setError("Only CSV, Excel, and JSON files are allowed");
        return;
      }
      setFile(selectedFile);
      setError("");
    }
  };

  return (
    <div className="ask-file-container">
      <h2>Upload Data for Analysis</h2>
      
      <div className="file-upload-section">
        <label className="file-upload-label">
          <input 
            type="file"
            accept=".csv,.xlsx,.xls,.json"
            onChange={handleFileChange}
            className="file-input"
          />
          <span className="file-upload-button">Choose File</span>
          {file && <span className="file-name">{file.name}</span>}
        </label>

        <div className="file-requirements">
          <p>Requirements:</p>
          <ul>
            <li>CSV or Excel format</li>
            <li>Max size: 10MB</li>
            <li>Should contain header row</li>
          </ul>
        </div>
      </div>

      {error && <div className="error-message">{error}</div>}
      {success && <div className="success-message">{success}</div>}

      <div className="prompt-section">
        <input 
          type="text" 
          placeholder="What would you like to analyze?" 
          value={prompt} 
          onChange={(e) => setPrompt(e.target.value)}
          className="prompt-input"
        />
        <button 
          onClick={handleUpload} 
          disabled={isLoading || !file}
          className="upload-button"
        >
          {isLoading ? "Processing..." : "Analyze"}
        </button>
        <button 
          className="new-chat-button"
          onClick={() => {
            setFile(null);
            setPrompt("");
            setCurrentAnswer("");
            setError("");
            setSuccess("");
            setChatHistory([]);
          }}
          disabled={isLoading}
        >
          New Chat
        </button>
      </div>

      <div className="chat-history">
        {chatHistory.map((chat, index) => (
          <div key={index} className="chat-entry">
            <div className="chat-prompt">
              <strong>You:</strong> {chat.prompt}
              {chat.file && <div className="chat-file">File: {chat.file}</div>}
            </div>
            <div className="chat-answer">
              <strong>Analysis:</strong> {chat.answer}
            </div>
          </div>
        ))}
        {currentAnswer && (
          <div className="answer-box" ref={answerBoxRef}>
            <h3>Analysis Results:</h3>
            <p>{currentAnswer}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default AskFile;
