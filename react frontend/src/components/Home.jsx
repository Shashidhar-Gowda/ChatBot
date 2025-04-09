import React, { useEffect, useState } from "react";
import axios from "axios";
import "../home.css";

const ChatBox = () => {
  const [prompt, setPrompt] = useState("");
  const [chatLog, setChatLog] = useState([]);
  const [chats, setChats] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [file, setFile] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);

  const token =
    localStorage.getItem("accessToken") || sessionStorage.getItem("accessToken");

  // Fetch historical chats
  useEffect(() => {
    const fetchChats = async () => {
      try {
        const res = await axios.get("http://127.0.0.1:8000/api/get_chats/", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setChats(res.data);
      } catch (error) {
        console.error("Error fetching chat history:", error);
      }
    };

    fetchChats();
  }, []);

  const sendMessage = async () => {
    if (!prompt.trim()) return;

    const userMessage = { user: prompt, bot: "loading..." };
    setChatLog((prevLog) => [...prevLog, userMessage]);
    const currentIndex = chatLog.length;
    setPrompt("");

    const sendRequest = async (accessToken) => {
      return axios.post(
        "http://127.0.0.1:8001/api/chat",
        { prompt },
        {
          headers: {
            Authorization: `Bearer ${accessToken}`,
          },
        }
      );
    };

    const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    const token = localStorage.getItem("accessToken") || sessionStorage.getItem("accessToken");

    try {
      const response = await axios.post("http://127.0.0.1:8001/api/upload_dataset", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Authorization: `Bearer ${token}`,
        },
      });

      alert("File uploaded successfully!");
      console.log("Upload response:", response.data);
    } catch (error) {
      console.error("File upload failed:", error);
      alert("File upload failed. Check console for details.");
    }
  };

    try {
      const response = await sendRequest(token);
      const botReply = response.data.reply || "I'm not sure how to respond.";

      setChatLog((prevLog) => {
        const updatedLog = [...prevLog];
        updatedLog[currentIndex] = { ...userMessage, bot: botReply };
        return updatedLog;
      });
    } catch (error) {
      console.error("Chat error:", error);
      setChatLog((prevLog) => {
        const updatedLog = [...prevLog];
        updatedLog[currentIndex] = {
          ...userMessage,
          bot: "Oops! Something went wrong.",
        };
        return updatedLog;
      });
    }
  };

  const resetChat = async () => {
    try {
      await axios.post(
        "http://127.0.0.1:8001/api/reset_chat",
        { prompt },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      alert("Chat memory has been reset!");
    } catch (err) {
      console.error("Reset error:", err);
    }
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    const token = localStorage.getItem("accessToken") || sessionStorage.getItem("accessToken");

    try {
      const response = await axios.post("http://127.0.0.1:8001/api/upload_dataset", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Authorization: `Bearer ${token}`,
        },
      });

      alert("File uploaded successfully!");
      console.log("Upload response:", response.data);
    } catch (error) {
      console.error("File upload failed:", error);
      alert("File upload failed. Check console for details.");
    }
  };
  

  return (
    <div className="chat-container">
      <div className="chat-header">Chat with AI 🤖</div>

      {/*  Upload UI */}
      <div style={{ marginBottom: "1rem", textAlign: "center" }}>
        <input type="file" accept=".csv,.xlsx" onChange={handleFileChange} />
        <button onClick={handleUpload} style={{ marginLeft: "10px" }}>
          📁 Upload Dataset
        </button>
      </div>

      {/* Live Chat */}
      <div className="chat-log">
        {chatLog.map((entry, idx) => (
          <div key={idx} className="chat-entry">
            <div className="chat-bubble user">{entry.user}</div>
            <div className="chat-bubble bot">{entry.bot}</div>
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="chat-input">
        <input
          type="text"
          placeholder="Ask me anything..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage}>Send</button>
        <button
          onClick={resetChat}
          style={{ marginLeft: "10px", backgroundColor: "#ff4d4f", color: "white" }}
        >
          Reset
        </button>
      </div>

      {/* Toggle Button */}
      <div style={{ marginTop: "20px", textAlign: "center" }}>
        <button
          onClick={() => setShowHistory((prev) => !prev)}
          style={{
            padding: "8px 16px",
            backgroundColor: "#4CAF50",
            color: "white",
            borderRadius: "5px",
            border: "none",
            cursor: "pointer",
          }}
        >
          {showHistory ? "⬆️ Hide History" : "📋 Show Chat History"}
        </button>
      </div>

      {/* Chat History Section */}
      {showHistory && (
        <div style={{ marginTop: "2rem" }}>
          <h3 style={{ textAlign: "center" }}>Your Chat History</h3>
          {chats.length === 0 ? (
            <p style={{ textAlign: "center" }}>No previous chats found.</p>
          ) : (
            chats.map((chat, index) => (
              <div
                key={index}
                style={{
                  padding: "1rem",
                  borderBottom: "1px solid #ccc",
                  background: "#f9f9f9",
                  borderRadius: "5px",
                  margin: "10px",
                }}
              >
                <p><strong>You:</strong> {chat.prompt}</p>
                <p><strong>Bot:</strong> {chat.response}</p>
                <p style={{ fontSize: "0.8rem", color: "gray" }}>
                  {new Date(chat.timestamp).toLocaleString()}
                </p>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
};

export default ChatBox;
