import React, { useEffect, useState } from "react";
import axios from "axios";

const Dashboard = () => {
  console.log("Dashboard Component Rendered");
  const token = localStorage.getItem("token");
  const [chatHistory, setChatHistory] = useState([]);
  const [refreshCount, setRefreshCount] = useState(0);

  const fetchChats = async () => {
    console.log("Fetching chat history from backend...");
    console.log("Token used for fetch:", token);
    try {
      const res = await axios.get("http://127.0.0.1:8000/api/get_chats/", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      console.log("Chat history fetched:", res.data);
      setChatHistory(res.data);
    } catch (error) {
      console.error("Error fetching chat history:", error);
      setChatHistory([]);
    }
  };

  useEffect(() => {
    console.log("useEffect triggered: fetching chats");
    fetchChats();
  }, [token, refreshCount]);

  const handleRefresh = () => {
    console.log("Refresh button clicked");
    setRefreshCount(prev => prev + 1);
  };

  return (
    <div>
      <h2>Your Chat History</h2>
      <button onClick={handleRefresh} style={{ marginBottom: "1rem" }}>
        Refresh Chat History
      </button>
      {chatHistory.length === 0 ? (
        <p>No chat history found.</p>
      ) : (
        <ul>
          {chatHistory.map((chat, index) => (
            <li key={index} style={{ marginBottom: "1rem" }}>
              <strong>Prompt:</strong> {chat.prompt} <br />
              <strong>Response:</strong> {typeof chat.response === 'object' ? (chat.response.response || JSON.stringify(chat.response)) : chat.response} <br />
              <small>
                <em>
                  {chat.timestamp
                    ? new Date(chat.timestamp).toLocaleString()
                    : "No timestamp"}
                </em>
              </small>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default Dashboard;
