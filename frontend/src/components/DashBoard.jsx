import React, { useEffect, useState } from "react";
import axios from "axios";

const Dashboard = () => {
  const token = localStorage.getItem("token");
  const [chatHistory, setChatHistory] = useState([]);

  useEffect(() => {
    const fetchChats = async () => {
      try {
        const res = await axios.get("http://127.0.0.1:8000/api/get_chats/", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setChatHistory(res.data);
      } catch (error) {
        console.error("Error fetching chat history:", error);
      }
    };

    fetchChats();
  }, [token]);

  return (
    <div>
      <h2>Your Chat History</h2>
      {chatHistory.length === 0 ? (
        <p>No chat history found.</p>
      ) : (
        <ul>
          {chatHistory.map((chat, index) => (
            <li key={index} style={{ marginBottom: "1rem" }}>
              <strong>Prompt:</strong> {chat.prompt} <br />
              <strong>Response:</strong> {typeof chat.response === 'object' ? JSON.stringify(chat.response) : chat.response} <br />
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
