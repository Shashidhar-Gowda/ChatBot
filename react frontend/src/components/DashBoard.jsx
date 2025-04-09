import React, { useEffect, useState } from "react";
import axios from "axios";

const Dashboard = () => {
  const [chats, setChats] = useState([]);
  const token = localStorage.getItem("token");

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

  return (
    <div>
      <h2>Your Chat History</h2>
      {chats.map((chat, index) => (
        <div key={index} style={{ padding: "1rem", borderBottom: "1px solid #ccc" }}>
          <p><strong>You:</strong> {chat.prompt}</p>
          <p><strong>Bot:</strong> {chat.response}</p>
          <p style={{ fontSize: "0.8rem", color: "gray" }}>{new Date(chat.timestamp).toLocaleString()}</p>
        </div>
      ))}
    </div>
  );
};

export default Dashboard;
