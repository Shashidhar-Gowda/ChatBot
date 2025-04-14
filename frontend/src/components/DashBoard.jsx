import React, { useEffect } from "react";
import axios from "axios";

const Dashboard = ({ setMessages }) => {
  const token = localStorage.getItem("token");

  useEffect(() => {
    const fetchChats = async () => {
      try {
        const res = await axios.get("http://127.0.0.1:8000/api/get_chats/", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setMessages(res.data); // Pass the fetched chats to the Home component
      } catch (error) {
        console.error("Error fetching chat history:", error);
      }
    };

    fetchChats();
  }, [setMessages, token]);

  return (
    <div>
      <h2>Your Chat History</h2>
      {/* Optionally render chat history here if needed */}
    </div>
  );
};

export default Dashboard;
