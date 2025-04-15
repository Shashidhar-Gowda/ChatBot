import { Routes, Route, Navigate } from "react-router-dom";
import Login from "./components/Login.jsx";
import ChatBox from "./components/Home.jsx";
import AskFile from "./components/AskFile.jsx";
import Dashboard from "./components/DashBoard.jsx"; 
import PrivateRoute from "./components/PrivateRoute.jsx";
import { useState } from "react";

function App() {
  const [messages, setMessages] = useState([]);

  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/" element={<Navigate to="/login" replace />} />
      <Route
        path="/chat"
        element={
          <PrivateRoute>
            <ChatBox messages={messages} />
          </PrivateRoute>
        }
      />
      <Route
        path="/ask-file"
        element={
          <PrivateRoute>
            <AskFile />
          </PrivateRoute>
        }
      />
      <Route
        path="/dashboard"
        element={
          <PrivateRoute>
            <Dashboard />
          </PrivateRoute>
        }
      />
    </Routes>
  );
}

export default App;
