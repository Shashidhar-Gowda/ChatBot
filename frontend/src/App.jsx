import { Routes, Route } from "react-router-dom";
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
      <Route path="/" element={<Login />} />
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
            <Dashboard setMessages={setMessages} />
          </PrivateRoute>
        }
      />
    </Routes>
  );
}

export default App;
