import { Routes, Route } from "react-router-dom";
import Login from "./components/Login";
import ChatBox from "./components/Home";
import AskFile from "./components/AskFile";
import Dashboard from "./components/DashBoard"; 
import PrivateRoute from "./components/PrivateRoute.jsx";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route
        path="/chat"
        element={
          <PrivateRoute>
            <ChatBox />
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
