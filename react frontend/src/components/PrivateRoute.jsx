// src/components/PrivateRoute.jsx
import React from "react";
import { Navigate } from "react-router-dom";

const PrivateRoute = ({ children }) => {
  // Check both localStorage and cookies for token
  const token = document.cookie.split('; ')
    .find(row => row.startsWith('token='))
    ?.split('=')[1] || localStorage.getItem("token");
  return token ? children : <Navigate to="/" />;
};

export default PrivateRoute;
