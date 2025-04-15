import React from "react";
import { Navigate, useLocation } from "react-router-dom";

const PrivateRoute = ({ children }) => {
  const location = useLocation();
  // Check both localStorage and cookies for token
  const token = document.cookie.split('; ')
    .find(row => row.startsWith('token='))
    ?.split('=')[1] || localStorage.getItem("token");
  return token ? children : <Navigate to="/login" state={{ from: location, message: "Please login first" }} replace />;
};

export default PrivateRoute;
