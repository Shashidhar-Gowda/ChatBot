import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const SignUp = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate(); // Use navigate to redirect

  const handleSignup = async (e) => {
    e.preventDefault();
  
    if (!email || !password) {
      setError('Please enter both email and password');
      return;
    }
  
    try {
      setLoading(true);
  
      // 1. Clear any previous tokens
      localStorage.removeItem("token");
      localStorage.removeItem("refresh");
      document.cookie = "token=; path=/; max-age=0";
  
      // 2. Signup
      await axios.post(
        "http://127.0.0.1:8000/api/signup/",
        { email, password },
        { headers: { "Content-Type": "application/json" } }
      );
  
      // 3. Login right after signup
      const loginResponse = await axios.post("http://127.0.0.1:8000/api/login/", {
        email,
        password,
      });
  
      const { access, refresh } = loginResponse.data;
  
      // 4. Store new user token
      localStorage.setItem("token", access);
      localStorage.setItem("refresh", refresh);
      document.cookie = `token=${access}; path=/; max-age=86400`;
  
      // 5. Navigate
      navigate("/chat");
  
      setLoading(false);
      setEmail("");
      setPassword("");
      setError("");
    } catch (error) {
      setLoading(false);
      const errorMessage =
        error.response?.data?.error ||
        error.response?.data?.detail ||
        "Signup failed. Please try again.";
      setError(errorMessage);
    }
  };
  
  

  return (
    <div className="signup-container">
      <h2>Sign Up</h2>
      
      {error && <div className="error-message">{error}</div>}

      <form onSubmit={handleSignup}>
        <div className="form-group">
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="form-input"
          />
        </div>
        <div className="form-group">
          <input
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="form-input"
          />
        </div>
        <div className="login-center-buttons">
          <button 
            type="submit" 
            className="btn-primary"
            disabled={loading}
          >
            {loading ? 'Signing Up...' : 'Sign Up'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default SignUp;
