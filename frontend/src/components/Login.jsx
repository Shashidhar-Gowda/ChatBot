import React, { useState } from "react";
import axios from "axios";
import Image from "../assets/image.png";
import GoogleSvg from "../assets/icons8-google.svg";
import { FaEye } from "react-icons/fa6";
import { FaEyeSlash } from "react-icons/fa6";
import SignUp from "./SignUp";
import { useNavigate } from "react-router-dom";

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isSignUp, setIsSignUp] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post("http://127.0.0.1:8000/api/login/", {
        email,
        password,
      });
  
      const { access, refresh } = res.data;
      localStorage.setItem("token", access);
      document.cookie = `token=${access}; path=/; max-age=86400`;
      
      navigate("/chat");
    } catch (error) {
      alert("Invalid credentials");
    }
  };

  return (
    <div className="login-main">
      <div className="login-left">
        <img src={Image} alt="" />
      </div>
      <div className="login-right">
        <div className="login-right-container">
          <div className="login-logo">
            <img src={Image} alt="App Logo" />
          </div>
          <div className="login-center">
            <h2>{isSignUp ? "Create an Account" : "Welcome back!"}</h2>
            <p>{isSignUp ? "Please fill in your details" : "Please enter your details"}</p>

            {isSignUp ? (
              <SignUp />
            ) : (
              <form onSubmit={handleLogin}>
                <input
                  type="email"
                  placeholder="Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />

                <div className="pass-input-div">
                  <input
                    type={showPassword ? "text" : "password"}
                    placeholder="Password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                  />
                  {showPassword ? (
                    <FaEyeSlash onClick={() => setShowPassword(!showPassword)} />
                  ) : (
                    <FaEye onClick={() => setShowPassword(!showPassword)} />
                  )}
                </div>

                <div className="login-center-options">
                  <div className="remember-div">
                    <input type="checkbox" id="remember-checkbox" />
                    <label htmlFor="remember-checkbox">
                      Remember for 30 days
                    </label>
                  </div>
                  <a href="#" className="forgot-pass-link">
                    Forgot password?
                  </a>
                </div>

                <div className="login-center-buttons">
                  <button type="submit" className="btn-primary">
                    Log In
                  </button>
                  <button type="button" className="btn-google">
                    <img src={GoogleSvg} alt="" />
                    Log In with Google
                  </button>
                </div>
              </form>
            )}
          </div>

          <p className="login-bottom-p">
            {isSignUp ? (
              <>
                Already have an account? <a href="#" onClick={() => setIsSignUp(false)}>Log In</a>
              </>
            ) : (
              <>
                Don't have an account? <a href="#" onClick={() => setIsSignUp(true)}>Sign Up</a>
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
