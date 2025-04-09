import React, { useState } from 'react';
import axios from 'axios';

const SignUp = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSignup = async (e) => {
    e.preventDefault();  // Prevent form submission from refreshing the page

    if (!email || !password) {
      setError('Please enter both email and password');
      return;
    }

    try {
      setLoading(true);  // Start loading while making the request

      const response = await axios.post(
        "http://127.0.0.1:8001/api/signup",
        {
          email,
          password,
        },
        {
          headers: {
            "Content-Type": "application/json",  // Ensure the content-type is set to JSON
          },
        }
      );

      setLoading(false); // Stop loading once we get a response
      alert("Signup successful! Please log in.");
      setEmail(""); // Clear form fields
      setPassword("");
      setError(""); // Clear any previous error
    } catch (error) {
      setLoading(false); // Stop loading if there's an error
      console.error("Signup Error:", error);  // Log the error for debugging

      const errorMessage =
        error.response?.data?.error || // Custom error like "User already exists"
        error.response?.data?.detail || // DRF default error messages
        "Signup failed. Please try again.";

    setError(errorMessage);

    }
  };

  return (
    <div className="signup-container">
      <h2>Sign Up</h2>
      
      {error && <div className="error-message">{error}</div>}  {/* Display error message */}

      <form onSubmit={handleSignup}> {/* Handle form submission */}
        <div>
          <label>Email</label>
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        <div>
          <label>Password</label>
          <input
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>
        <div>
          <button type="submit" disabled={loading}>  {/* Use type="submit" for form submission */}
            {loading ? 'Signing Up...' : 'Sign Up'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default SignUp;
