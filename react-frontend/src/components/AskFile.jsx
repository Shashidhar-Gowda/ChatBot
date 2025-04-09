import React, { useState } from 'react';
import './AskFile.css';

const AskFile = ({ onFileSubmit }) => {
  const [file, setFile] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const validTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/json'];
      if (validTypes.includes(selectedFile.type)) {
        setFile(selectedFile);
        setError('');
      } else {
        setError('Please upload a valid CSV, Excel, or JSON file');
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (!file) {
      setError('Please select a file');
      return;
    }
    if (!prompt.trim()) {
      setError('Please enter an analysis prompt');
      return;
    }

    setIsLoading(true);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('prompt', prompt);
      
      const response = await fetch('/api/upload', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: formData
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Upload failed');
      
      onFileSubmit(data.response);
    } catch (err) {
      setError(err.message || 'Failed to process file');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="file-upload-container">
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="file-upload">Choose File (CSV/Excel/JSON):</label>
          <input
            id="file-upload"
            type="file"
            accept=".csv,.xlsx,.xls,.json"
            onChange={handleFileChange}
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="analysis-prompt">Analysis Prompt:</label>
          <textarea
            id="analysis-prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="What analysis would you like to perform?"
            required
          />
        </div>
        
        {error && <div className="error-message">{error}</div>}
        
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Processing...' : 'Analyze File'}
        </button>
      </form>
    </div>
  );
};

export default AskFile;
