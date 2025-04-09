// components/AskFile.js
import React, { useState } from "react";
import axios from "axios";

const AskFile = () => {
  const [file, setFile] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [answer, setAnswer] = useState("");

  const handleUpload = async () => {
    const token = localStorage.getItem("accessToken");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("prompt", prompt);

    try {
      const res = await axios.post("http://127.0.0.1:8001/api/ask-file", formData, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "multipart/form-data",
        },
      });

      setAnswer(res.data.answer);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div>
      <h2>Ask Questions on Your File</h2>
      <input type="file" accept=".pdf,.csv" onChange={(e) => setFile(e.target.files[0])} />
      <input type="text" placeholder="Ask something..." value={prompt} onChange={(e) => setPrompt(e.target.value)} />
      <button onClick={handleUpload}>Ask</button>

      {answer && <div className="answer-box">Bot: {answer}</div>}
    </div>
  );
};

export default AskFile;
