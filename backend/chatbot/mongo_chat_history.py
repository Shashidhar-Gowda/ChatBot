import os
from pymongo import MongoClient
from typing import List, Dict, Optional
from datetime import datetime

# MongoDB connection URI - set this in your environment variables for security
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017/chatbot_db")
client = MongoClient(MONGO_URI)
db = client["chatbot_db"]
collection = db["chat_history"]

def save_chat_history(user_id: str, prompt: str, response: str, session_id: Optional[str] = None) -> None:
    """
    Save chat history document to MongoDB.
    If session_id is provided, store chat under that session, else user-based.
    """
    chat_doc = {
        "user_id": user_id,
        "session_id": session_id,
        "prompt": prompt,
        "response": response,
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(chat_doc)
from pymongo import MongoClient



def get_chat_history_by_session(username, session_id):
    return list(collection.find({
        "username": username,
        "session_id": session_id
    }).sort("timestamp", 1))
