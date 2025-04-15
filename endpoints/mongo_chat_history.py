import os
from pymongo import MongoClient
from typing import List, Dict, Optional
from datetime import datetime

# MongoDB connection URI - set this in your environment variables for security
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

# Database and collection names
DB_NAME = "chat_history_db"
COLLECTION_NAME = "user_chats"

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

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

def get_chat_history_by_user(user_id: str, limit: int = 50) -> List[Dict]:
    """
    Retrieve chat history for a user, sorted by timestamp descending.
    """
    cursor = collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
    return list(cursor)

def get_chat_history_by_session(session_id: str, limit: int = 50) -> List[Dict]:
    """
    Retrieve chat history for a session, sorted by timestamp descending.
    """
    cursor = collection.find({"session_id": session_id}).sort("timestamp", -1).limit(limit)
    return list(cursor)

def clear_chat_history_by_session(session_id: str) -> None:
    """
    Delete chat history documents for a session.
    """
    collection.delete_many({"session_id": session_id})

def clear_chat_history_by_user(user_id: str) -> None:
    """
    Delete chat history documents for a user.
    """
    collection.delete_many({"user_id": user_id})

# Example usage:
# save_chat_history("user123", "Hello", "Hi there!", session_id="sess456")
# chats = get_chat_history_by_user("user123")
