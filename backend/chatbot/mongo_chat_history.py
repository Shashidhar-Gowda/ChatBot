import os
from pymongo import MongoClient
from typing import List, Dict, Optional
from datetime import datetime

MONGO_URI = "mongodb://localhost:27017/chatbot_db"
client = MongoClient(MONGO_URI)
db = client["chatbot_db"]
collection = db["chat_history"]

def save_chat_history(user_id: str, prompt: str, response: str, session_id: Optional[str] = None) -> None:
    chat_doc = {
        "user_id": user_id,
        "session_id": session_id,
        "prompt": prompt,
        "response": response,
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(chat_doc)

def get_chat_history_by_session(user_id: str, session_id: str) -> List[Dict]:
    return list(collection.find({
        "user_id": user_id,
        "session_id": session_id
    }).sort("timestamp", 1))

def get_user_sessions(user_id: str) -> List[Dict]:
    # Return list of distinct session IDs
    sessions = collection.aggregate([
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": "$session_id", "latest_message": {"$max": "$timestamp"}}},
        {"$sort": {"latest_message": -1}}
    ])
    return list(sessions)
