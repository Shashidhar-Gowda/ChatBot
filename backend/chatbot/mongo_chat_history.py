import os
from pymongo import MongoClient
from typing import List, Dict, Optional
from datetime import datetime

import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")  # fallback if not set
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

from datetime import datetime, timedelta
from collections import defaultdict

def get_grouped_chat_history(user_id: str, limit_per_group=50) -> Dict[str, List[Dict]]:
    now = datetime.utcnow()
    today = now.date()
    yesterday = today - timedelta(days=1)
    last_7_days = today - timedelta(days=7)

    raw_chats = collection.find({"user_id": user_id}).sort("timestamp", -1)
    grouped = defaultdict(list)

    for chat in raw_chats:
        timestamp = chat["timestamp"]
        chat_date = timestamp.date()
        chat_item = {
            "prompt": chat.get("prompt", ""),
            "response": chat.get("response", ""),
            "timestamp": timestamp.isoformat()
        }

        if chat_date == today:
            group = "Today"
        elif chat_date == yesterday:
            group = "Yesterday"
        elif chat_date >= last_7_days:
            group = "Last 7 Days"
        else:
            group = chat_date.strftime("%Y-%m-%d")

        if len(grouped[group]) < limit_per_group:
            grouped[group].append(chat_item)

    return dict(grouped)

