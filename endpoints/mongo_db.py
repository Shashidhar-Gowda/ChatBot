"""In-memory data storage replacement for MongoDB functionality"""
from typing import Optional, Dict, Any

# In-memory storage dictionaries
user_memories: Dict[str, Dict[str, Any]] = {}
conversation_history: Dict[str, list] = {}

def get_user_memory(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user memory from in-memory storage"""
    return user_memories.get(user_id)

def remember_name(user_id: str, name: str) -> None:
    """Store user name in memory"""
    if user_id not in user_memories:
        user_memories[user_id] = {}
    user_memories[user_id]['name'] = name

def recall_name(user_id: str) -> Optional[str]:
    """Recall user name from memory"""
    user_data = user_memories.get(user_id)
    return user_data.get('name') if user_data else None

def save_conversation(user_id: str, message: str, response: str) -> None:
    """Save conversation history in memory"""
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id].append({
        'message': message,
        'response': response,
        'timestamp': datetime.now().isoformat()
    })

def get_conversation_history(user_id: str) -> list:
    """Get conversation history from memory"""
    return conversation_history.get(user_id, [])
