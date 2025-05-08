from langchain_community.utilities import SQLDatabase
from django.conf import settings 

def get_sql_database():
    """Connects to the PostgreSQL database using Django settings."""
    db_config = settings.DATABASES['default']  # Safely get db config
    db_url = f"postgresql://{db_config['root']}:{db_config['root']}@{db_config['localhost']}:{db_config['5432']}/{db_config['chatbot']}"
    try:
        db = SQLDatabase.from_uri(db_url)
        return db
    except Exception as e:
        print(f"Database connection error: {e}")
        return None  # Handle connection errors gracefully
    

import os
from django.conf import settings

def resolve_file_path(filename: str) -> str:
    media_root = settings.MEDIA_ROOT
    full_path = os.path.join(media_root, 'user_uploads', filename)
    print(f"[resolve_file_path] MEDIA_ROOT={media_root}, Full path: {full_path}")
    return full_path
