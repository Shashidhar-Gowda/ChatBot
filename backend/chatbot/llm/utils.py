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
    """
    Resolve the full file path for an uploaded file.

    This function constructs the path relative to the MEDIA_ROOT.
    It does NOT assume the current working directory.

    Args:
        filename (str): The name of the file (e.g., 'data.csv').

    Returns:
        str: The full, absolute path to the file.
    """
    # Construct the path relative to MEDIA_ROOT
    full_path = os.path.join(settings.MEDIA_ROOT, 'user_uploads', filename)

    # Ensure the path is absolute (more robust)
    full_path = os.path.abspath(full_path)  # Add this line?

    return full_path