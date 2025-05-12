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
    More robust file path resolution with validation
    """
    try:
        if not filename:
            raise ValueError("Filename cannot be empty")
            
        # Normalize path and prevent directory traversal
        filename = os.path.normpath(filename).lstrip('/')
        full_path = os.path.join(settings.MEDIA_ROOT, 'user_uploads', filename)
        full_path = os.path.abspath(full_path)
        
        # Security check - ensure path is within MEDIA_ROOT
        if not full_path.startswith(os.path.abspath(settings.MEDIA_ROOT)):
            raise ValueError("Invalid file path - outside of allowed directory")
            
        # Check file exists
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found at path: {full_path}")
            
        return full_path
    except Exception as e:
        print(f"File path resolution error: {e}")
        raise  # Re-raise for calling code to handle

# In utils.py, add:
def parse_tool_input(input_str: str, context: dict) -> str:
    """
    Normalize tool input parameters based on context.
    Returns properly formatted input string for tools.
    """
    # If it's already in param=value format, return as-is
    if '=' in input_str:
        return input_str
        
    # Special cases for file-based tools
    file_tools = [
        'get_report_generator_tool',
        'get_simple_summary_tool',
        'linear_regression_tool',
        'polynomial_regression_tool',
        # Add other file-based tools...
    ]
    
    current_tool = context.get('current_tool')
    if current_tool in file_tools and context.get('file_id'):
        return f"file_path={context['file_id']}"
        
    return input_str