from langchain_openai import ChatOpenAI  # You might replace this with your preferred LLM
from langchain_experimental.sql import SQLDatabaseChain  # new import
from langchain.tools import Tool
from .utils import get_sql_database  # Import your database connection function

def create_db_aware_agent(llm):
    """Creates an SQL agent connected to the PostgreSQL database."""
    db = get_sql_database()
    if db is None:
        return None  # Or raise an exception, depending on your error handling strategy

    db_chain = SQLDatabaseChain.from_llm(
        llm=llm,
        db=db,
        verbose=True,
        handle_parsing_errors=True,
    )
    return db_chain

sql_tool = Tool(
    name="sql_db",
    func=create_db_aware_agent,
    description="Useful for interacting with a SQL database."
)
