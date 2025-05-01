from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load LLM
groq_model = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b",
    api_key="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",
    temperature=0.0,
)

# Prompt template for column matching
column_prompt = ChatPromptTemplate.from_template(
    """
You are an expert data assistant.  
Given a list of columns from a dataset: {columns}  
and a user query: {query}

Return only the names of the most relevant column(s) from the list, comma-separated.  
If no match, return "None".
Do not invent new columns.

Example: 
Columns: ['order_id', 'customer_name', 'total_spend']
Query: "Show me customer names and spending"
Answer: customer_name, total_spend
"""
)

# Create column matching chain
column_chain = column_prompt | groq_model | StrOutputParser()

def match_columns(columns_list, user_query):
    columns_as_text = ", ".join(columns_list)
    result = column_chain.invoke({"columns": columns_as_text, "query": user_query})
    matched_columns = [col.strip() for col in result.split(",") if col.strip()]
    return matched_columns