from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load model
model = ChatGroq(model="deepseek-r1-distill-llama-70b")

# Define the prompt template for intent detection
intent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent assistant that detects the intent behind user queries in a data analysis context. Only return the intent name, nothing else."),
    ("human", "{query}")
])

def detect_intent(query):
    chain = intent_prompt | model
    result = chain.invoke({"query": query})
    return result.content.strip().lower()