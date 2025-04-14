from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGroq(model="deepseek-r1-distill-llama-70b")

chat_history = [] #use to store messages


intent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent assistant that detects the intent behind user queries in a data analysis context. Only return the intent name, nothing else."),
    ("human", "{query}")
])

def detect_intent(query):
    chain = intent_prompt | model
    result = chain.invoke({"query": query})
    return result.content.strip().lower()

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    # Detect intent
    intent = detect_intent(query)
    print(f"[Detected Intent]: {intent}")

    # Continue conversation
    chat_history.append(HumanMessage(content=query))
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"Nuclei: {response.content}")


