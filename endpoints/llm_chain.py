import os
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 1. Initialize Groq model (API key must already be set in environment)
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    api_key="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"
    )

# 2. Memory + Conversation chain setup
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# 3. System greeting (printed once when app starts)
system_message = SystemMessage(
    content="Hi! Welcome to the world of data analysis and reporting. I am BrainBot who will guide you through this entire process."
)
print(f"🧠 BrainBot: {system_message.content}")

# 4. Response function
def get_bot_response(user_input: str):
    if not memory.chat_memory.messages:  # If first message
        memory.chat_memory.add_message(SystemMessage(
            content="Hi! Welcome to the world of data analysis and reporting. I am BrainBot who will guide you through this entire process."
        ))
    return conversation.predict(input=user_input)

# 5. Reset memory
def reset_memory():
    global memory, conversation
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory)
