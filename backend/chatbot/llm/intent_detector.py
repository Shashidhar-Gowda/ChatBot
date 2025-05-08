import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

# Set up your LLM
chat_llm = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b",
    temperature=0
)

# Prompt for Intent Detection
intent_prompt = ChatPromptTemplate.from_template("""
You are an intent detection system.

Your task is to classify the user's question into one of the following intents and respond with only the intent name.

Intents:
- CORRELATION ANALYSIS
- LINEAR REGRESSION ANALYSIS
- CLASSIFICATION
- HISTORICAL DATA
- VISUALIZATION
- SUMMARIZATION
- REPORTING
- GREETINGS
- GENERAL QUESTIONS

User Question: {user_query}

Respond with ONLY the intent name, exactly as listed above.
Do not include any explanation or extra formatting.
""")

def detect_intent(user_query: str) -> str:
    try:
        # Step 1: Format prompt and invoke model
        prompt = intent_prompt.format(user_query=user_query)
        response = chat_llm.invoke([HumanMessage(content=prompt)])
        full_text = response.content.strip()

        # Step 2: Clean out <think> blocks and line breaks
        clean_text = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL).strip()
        clean_text = clean_text.replace("\n", "").strip()

        # Step 3: Return uppercase intent
        return clean_text.upper()
    except Exception as e:
        print(f"Intent detection failed: {e}")
        return "UNKNOWN"