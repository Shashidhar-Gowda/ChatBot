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

Your task is to classify the user's question into one of the following intents and Respond with ONLY the intent name, exactly as listed above. Do not include any explanation or extra formatting..

Intents:
- CORRELATION ANALYSIS
- REGRESSION ANALYSIS
- CLASSIFICATION
- HISTORICAL DATA
- VISUALIZATION
- SUMMARIZATION
- REPORTING
- GREETINGS
- GENERAL QUESTIONS
- PREDICTION

User Question: {user_query}

Respond with ONLY the intent name, exactly as listed above.
Do not include any explanation or extra formatting.
""")

def detect_intent(user_query: str) -> str:
    try:
        # Basic keyword fallback before LLM call
        keyword_intents = {
            r'\b(hi|hello|hey)\b': "GREETINGS",
            r'\b(correlat|relation|pearson|spearman)\b': "CORRELATION ANALYSIS",
            r'\b(regress|predict|forecast)\b': "REGRESSION ANALYSIS",
            r'\b(classify|category|neural network|ann)\b': "CLASSIFICATION",
            r'\b(visualize|plot|chart|graph)\b': "VISUALIZATION",
            r'\b(summary|overview)\b': "SUMMARIZATION"
        }
        
        for pattern, intent in keyword_intents.items():
            if re.search(pattern, user_query, re.IGNORECASE):
                return intent
                
        # Only call LLM if keyword matching fails
        prompt = intent_prompt.format(user_query=user_query)
        response = chat_llm([HumanMessage(content=prompt)])
        clean_text = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL)
        clean_text = clean_text.replace("\n", "").strip().upper()
        
        # Validate the intent is in our list
        valid_intents = [
            "CORRELATION ANALYSIS", "REGRESSION ANALYSIS", "CLASSIFICATION",
            "HISTORICAL DATA", "VISUALIZATION", "SUMMARIZATION", "REPORTING",
            "GREETINGS", "GENERAL QUESTIONS", "PREDICTION"
        ]
        
        return clean_text if clean_text in valid_intents else "UNKNOWN"
        
    except Exception as e:
        print(f"Intent detection failed: {e}")
        return "UNKNOWN"