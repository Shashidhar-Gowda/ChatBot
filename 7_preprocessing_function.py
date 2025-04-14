import os
import pandas as pd
import re
import io
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load .env
load_dotenv()
model = ChatGroq(model="deepseek-r1-distill-llama-70b")

safe_globals = {
    "pd": pd,
    "__builtins__": __builtins__,  # Optional - restrict built-ins if needed
}

# Load dataset
df = pd.read_csv("/Users/janhavi.kumbhar/Desktop/nuclei/data/GoDaddy_MMM.csv")  # Change to your CSV filename
print("Dataset loaded.")

# Prepare summary
summary = df.describe(include="all").to_string()
info_buf = io.StringIO()
df.info(buf=info_buf)
info_str = info_buf.getvalue()

# Prompt template
preprocess_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data cleaning expert. Based on the dataset summary and info, generate a pandas-only cleaning script. Do NOT use sklearn or any external libraries."),
    ("human", "Dataset summary:\n{summary}\n\nDataset info:\n{info}\n\nReturn ONLY valid Python code (no markdown or explanations) that cleans the dataframe named 'df'.")
])


# Run prompt
chain = preprocess_prompt | model
response = chain.invoke({
    "summary": summary,
    "info": info_str
})
code = response.content.strip()

# Extract and run the code
code_block = re.findall(r"```(?:python)?(.*?)```", code, re.DOTALL)
final_code = code_block[0] if code_block else code

print("\nCleaning code from LLM:\n")
print(final_code)

# Execute it
local_vars = {"df": df}
exec(final_code, {"pd": pd}, local_vars)
exec(final_code, safe_globals, local_vars)
cleaned_df = local_vars["df"]


# Save cleaned dataset
cleaned_df.to_csv("cleaned_dataset.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_dataset.csv'")
