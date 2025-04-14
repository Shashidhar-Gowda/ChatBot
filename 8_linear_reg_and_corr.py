import os
import json
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# === Load model ===
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

# === Load cleaned dataset ===
df = pd.read_csv("cleaned_dataset.csv")
print("Loaded cleaned_dataset.csv")

# === Tool 1: Correlation Analysis ===
def correlation_tool_func(query: str) -> str:
    method = "pearson"
    if "spearman" in query.lower():
        method = "spearman"
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr(method=method)
    return f"{method.title()} correlation matrix:\n\n{corr.to_string()}"

correlation_tool = Tool(
    name="CorrelationAnalysis",
    func=correlation_tool_func,
    description="Use this to compute Pearson or Spearman correlation matrix on numerical columns of the dataset."
)

# === Tool 2: Linear Regression ===
def regression_tool_func(query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a regression analysis expert."),
        ("human", f"Dataset columns: {list(df.columns)}\n\nQuery: {query}\n\nWhich columns should be used as features, and which as target? Respond in JSON like: {{'features': [...], 'target': '...'}}")
    ])
    chain = prompt | llm
    result = chain.invoke({"query": query})

    try:
        col_selection = json.loads(result.content)
        features = col_selection["features"]
        target = col_selection["target"]

        X = df[features]
        y = df[target]
        model = LinearRegression()
        model.fit(X, y)

        score = model.score(X, y)
        coeffs = dict(zip(features, model.coef_))

        return f"""Linear Regression:
Target: {target}
Features: {features}
R²: {score:.4f}
Coefficients: {coeffs}
Intercept: {model.intercept_:.4f}"""

    except Exception as e:
        return f"Error running regression: {str(e)}"

regression_tool = Tool(
    name="LinearRegression",
    func=regression_tool_func,
    description="Use this to run linear regression and predict one column based on others."
)

# === Agent Tools and System Prompt Fix ===
tools = [correlation_tool, regression_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "system_message": (
            "You are a data science assistant. "
            "Your goal is to decide which tool to use to answer the user's query. "
            "Use ONLY the tools provided. "
            "Follow this exact format:\n\n"
            "Thought: your reasoning\n"
            "Action: name of the tool\n"
            "Action Input: input to the tool\n\n"
            "When done:\n"
            "Final Answer: your final response to the user"
        )
    }
)

# === Chat Loop ===
print("\nAsk a question about correlation or regression. Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    try:
        response = agent.run(query)
        print(f"\nNuclei: {response}\n")
    except Exception as e:
        print(f"Agent error: {e}")
