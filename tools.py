# tools.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Prompt-driven preprocessing
def get_preprocessing_tool(llm):
    def tool(df):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a data preprocessing assistant. Decide what preprocessing is needed for the columns: missing values, encoding, dropping, etc. Return a summary of operations."),
            ("human", "{columns}")
        ])

        chain = RunnableWithMessageHistory(
            prompt | llm,
            lambda session_id: InMemoryChatMessageHistory(),
            input_messages_key="columns",
            history_messages_key="history"
        )

        result = chain.invoke({"columns": ', '.join(df.columns)}, config={"configurable": {"session_id": "preprocessing"}}).content
        print("[Preprocessing LLM Plan]:", result)
        return df.dropna()
    return tool

# Prompt-driven correlation tool
def get_correlation_tool(llm):
    def tool(df):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a correlation analysis expert. Choose columns for correlation from: {columns}. Return pairs you would analyze."),
            ("human", "Pick top correlated columns.")
        ])

        chain = RunnableWithMessageHistory(
            prompt | llm,
            lambda session_id: InMemoryChatMessageHistory(),
            input_messages_key="columns",
            history_messages_key="history"
        )

        pairs = chain.invoke({"columns": ', '.join(df.select_dtypes(include='number').columns)}, config={"configurable": {"session_id": "correlation"}}).content
        print("[Correlation Pairs LLM Suggestion]:", pairs)

        corr = df.select_dtypes(include='number').corr()
        return str(corr)
    return tool

# Prompt-driven regression tool
def get_regression_tool(llm):
    def tool(df):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a regression assistant. From columns: {columns}, select a target and appropriate features for linear regression. Respond with a JSON or string plan."),
            ("human", "Select target and features")
        ])

        chain = RunnableWithMessageHistory(
            prompt | llm,
            lambda session_id: InMemoryChatMessageHistory(),
            input_messages_key="columns",
            history_messages_key="history"
        )

        columns = df.select_dtypes(include='number').columns
        result = chain.invoke({"columns": ', '.join(columns)}, config={"configurable": {"session_id": "regression"}}).content
        print("[Regression Plan]:", result)

        # Fallback example logic: use last column as target, others as features
        target = columns[-1]
        features = columns[:-1]

        X = df[features]
        y = df[target]

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)

        return f"Model trained using features: {features.tolist()} -> Target: {target}\nMSE: {mse:.2f}, R²: {r2:.2f}"
    return tool

# Prompt-driven visualization tool
def get_visualization_tool(llm):
    def tool(df):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a visualization assistant. Based on the dataset columns: {columns}, suggest a chart to plot. Return column names for x and y."),
            ("human", "Which columns to plot?")
        ])

        chain = RunnableWithMessageHistory(
            prompt | llm,
            lambda session_id: InMemoryChatMessageHistory(),
            input_messages_key="columns",
            history_messages_key="history"
        )

        result = chain.invoke({"columns": ', '.join(df.columns)}, config={"configurable": {"session_id": "visualization"}}).content
        print("[Visualization LLM Suggestion]:", result)

        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) >= 2:
            x, y = numeric_cols[:2]
            plt.figure()
            sns.scatterplot(data=df, x=x, y=y)
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            return image_base64
        return "Not enough numeric columns to plot."
    return tool

# Prompt-driven summarization tool
def get_summarization_tool():
    def tool(df, llm):
        summary = df.describe().to_string() + "\n\n" + df.head().to_string()

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a data summarization expert. Summarize this data for a human-readable report."),
            ("human", "{summary}")
        ])

        chain = prompt | llm
        result = chain.invoke({"summary": summary})
        return result.content

    return tool
