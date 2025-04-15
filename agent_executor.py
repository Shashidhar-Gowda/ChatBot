from tools import get_preprocessing_tool, get_correlation_tool, get_visualization_tool, get_regression_tool

class AgentExecutorHandler:
    def invoke_agent(self, intent, df):
        intent = intent.lower()
        if intent == "preprocessing":
            tool = get_preprocessing_tool()
            return tool(df).to_string()
        elif intent == "correlation analysis":
            tool = get_correlation_tool()
            return tool(df)
        elif intent == "linear regression":
            target_column = input("Enter the target column for regression: ")
            tool = get_regression_tool()
            return tool(df, target_column)
        elif intent == "visualization":
            tool = get_visualization_tool()
            return tool(df)[:100] + "... [truncated base64 image]"
        else:
            return f"Unknown intent: {intent}"
