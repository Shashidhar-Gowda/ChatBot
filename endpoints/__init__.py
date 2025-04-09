# Makes endpoints a Python package
from .llm_chain import get_bot_response, reset_memory, llm
from .tools import AnalysisTools
from langchain.agents import Tool

# Expose tools through the AnalysisTools class
available_tools = [
    Tool(
        name="Data Analysis",
        func=AnalysisTools.describe_data,
        description="For performing basic descriptive statistics on data"
    ),
    Tool(
        name="Correlation Analysis",
        func=AnalysisTools.correlation_analysis,
        description="For calculating correlation between two columns in a dataset"
    ),
    Tool(
        name="Regression Analysis",
        func=AnalysisTools.linear_regression,
        description="For performing linear regression analysis on data"
    ),
    Tool(
        name="Classification",
        func=AnalysisTools.classify_data,
        description="For classifying data using simple artificial neural networks"
    )
]
