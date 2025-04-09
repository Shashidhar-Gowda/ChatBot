# tools.py
from langchain.tools import Tool

def greet_tool_func(inputs: dict) -> str:
    return "Hi there! How can I help you today?"

def weather_tool_func(inputs: dict) -> str:
    location = inputs.get("location", "your area")
    return f"The weather in {location} is sunny with 25°C."

def calc_tool_func(inputs: dict) -> str:
    expr = inputs.get("expression", "0")
    try:
        result = eval(expr)
        return f"The result of {expr} is {result}"
    except:
        return "Invalid expression."

tool_map = {
    "greet": Tool(name="GreetTool", func=greet_tool_func, description="Responds to greetings"),
    "weather": Tool(name="WeatherTool", func=weather_tool_func, description="Fetches weather"),
    "calc": Tool(name="CalcTool", func=calc_tool_func, description="Performs calculations"),
}
