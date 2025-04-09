from .tools import tool_map

def route_to_tool(intent: str, inputs: dict) -> str:
    """
    Routes the request to the appropriate tool based on intent.
    """
    tool = tool_map.get(intent)

    if tool:
        return tool.func(inputs)
    else:
        return "I'm not sure how to handle that request yet."
