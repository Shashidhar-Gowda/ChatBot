from typing import Dict, Any
import json
import re
from langchain.agents import Tool
from langchain.schema import AgentAction

class RuleBasedAgent:
    def __init__(self, tools: Dict[str, Tool], intents_file: str = "endpoints/intents.json"):
        self.tools = tools
        with open(intents_file) as f:
            self.intents = json.load(f)["intents"]
        
        # Agent configuration
        self.max_retries = 5
        self.timeout_sec = 30
        self.early_stopping = True
        
    def _detect_intent(self, user_input: str) -> Dict[str, Any]:
        """Detect intent using pattern matching from intents.json"""
        if not user_input or len(user_input.strip()) < 2:
            return None
            
        user_input = user_input.lower()
        for intent in self.intents:
            for pattern in intent["patterns"]:
                if re.search(r'\b' + re.escape(pattern.lower()) + r'\b', user_input):
                    return intent
        return None
        
    def determine_action(self, user_input: str, retry_count: int = 0) -> AgentAction:
        """Determine the appropriate action with enhanced loop control"""
        # Check for max retries
        if retry_count >= self.max_retries:
            return AgentAction(
                tool="Final Answer",
                tool_input={"output": "Maximum retries reached. Please try a different query."},
                log=f"Max retries ({self.max_retries}) reached"
            )
            
        # Check for timeout
        if retry_count * 2 > self.timeout_sec:
            return AgentAction(
                tool="Final Answer",
                tool_input={"output": "Analysis timed out. Please try a simpler query."},
                log=f"Timeout after {retry_count} retries"
            )
            
        intent = self._detect_intent(user_input)
        
        # Fallback to LLM for:
        # 1. No detected intent
        # 2. Questions not about data/analysis
        # 3. Complex natural language queries
        if (not intent or 
            not any(keyword in user_input.lower() for keyword in ['data', 'analyze', 'stat', 'model', 'predict']) or
            len(user_input.split()) > 15):  # Long/complex questions
            return AgentAction(
                tool="llm_chain", 
                tool_input={
                    "query": user_input,
                    "context": "User asked a general knowledge question"
                },
                log="Forwarding to LLM for general Q&A"
            )
            
        if "tool" not in intent:
            return AgentAction(
                tool="Final Answer",
                tool_input={"output": intent["responses"][0]},
                log="Basic response"
            )
            
        # Initialize tool input with enhanced data handling
        tool_input = {
            "data": user_input,
            "timeout_sec": 10 - retry_count * 2,
            "data_type_check": "auto",  # Auto-detect numeric vs categorical
            "fallback_strategy": "describe"  # Fallback to descriptive stats if analysis fails
        }
        
        # Enhanced handling for different analysis types
        if intent["tag"] in ["correlation_analysis", "regression_analysis"]:
            columns = re.findall(r'\b([A-Za-z_]+)\b', user_input)
            if len(columns) >= 2:
                tool_input.update({
                    "x_col": columns[0],
                    "y_col": columns[1],
                    "force_numeric": False,  # Allow categorical data
                    "fallback_strategy": "crosstab"  # Use contingency tables for categorical
                })
        elif intent["tag"] == "price_comparison":
            tool_input.update({
                "timeout_sec": 15,
                "group_by": "area",  # Explicit grouping for price comparison
                "agg_func": ["mean", "max", "min"]  # Multiple aggregation functions
            })
        elif intent["tag"] == "descriptive_stats":
            tool_input["stats_for"] = "all"  # Get stats for all columns
        
        return AgentAction(
            tool=intent["tool"],
            tool_input=tool_input,
            log=f"Running {intent['tag']} analysis (attempt {retry_count + 1})",
            handle_tool_error=True
        )
