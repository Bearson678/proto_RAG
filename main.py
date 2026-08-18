import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

class AgenticRAGReActPipeline:
    
    def __init__(self,temperature: float = 0.2, debug:bool = False):
        self.llm = ChatGoogleGenerativeAI(temperature=temperature,api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
        self.debug = debug
        
        pass