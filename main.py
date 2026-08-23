import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from Tools.Retrieval_Layer_Tools.utils import serp_search
from AgentStates.agentic_rag_states import AgentState
from Nodes.agentic_rag_nodes import agent, grade_documents, generate,rewrite,route_after_agent
from langchain_core.messages import ToolMessage
from typing import Literal

load_dotenv()

class AgenticRAGReActPipeline:
    
    def __init__(self,temperature: float = 0.2, debug:bool = False):
        self.llm = ChatGoogleGenerativeAI(temperature=temperature,
                                          api_key=os.getenv("GOOGLE_GENAI_API_KEY"),
                                          model="gemini-3.6-flash")
        self.debug = debug
        self.tools = [serp_search]
        self.graph = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", lambda state: agent(self,state))
        workflow.add_node("retrieve",ToolNode(self.tools))
        workflow.add_node("rewrite",lambda state: rewrite(self,state))
        workflow.add_node("generate",lambda state: generate(self,state))
        
        def _restricted(_: AgentState) -> Dict[str,Any]:
            msg = (
                "This app only supports web search."
                "Your request appears outside this scope. Please use one of the supported capabilities."
            )
            return {"messages": AIMessage(content=msg)}
        
        workflow.add_node("restricted",_restricted)
        workflow.add_edge(START,"agent")
        workflow.add_conditional_edges(
            source="agent",
            path=route_after_agent,
            path_map={"tools": "retrieve", "generate": "generate", "restricted": "restricted"}
        )
        workflow.add_conditional_edges("retrieve",lambda state: grade_documents(self,state)) # The return of grade_dcoument function is either generate or rewrite as a string.
        workflow.add_edge("generate",END)
        workflow.add_edge("restricted",END)
        workflow.add_edge("rewrite","agent")
        
        return workflow.compile()
    
    
    def answer(self,question: str) -> str:
        result = self.graph.invoke({"messages":[HumanMessage(content=question)]})
        
        msgs = result.get("messages",[])
        if not msgs:
            return ""
        last = msgs[-1]
        try:
            return getattr(last,"content",str(last))
        except Exception:
            return str(last)
        
if __name__ == "__main__":
    graph = AgenticRAGReActPipeline(debug=True)
    
    question = input("What is your question?:\n")
    
    answer = graph.answer(question)
    print("==========================================")
    print("\n\nAnswer: ",answer)
    print("==========================================")   
        
        