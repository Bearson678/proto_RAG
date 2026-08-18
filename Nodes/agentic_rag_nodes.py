from typing import Any, Dict, Literal
from AgentStates.agentic_rag_states import AgentState, RelevanceGrade
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_core.output_parsers import StrOutputParser

def agent(self, state: AgentState) -> Dict[str,Any]:
    """Decide whether to use the tool"""
    
    messages = state["messages"]
    
    sys = SystemMessage(content=(
        "You are a researcher who is tasked to provide recommendations for different products.\n"
        "You are to advice product owners what are the standard procedures and methods to meet their product requirements.\n"
        "You are restricted to these capabilities only :\n" 
        "1) web_search.\n"
        "- Always use one of these tools to act.\n"
        "- For factual lookup, call web_search.\n"
        "- Ensure that the searched advice is compliant with FDA regulations and ISO standards."
    ))
    
    model = self.llm.bind_tools(self.tools)
    response = model.invoke([sys,*messages])
    
    if getattr(self,"debug",False):
        try:
            print("[agent debug] tool_calls:", getattr(response,"tool_calls",None))
            print("[agent debug] content sample:",str(getattr(response,"content","")))
        except Exception:
            pass
    return {"messages":[response]}


def grade_documents(self,state:AgentState) -> Literal["generate","rewrite"]:
    """Check if retrieved docs are relevant to the question using Pydantic-validated output."""
    
    grader = self.llm.with_structed_output(RelevanceGrade)
    prompt = PromptTemplate(
        template=(
            "You are a grader assessing relevance of a retrieved document to a user question.\n"
            "Here is the retrieved document: \n\n{context}\n\n"
            "Here is the user question: {question}\n"
            "If the document contains keyword(s) or semantic meaning related to the user question,"
            "grade it as relevant. Give a binary score 'yes' or 'no' ."
        ),
        input_variables=["context","question"]
    )
    
    print(f"_grade documents prompt: {prompt}")
    
    chain = prompt | grader
    
    messages = state["messages"]
    question = messages[0].content if messages else ""
    last_message = messages[-1] if messages else None
    docs_content = getattr(last_message,"content","") if last_message else ""
    
    scored = chain.invoke({"question":question,"context":docs_content})
    score = (scored.binary_Score or "").strip().lower()
    print(f"_grade documents score: {score}")
    return "generate" if score == "yes" else "rewrite"


def generate(self, state: AgentState) -> Dict[str, Any]:
    """RAG answer generation from docs and question."""
    messages = state["messages"]
    question = messages[0].content if messages else ""
    docs_content = getattr(messages[-1], "content", "") if messages else ""

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Use the provided context to answer the question.\n"
            "Be concise and cite sources with links when available.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        ),
        input_variables=["context", "question"],
    )
    
    chain = prompt | self.llm | StrOutputParser()
    response = chain.invoke({"context": docs_content, "question": question})
    return {"messages": [AIMessage(content=response)]}

def rewrite(self, state: AgentState) -> Dict[str, Any]:
    """Rewrite the question to improve retrieval."""
    messages = state["messages"]
    question = messages[0].content if messages else ""

    rewrite_prompt = (
        "Look at the input and reason about the underlying semantic intent/meaning.\n"
        "Here is the initial question:\n"
        f"{question}\n\n"
        "Formulate an improved question:"
    )

    response = self.llm.invoke([HumanMessage(content=rewrite_prompt)])
    return {"messages": [response]}