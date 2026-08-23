from typing import Any, Dict, Literal
from AgentStates.agentic_rag_states import AgentState, RelevanceGrade
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_core.output_parsers import StrOutputParser


from langchain_core.messages import ToolMessage

def _collect_tool_context(messages) -> str:
    """Concatenate content from every ToolMessage gathered so far."""
    parts = [str(m.content) for m in messages if isinstance(m, ToolMessage) and m.content]
    return "\n\n---\n\n".join(parts)

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


def grade_documents(self, state: AgentState) -> Literal["generate", "rewrite"]:
    grader = self.llm.with_structured_output(RelevanceGrade)
    prompt = PromptTemplate(
        template=(
            "You are grading whether retrieved web search results cover ALL distinct parts of a "
            "user's question, not just one part.\n\n"
            "Retrieved context so far:\n{context}\n\n"
            "User question: {question}\n\n"
            "First mentally break the question into its distinct parts. Grade 'yes' only if every "
            "part has supporting evidence in the context. If any part is unaddressed, grade 'no'."
        ),
        input_variables=["context", "question"],
    )
    messages = state["messages"]
    question = messages[0].content if messages else ""
    context = _collect_tool_context(messages)

    scored = (prompt | grader).invoke({"question": question, "context": context})
    score = (scored.binary_score or "").strip().lower()

    rounds = state.get("research_rounds", 0)
    if score == "yes" or rounds >= 3:   # hard cap — avoids infinite rewrite loops
        return "generate"
    return "rewrite"


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
    context = _collect_tool_context(messages)   # from the earlier fix

    rewrite_prompt = (
        "Original question:\n"
        f"{question}\n\n"
        "Context gathered so far (may be incomplete):\n"
        f"{context}\n\n"
        "Identify which distinct part(s) of the original question are NOT yet answered, and write a "
        "focused follow-up search query targeting ONLY the missing part(s). Return only the query."
    )
    response = self.llm.invoke([HumanMessage(content=rewrite_prompt)])

    # Wrap as HumanMessage, not AIMessage: this becomes the next turn in the
    # shared conversation history, and gemini-3.x rejects requests whose final
    # message is an assistant turn ("model prefilling").
    return {
        "messages": [HumanMessage(content=response.content)],
        "research_rounds": state.get("research_rounds", 0) + 1,
    }


def route_after_agent(state: AgentState) -> Literal["tools", "generate", "restricted"]:
        """Route based on whether the agent called a tool, and whether prior
        research already exists in state."""
        messages = state["messages"]
        last = messages[-1] if messages else None

        tool_calls = getattr(last, "tool_calls", None) or []
        if tool_calls:
            return "tools"

        # No tool call this turn. Did an earlier round already gather results?
        has_prior_research = any(
            isinstance(m, ToolMessage) and m.content for m in messages
        )

        if has_prior_research:
            # Agent looped back after `rewrite`, decided it now has enough,
            # and answered directly instead of calling web_search again.
            # Don't discard the gathered context -- generate from it.
            return "generate"

        # No tool call, no prior research -> the request really is out of scope.
        return "restricted"