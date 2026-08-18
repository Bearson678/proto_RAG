from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel,Field,StringConstraints,conint

class AgentState(TypedDict):
    """A dictionary representing the state of an agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    
class RelevanceGrade(BaseModel):
    """Binary Score for relevance check."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")
    

class WebSearchInput(BaseModel):
    query: Annotated[str,StringConstraints(min_length=1)]
    num: Annotated[int,Field(ge=1,le=10,default=5)]
    
class AgenticRetrieverInput(BaseModel):
    query: Annotated[str,StringConstraints(min_length=1)]
    top_k: Annotated[int,Field(ge=1,le=20,default=5)]