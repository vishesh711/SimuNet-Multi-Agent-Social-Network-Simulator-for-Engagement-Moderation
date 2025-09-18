"""Event models for agent communication."""

from datetime import datetime
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class AgentEvent(BaseModel):
    """Event model for agent communication."""
    
    event_type: str = Field(..., description="Type of event")
    source_agent: str = Field(..., description="ID of the agent that created the event")
    target_agents: List[str] = Field(
        default_factory=list,
        description="List of target agent IDs (empty for broadcast)"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload data"
    )
    timestamp: datetime = Field(..., description="Event creation timestamp")
    correlation_id: str = Field(..., description="Correlation ID for event tracking")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }