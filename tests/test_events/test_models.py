"""Tests for event models."""

import pytest
from datetime import datetime

from simu_net.events.models import AgentEvent


class TestAgentEvent:
    """Test agent event model."""
    
    def test_event_creation(self):
        """Test basic event creation."""
        timestamp = datetime.utcnow()
        
        event = AgentEvent(
            event_type="test_event",
            source_agent="agent-1",
            target_agents=["agent-2", "agent-3"],
            payload={"message": "hello", "value": 42},
            timestamp=timestamp,
            correlation_id="test-correlation"
        )
        
        assert event.event_type == "test_event"
        assert event.source_agent == "agent-1"
        assert event.target_agents == ["agent-2", "agent-3"]
        assert event.payload == {"message": "hello", "value": 42}
        assert event.timestamp == timestamp
        assert event.correlation_id == "test-correlation"
    
    def test_event_defaults(self):
        """Test event creation with defaults."""
        timestamp = datetime.utcnow()
        
        event = AgentEvent(
            event_type="test_event",
            source_agent="agent-1",
            timestamp=timestamp,
            correlation_id="test-correlation"
        )
        
        assert event.target_agents == []
        assert event.payload == {}
    
    def test_event_serialization(self):
        """Test event JSON serialization."""
        timestamp = datetime.utcnow()
        
        event = AgentEvent(
            event_type="test_event",
            source_agent="agent-1",
            target_agents=["agent-2"],
            payload={"data": "test"},
            timestamp=timestamp,
            correlation_id="test-correlation"
        )
        
        # Test JSON serialization
        json_data = event.model_dump_json()
        assert isinstance(json_data, str)
        
        # Test deserialization
        import json
        data = json.loads(json_data)
        reconstructed_event = AgentEvent(**data)
        
        assert reconstructed_event.event_type == event.event_type
        assert reconstructed_event.source_agent == event.source_agent
        assert reconstructed_event.target_agents == event.target_agents
        assert reconstructed_event.payload == event.payload
        assert reconstructed_event.correlation_id == event.correlation_id
    
    def test_event_validation(self):
        """Test event validation."""
        timestamp = datetime.utcnow()
        
        # Missing required fields should raise validation error
        with pytest.raises(ValueError):
            AgentEvent(
                source_agent="agent-1",
                timestamp=timestamp,
                correlation_id="test-correlation"
                # Missing event_type
            )
        
        with pytest.raises(ValueError):
            AgentEvent(
                event_type="test_event",
                timestamp=timestamp,
                correlation_id="test-correlation"
                # Missing source_agent
            )
    
    def test_event_equality(self):
        """Test event equality comparison."""
        timestamp = datetime.utcnow()
        
        event1 = AgentEvent(
            event_type="test_event",
            source_agent="agent-1",
            target_agents=["agent-2"],
            payload={"data": "test"},
            timestamp=timestamp,
            correlation_id="test-correlation"
        )
        
        event2 = AgentEvent(
            event_type="test_event",
            source_agent="agent-1",
            target_agents=["agent-2"],
            payload={"data": "test"},
            timestamp=timestamp,
            correlation_id="test-correlation"
        )
        
        event3 = AgentEvent(
            event_type="different_event",
            source_agent="agent-1",
            target_agents=["agent-2"],
            payload={"data": "test"},
            timestamp=timestamp,
            correlation_id="test-correlation"
        )
        
        assert event1 == event2
        assert event1 != event3