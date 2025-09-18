"""Tests for base agent functionality."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from simu_net.agents.base import SimuNetAgent, AgentState
from simu_net.events import EventManager, AgentEvent


class TestAgent(SimuNetAgent):
    """Test agent implementation."""
    
    def __init__(self, tick_interval: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.tick_interval = tick_interval
        self.tick_count = 0
        self.events_received = []
    
    async def _process_tick(self) -> None:
        """Process a single tick."""
        self.tick_count += 1
    
    def _get_tick_interval(self) -> float:
        """Get tick interval."""
        return self.tick_interval
    
    def _get_subscribed_events(self) -> list[str]:
        """Get subscribed events."""
        return ["test_event", "broadcast_event"]
    
    async def _on_event_received(self, event: AgentEvent) -> None:
        """Handle received events."""
        self.events_received.append(event)


class TestAgentState:
    """Test agent state model."""
    
    def test_agent_state_creation(self):
        """Test agent state creation."""
        state = AgentState(
            agent_id="test-agent",
            agent_type="TestAgent",
            created_at=pytest.approx_now(),
            last_active=pytest.approx_now()
        )
        
        assert state.agent_id == "test-agent"
        assert state.agent_type == "TestAgent"
        assert state.is_active is True
        assert isinstance(state.metadata, dict)


class TestSimuNetAgent:
    """Test base agent functionality."""
    
    @pytest.fixture
    def mock_event_manager(self):
        """Create mock event manager."""
        manager = AsyncMock(spec=EventManager)
        return manager
    
    @pytest.fixture
    def test_agent(self, mock_event_manager):
        """Create test agent."""
        return TestAgent(
            agent_id="test-agent-1",
            event_manager=mock_event_manager,
            tick_interval=0.01
        )
    
    def test_agent_initialization(self, test_agent):
        """Test agent initialization."""
        assert test_agent.agent_id == "test-agent-1"
        assert test_agent.state.agent_type == "TestAgent"
        assert test_agent.state.is_active is True
        assert not test_agent.is_running
        assert test_agent.is_active
    
    def test_agent_auto_id_generation(self):
        """Test automatic agent ID generation."""
        agent = TestAgent()
        assert agent.agent_id is not None
        assert len(agent.agent_id) > 0
    
    async def test_agent_lifecycle(self, test_agent):
        """Test agent start/stop lifecycle."""
        # Initially not running
        assert not test_agent.is_running
        
        # Start agent
        await test_agent.start()
        assert test_agent.is_running
        assert test_agent.is_active
        
        # Wait for a few ticks
        await asyncio.sleep(0.05)
        assert test_agent.tick_count > 0
        
        # Stop agent
        await test_agent.stop()
        assert not test_agent.is_running
        assert not test_agent.is_active
    
    async def test_agent_pause_resume(self, test_agent):
        """Test agent pause/resume functionality."""
        await test_agent.start()
        
        # Pause agent
        await test_agent.pause()
        assert not test_agent.is_active
        assert test_agent.is_running  # Still running but not active
        
        tick_count_before = test_agent.tick_count
        await asyncio.sleep(0.05)
        
        # Should not process ticks when paused
        assert test_agent.tick_count == tick_count_before
        
        # Resume agent
        await test_agent.resume()
        assert test_agent.is_active
        
        await asyncio.sleep(0.05)
        
        # Should process ticks again
        assert test_agent.tick_count > tick_count_before
        
        await test_agent.stop()
    
    async def test_event_publishing(self, test_agent, mock_event_manager):
        """Test event publishing."""
        await test_agent.publish_event(
            event_type="test_event",
            payload={"message": "hello"},
            target_agents=["agent-2"]
        )
        
        # Verify event manager was called
        mock_event_manager.publish.assert_called_once()
        
        # Check event details
        published_event = mock_event_manager.publish.call_args[0][0]
        assert published_event.event_type == "test_event"
        assert published_event.source_agent == test_agent.agent_id
        assert published_event.target_agents == ["agent-2"]
        assert published_event.payload == {"message": "hello"}
    
    async def test_event_subscription(self, test_agent, mock_event_manager):
        """Test event subscription."""
        await test_agent.start()
        
        # Verify subscriptions were made
        expected_events = ["test_event", "broadcast_event"]
        assert mock_event_manager.subscribe.call_count == len(expected_events)
        
        # Check each subscription
        for call in mock_event_manager.subscribe.call_args_list:
            event_type, handler = call[0]
            assert event_type in expected_events
            assert handler == test_agent._handle_event
        
        await test_agent.stop()
    
    async def test_event_handling(self, test_agent):
        """Test event handling."""
        # Create test event
        event = AgentEvent(
            event_type="test_event",
            source_agent="other-agent",
            target_agents=[test_agent.agent_id],
            payload={"data": "test"},
            timestamp=pytest.approx_now(),
            correlation_id="test-correlation"
        )
        
        # Handle event
        await test_agent._handle_event(event)
        
        # Verify event was received
        assert len(test_agent.events_received) == 1
        assert test_agent.events_received[0] == event
    
    async def test_event_filtering(self, test_agent):
        """Test event filtering."""
        # Event from self (should be ignored)
        self_event = AgentEvent(
            event_type="test_event",
            source_agent=test_agent.agent_id,
            target_agents=[],
            payload={},
            timestamp=pytest.approx_now(),
            correlation_id="self-event"
        )
        
        await test_agent._handle_event(self_event)
        assert len(test_agent.events_received) == 0
        
        # Event not targeted to this agent (should be ignored)
        targeted_event = AgentEvent(
            event_type="test_event",
            source_agent="other-agent",
            target_agents=["different-agent"],
            payload={},
            timestamp=pytest.approx_now(),
            correlation_id="targeted-event"
        )
        
        await test_agent._handle_event(targeted_event)
        assert len(test_agent.events_received) == 0
        
        # Broadcast event (should be received)
        broadcast_event = AgentEvent(
            event_type="test_event",
            source_agent="other-agent",
            target_agents=[],
            payload={},
            timestamp=pytest.approx_now(),
            correlation_id="broadcast-event"
        )
        
        await test_agent._handle_event(broadcast_event)
        assert len(test_agent.events_received) == 1
    
    def test_state_management(self, test_agent):
        """Test agent state management."""
        # Get initial state
        state = test_agent.get_state()
        assert state["agent_id"] == test_agent.agent_id
        assert state["agent_type"] == "TestAgent"
        assert state["is_active"] is True
        
        # Update metadata
        test_agent.update_metadata({"key": "value", "number": 42})
        
        updated_state = test_agent.get_state()
        assert updated_state["metadata"]["key"] == "value"
        assert updated_state["metadata"]["number"] == 42
    
    async def test_no_event_manager(self):
        """Test agent behavior without event manager."""
        agent = TestAgent(agent_id="no-event-agent")
        
        # Should not raise errors
        await agent.start()
        await agent.publish_event("test", {})
        await agent.stop()
    
    async def test_error_handling_in_tick(self, mock_event_manager):
        """Test error handling in agent tick processing."""
        
        class ErrorAgent(TestAgent):
            async def _process_tick(self):
                raise ValueError("Test error")
        
        agent = ErrorAgent(
            agent_id="error-agent",
            event_manager=mock_event_manager,
            tick_interval=0.01
        )
        
        await agent.start()
        
        # Wait for error to occur and be handled
        await asyncio.sleep(0.05)
        
        # Agent should still be running despite errors
        assert agent.is_running
        
        await agent.stop()
    
    async def test_multiple_start_stop_calls(self, test_agent):
        """Test multiple start/stop calls."""
        # Multiple starts should not cause issues
        await test_agent.start()
        await test_agent.start()  # Should log warning but not fail
        assert test_agent.is_running
        
        # Multiple stops should not cause issues
        await test_agent.stop()
        await test_agent.stop()  # Should log warning but not fail
        assert not test_agent.is_running