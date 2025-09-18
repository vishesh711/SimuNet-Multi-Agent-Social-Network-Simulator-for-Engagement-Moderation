"""Tests for agent registry."""

import pytest
from unittest.mock import AsyncMock

from simu_net.agents.base import SimuNetAgent
from simu_net.agents.registry import AgentRegistry


class MockAgent(SimuNetAgent):
    """Mock agent for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.started = False
        self.stopped = False
    
    async def _process_tick(self) -> None:
        pass
    
    def _get_tick_interval(self) -> float:
        return 1.0
    
    def _get_subscribed_events(self) -> list[str]:
        return []
    
    async def start(self) -> None:
        await super().start()
        self.started = True
    
    async def stop(self) -> None:
        await super().stop()
        self.stopped = True


class TestAgentRegistry:
    """Test agent registry functionality."""
    
    @pytest.fixture
    def registry(self):
        """Create agent registry."""
        return AgentRegistry()
    
    def test_registry_initialization(self, registry):
        """Test registry initialization."""
        stats = registry.get_stats()
        assert stats["total_agents"] == 0
        assert stats["active_agents"] == 0
        assert stats["running_agents"] == 0
        assert stats["agent_types"] == {}
        assert stats["registered_types"] == []
    
    def test_register_agent_type(self, registry):
        """Test agent type registration."""
        registry.register_agent_type("MockAgent", MockAgent)
        
        stats = registry.get_stats()
        assert "MockAgent" in stats["registered_types"]
    
    async def test_create_agent(self, registry):
        """Test agent creation."""
        registry.register_agent_type("MockAgent", MockAgent)
        
        agent = await registry.create_agent("MockAgent", agent_id="test-agent")
        
        assert agent.agent_id == "test-agent"
        assert isinstance(agent, MockAgent)
        
        # Verify agent is registered
        retrieved_agent = await registry.get_agent("test-agent")
        assert retrieved_agent is agent
    
    async def test_create_agent_auto_id(self, registry):
        """Test agent creation with auto-generated ID."""
        registry.register_agent_type("MockAgent", MockAgent)
        
        agent = await registry.create_agent("MockAgent")
        
        assert agent.agent_id is not None
        assert len(agent.agent_id) > 0
        
        # Verify agent is registered
        retrieved_agent = await registry.get_agent(agent.agent_id)
        assert retrieved_agent is agent
    
    async def test_create_unknown_agent_type(self, registry):
        """Test creating agent with unknown type."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            await registry.create_agent("UnknownAgent")
    
    async def test_register_existing_agent(self, registry):
        """Test registering an existing agent."""
        agent = MockAgent(agent_id="existing-agent")
        
        await registry.register_agent(agent)
        
        retrieved_agent = await registry.get_agent("existing-agent")
        assert retrieved_agent is agent
    
    async def test_unregister_agent(self, registry):
        """Test agent unregistration."""
        registry.register_agent_type("MockAgent", MockAgent)
        agent = await registry.create_agent("MockAgent", agent_id="test-agent")
        
        # Unregister agent
        unregistered_agent = await registry.unregister_agent("test-agent")
        assert unregistered_agent is agent
        
        # Verify agent is no longer registered
        retrieved_agent = await registry.get_agent("test-agent")
        assert retrieved_agent is None
    
    async def test_unregister_nonexistent_agent(self, registry):
        """Test unregistering non-existent agent."""
        result = await registry.unregister_agent("nonexistent")
        assert result is None
    
    async def test_get_agents_by_type(self, registry):
        """Test getting agents by type."""
        registry.register_agent_type("MockAgent", MockAgent)
        
        # Create multiple agents
        agent1 = await registry.create_agent("MockAgent", agent_id="agent-1")
        agent2 = await registry.create_agent("MockAgent", agent_id="agent-2")
        
        # Get agents by type
        mock_agents = await registry.get_agents_by_type("MockAgent")
        assert len(mock_agents) == 2
        assert agent1 in mock_agents
        assert agent2 in mock_agents
        
        # Test with non-existent type
        empty_list = await registry.get_agents_by_type("NonExistentType")
        assert len(empty_list) == 0
    
    async def test_get_all_agents(self, registry):
        """Test getting all agents."""
        registry.register_agent_type("MockAgent", MockAgent)
        
        # Initially empty
        all_agents = await registry.get_all_agents()
        assert len(all_agents) == 0
        
        # Create agents
        agent1 = await registry.create_agent("MockAgent", agent_id="agent-1")
        agent2 = await registry.create_agent("MockAgent", agent_id="agent-2")
        
        # Get all agents
        all_agents = await registry.get_all_agents()
        assert len(all_agents) == 2
        assert agent1 in all_agents
        assert agent2 in all_agents
    
    async def test_get_active_agents(self, registry):
        """Test getting active agents."""
        registry.register_agent_type("MockAgent", MockAgent)
        
        # Create agents
        agent1 = await registry.create_agent("MockAgent", agent_id="agent-1")
        agent2 = await registry.create_agent("MockAgent", agent_id="agent-2")
        
        # Initially both active
        active_agents = await registry.get_active_agents()
        assert len(active_agents) == 2
        
        # Pause one agent
        await agent1.pause()
        
        active_agents = await registry.get_active_agents()
        assert len(active_agents) == 1
        assert agent2 in active_agents
        assert agent1 not in active_agents
    
    async def test_start_agent(self, registry):
        """Test starting individual agent."""
        registry.register_agent_type("MockAgent", MockAgent)
        agent = await registry.create_agent("MockAgent", agent_id="test-agent")
        
        # Start agent
        result = await registry.start_agent("test-agent")
        assert result is True
        assert agent.started is True
        assert agent.is_running is True
        
        await agent.stop()
    
    async def test_start_nonexistent_agent(self, registry):
        """Test starting non-existent agent."""
        result = await registry.start_agent("nonexistent")
        assert result is False
    
    async def test_stop_agent(self, registry):
        """Test stopping individual agent."""
        registry.register_agent_type("MockAgent", MockAgent)
        agent = await registry.create_agent("MockAgent", agent_id="test-agent")
        
        await agent.start()
        
        # Stop agent
        result = await registry.stop_agent("test-agent")
        assert result is True
        assert agent.stopped is True
        assert agent.is_running is False
    
    async def test_stop_nonexistent_agent(self, registry):
        """Test stopping non-existent agent."""
        result = await registry.stop_agent("nonexistent")
        assert result is False
    
    async def test_start_all_agents(self, registry):
        """Test starting all agents."""
        registry.register_agent_type("MockAgent", MockAgent)
        
        # Create agents
        agent1 = await registry.create_agent("MockAgent", agent_id="agent-1")
        agent2 = await registry.create_agent("MockAgent", agent_id="agent-2")
        
        # Start all agents
        await registry.start_all_agents()
        
        assert agent1.started is True
        assert agent2.started is True
        assert agent1.is_running is True
        assert agent2.is_running is True
        
        # Cleanup
        await registry.stop_all_agents()
    
    async def test_stop_all_agents(self, registry):
        """Test stopping all agents."""
        registry.register_agent_type("MockAgent", MockAgent)
        
        # Create and start agents
        agent1 = await registry.create_agent("MockAgent", agent_id="agent-1")
        agent2 = await registry.create_agent("MockAgent", agent_id="agent-2")
        
        await agent1.start()
        await agent2.start()
        
        # Stop all agents
        await registry.stop_all_agents()
        
        assert agent1.stopped is True
        assert agent2.stopped is True
        assert agent1.is_running is False
        assert agent2.is_running is False
    
    async def test_cleanup_stopped_agents(self, registry):
        """Test cleanup of stopped agents."""
        registry.register_agent_type("MockAgent", MockAgent)
        
        # Create agents
        agent1 = await registry.create_agent("MockAgent", agent_id="agent-1")
        agent2 = await registry.create_agent("MockAgent", agent_id="agent-2")
        
        # Start and stop one agent
        await agent1.start()
        await agent1.stop()
        
        # Keep agent2 running
        await agent2.start()
        
        # Cleanup stopped agents
        removed_count = await registry.cleanup_stopped_agents()
        assert removed_count == 1
        
        # Verify only running agent remains
        all_agents = await registry.get_all_agents()
        assert len(all_agents) == 1
        assert agent2 in all_agents
        
        # Cleanup
        await agent2.stop()
    
    async def test_registry_stats(self, registry):
        """Test registry statistics."""
        registry.register_agent_type("MockAgent", MockAgent)
        
        # Create agents
        agent1 = await registry.create_agent("MockAgent", agent_id="agent-1")
        agent2 = await registry.create_agent("MockAgent", agent_id="agent-2")
        
        # Start one agent
        await agent1.start()
        
        # Pause one agent
        await agent2.pause()
        
        stats = registry.get_stats()
        assert stats["total_agents"] == 2
        assert stats["active_agents"] == 1  # agent2 is paused
        assert stats["running_agents"] == 1  # only agent1 is running
        assert stats["agent_types"]["MockAgent"] == 2
        assert "MockAgent" in stats["registered_types"]
        
        # Cleanup
        await agent1.stop()