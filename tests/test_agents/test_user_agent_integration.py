"""Integration tests for UserAgent with event system."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from simu_net.agents.user_agent import UserAgent
from simu_net.agents.registry import AgentRegistry
from simu_net.data.models import PersonaType
from simu_net.events import EventManager, AgentEvent


class TestUserAgentIntegration:
    """Test UserAgent integration with other components."""
    
    @pytest.fixture
    async def event_manager(self):
        """Create and start event manager."""
        manager = AsyncMock(spec=EventManager)
        return manager
    
    @pytest.fixture
    async def agent_registry(self):
        """Create agent registry."""
        registry = AgentRegistry()
        registry.register_agent_type("UserAgent", UserAgent)
        return registry
    
    async def test_user_agent_with_registry(self, agent_registry):
        """Test UserAgent creation through registry."""
        # Create user agent through registry
        user = await agent_registry.create_agent(
            "UserAgent",
            agent_id="registry-user-1",
            persona_type=PersonaType.INFLUENCER
        )
        
        assert isinstance(user, UserAgent)
        assert user.agent_id == "registry-user-1"
        assert user.persona_type == PersonaType.INFLUENCER
        
        # Verify it's registered
        retrieved_user = await agent_registry.get_agent("registry-user-1")
        assert retrieved_user is user
    
    async def test_multiple_user_agents_interaction(self, event_manager, agent_registry):
        """Test multiple user agents interacting."""
        # Create multiple user agents with different personas
        casual_user = await agent_registry.create_agent(
            "UserAgent",
            agent_id="casual-1",
            persona_type=PersonaType.CASUAL,
            event_manager=event_manager
        )
        
        influencer_user = await agent_registry.create_agent(
            "UserAgent", 
            agent_id="influencer-1",
            persona_type=PersonaType.INFLUENCER,
            event_manager=event_manager
        )
        
        bot_user = await agent_registry.create_agent(
            "UserAgent",
            agent_id="bot-1", 
            persona_type=PersonaType.BOT,
            event_manager=event_manager
        )
        
        # Start all agents
        await casual_user.start()
        await influencer_user.start()
        await bot_user.start()
        
        # Let them run for a short time
        await asyncio.sleep(0.2)
        
        # Check that they're all running
        assert casual_user.is_running
        assert influencer_user.is_running
        assert bot_user.is_running
        
        # Stop all agents
        await casual_user.stop()
        await influencer_user.stop()
        await bot_user.stop()
        
        # Verify they're stopped
        assert not casual_user.is_running
        assert not influencer_user.is_running
        assert not bot_user.is_running
    
    async def test_user_agent_event_publishing(self, event_manager):
        """Test that user agents publish events correctly."""
        user = UserAgent(
            agent_id="publisher-user",
            persona_type=PersonaType.INFLUENCER,
            event_manager=event_manager
        )
        
        # Manually trigger content creation
        await user._create_content()
        
        # Verify event was published
        event_manager.publish.assert_called_once()
        
        # Check event details
        published_event = event_manager.publish.call_args[0][0]
        assert published_event.event_type == "content_created"
        assert published_event.source_agent == user.agent_id
        assert "content_id" in published_event.payload
        assert "text_content" in published_event.payload
    
    async def test_user_agent_event_handling(self, event_manager):
        """Test that user agents handle events correctly."""
        user = UserAgent(
            agent_id="handler-user",
            persona_type=PersonaType.CASUAL,
            event_manager=event_manager
        )
        
        initial_followers = user.user_data.follower_count
        
        # Create a network connection event targeting this user
        event = AgentEvent(
            event_type="network_connection_created",
            source_agent="other-user",
            target_agents=[],
            payload={
                "connection_id": "test-connection",
                "user_a_id": "other-user",
                "user_b_id": user.agent_id,
                "connection_type": "follow"
            },
            timestamp=pytest.approx_now(),
            correlation_id="test-correlation"
        )
        
        # Handle the event
        await user._handle_event(event)
        
        # Should have increased follower count
        assert user.user_data.follower_count == initial_followers + 1
    
    async def test_persona_behavior_differences(self, event_manager):
        """Test that different personas behave differently."""
        # Create users with different personas
        casual = UserAgent(
            agent_id="casual-test",
            persona_type=PersonaType.CASUAL,
            event_manager=event_manager
        )
        
        bot = UserAgent(
            agent_id="bot-test",
            persona_type=PersonaType.BOT,
            event_manager=event_manager
        )
        
        # Compare behavior parameters
        assert casual.behavior_params["posting_frequency"] < bot.behavior_params["posting_frequency"]
        assert casual.behavior_params["engagement_likelihood"] < bot.behavior_params["engagement_likelihood"]
        
        # Compare tick intervals
        assert casual._get_tick_interval() > bot._get_tick_interval()
        
        # Compare content generation
        casual_content = casual._generate_content_text()
        bot_content = bot._generate_content_text()
        
        # Content should be different in style (this is probabilistic)
        assert isinstance(casual_content, str)
        assert isinstance(bot_content, str)
        assert len(casual_content) > 0
        assert len(bot_content) > 0
    
    async def test_behavior_parameter_updates(self, event_manager):
        """Test dynamic behavior parameter updates."""
        user = UserAgent(
            agent_id="updatable-user",
            persona_type=PersonaType.CASUAL,
            event_manager=event_manager
        )
        
        initial_posting_freq = user.behavior_params["posting_frequency"]
        initial_engagement = user.behavior_params["engagement_likelihood"]
        
        # Update behavior parameters
        new_params = {
            "posting_frequency": initial_posting_freq * 2,
            "engagement_likelihood": initial_engagement * 1.5,
            "custom_param": 0.75
        }
        
        user.update_behavior_params(new_params)
        
        # Verify updates
        assert user.behavior_params["posting_frequency"] == initial_posting_freq * 2
        assert user.behavior_params["engagement_likelihood"] == initial_engagement * 1.5
        assert user.behavior_params["custom_param"] == 0.75
        
        # Verify user data model is also updated
        assert user.user_data.posting_frequency == initial_posting_freq * 2
        assert user.user_data.engagement_likelihood == initial_engagement * 1.5
    
    async def test_user_stats_tracking(self, event_manager):
        """Test user statistics tracking."""
        user = UserAgent(
            agent_id="stats-user",
            persona_type=PersonaType.INFLUENCER,
            event_manager=event_manager
        )
        
        # Get initial stats
        initial_stats = user.get_user_stats()
        
        # Simulate some activity
        await user._grow_network()  # Should increase following count
        
        # Get updated stats
        updated_stats = user.get_user_stats()
        
        # Verify stats changed
        assert updated_stats["following_count"] > initial_stats["following_count"]
        assert updated_stats["network_connections"] > initial_stats["network_connections"]
        
        # Verify all required fields are present
        required_fields = [
            "agent_id", "persona_type", "behavior_params",
            "follower_count", "following_count", "influence_score",
            "credibility_score", "network_connections",
            "content_creation_cooldown", "last_active",
            "is_active", "is_running"
        ]
        
        for field in required_fields:
            assert field in updated_stats
    
    async def test_agent_lifecycle_with_registry(self, agent_registry):
        """Test complete agent lifecycle through registry."""
        # Create agent
        user = await agent_registry.create_agent(
            "UserAgent",
            agent_id="lifecycle-user",
            persona_type=PersonaType.ACTIVIST
        )
        
        # Start through registry
        success = await agent_registry.start_agent("lifecycle-user")
        assert success is True
        assert user.is_running
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Stop through registry
        success = await agent_registry.stop_agent("lifecycle-user")
        assert success is True
        assert not user.is_running
        
        # Unregister
        unregistered = await agent_registry.unregister_agent("lifecycle-user")
        assert unregistered is user
        
        # Verify it's no longer in registry
        retrieved = await agent_registry.get_agent("lifecycle-user")
        assert retrieved is None