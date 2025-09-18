"""Tests for UserAgent implementation."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from simu_net.agents.user_agent import UserAgent
from simu_net.data.models import PersonaType
from simu_net.events import AgentEvent, EventManager


class TestUserAgent:
    """Test UserAgent functionality."""
    
    @pytest.fixture
    def mock_event_manager(self):
        """Create mock event manager."""
        return AsyncMock(spec=EventManager)
    
    @pytest.fixture
    def casual_user(self, mock_event_manager):
        """Create casual user agent."""
        return UserAgent(
            agent_id="casual-user-1",
            persona_type=PersonaType.CASUAL,
            event_manager=mock_event_manager
        )
    
    @pytest.fixture
    def influencer_user(self, mock_event_manager):
        """Create influencer user agent."""
        return UserAgent(
            agent_id="influencer-user-1",
            persona_type=PersonaType.INFLUENCER,
            event_manager=mock_event_manager
        )
    
    @pytest.fixture
    def bot_user(self, mock_event_manager):
        """Create bot user agent."""
        return UserAgent(
            agent_id="bot-user-1",
            persona_type=PersonaType.BOT,
            event_manager=mock_event_manager
        )
    
    @pytest.fixture
    def activist_user(self, mock_event_manager):
        """Create activist user agent."""
        return UserAgent(
            agent_id="activist-user-1",
            persona_type=PersonaType.ACTIVIST,
            event_manager=mock_event_manager
        )
    
    def test_user_agent_initialization(self, casual_user):
        """Test user agent initialization."""
        assert casual_user.agent_id == "casual-user-1"
        assert casual_user.persona_type == PersonaType.CASUAL
        assert casual_user.user_data.agent_id == "casual-user-1"
        assert casual_user.user_data.persona_type == PersonaType.CASUAL
        assert isinstance(casual_user.behavior_params, dict)
        assert len(casual_user.behavior_params) > 0
    
    def test_persona_behavior_initialization(self, casual_user, influencer_user, bot_user, activist_user):
        """Test that different personas have different behavior parameters."""
        # Check that each persona has different posting frequencies
        assert casual_user.behavior_params["posting_frequency"] != influencer_user.behavior_params["posting_frequency"]
        assert influencer_user.behavior_params["posting_frequency"] != bot_user.behavior_params["posting_frequency"]
        assert bot_user.behavior_params["posting_frequency"] != activist_user.behavior_params["posting_frequency"]
        
        # Check that bot has highest posting frequency
        assert bot_user.behavior_params["posting_frequency"] > influencer_user.behavior_params["posting_frequency"]
        assert bot_user.behavior_params["posting_frequency"] > casual_user.behavior_params["posting_frequency"]
        
        # Check that casual user has lowest engagement likelihood
        assert casual_user.behavior_params["engagement_likelihood"] < influencer_user.behavior_params["engagement_likelihood"]
        assert casual_user.behavior_params["engagement_likelihood"] < bot_user.behavior_params["engagement_likelihood"]
    
    def test_custom_behavior_params(self, mock_event_manager):
        """Test custom behavior parameters override defaults."""
        custom_params = {
            "posting_frequency": 5.0,
            "engagement_likelihood": 0.9,
            "custom_param": 0.5
        }
        
        user = UserAgent(
            agent_id="custom-user",
            persona_type=PersonaType.CASUAL,
            behavior_params=custom_params,
            event_manager=mock_event_manager
        )
        
        # Custom parameters should be preserved
        assert user.behavior_params["posting_frequency"] == 5.0
        assert user.behavior_params["engagement_likelihood"] == 0.9
        assert user.behavior_params["custom_param"] == 0.5
        
        # Default parameters should still be present for non-overridden values
        assert "sharing_likelihood" in user.behavior_params
        assert "misinformation_susceptibility" in user.behavior_params
    
    def test_tick_interval_by_persona(self, casual_user, influencer_user, bot_user, activist_user):
        """Test that different personas have different tick intervals."""
        casual_interval = casual_user._get_tick_interval()
        influencer_interval = influencer_user._get_tick_interval()
        bot_interval = bot_user._get_tick_interval()
        activist_interval = activist_user._get_tick_interval()
        
        # Bot should have shortest interval (most active)
        assert bot_interval < casual_interval
        assert bot_interval < activist_interval
        
        # Influencer should be more active than casual
        assert influencer_interval < casual_interval
    
    def test_subscribed_events(self, casual_user):
        """Test event subscription list."""
        events = casual_user._get_subscribed_events()
        
        expected_events = [
            "content_created",
            "content_interaction", 
            "network_connection_created",
            "moderation_action"
        ]
        
        for event_type in expected_events:
            assert event_type in events
    
    @patch('random.random')
    async def test_content_creation_decision(self, mock_random, casual_user):
        """Test content creation decision logic."""
        # Mock random to return low value (should create content)
        mock_random.return_value = 0.01
        
        should_create = await casual_user._should_create_content()
        assert should_create is True
        
        # Mock random to return high value (should not create content)
        mock_random.return_value = 0.99
        
        should_create = await casual_user._should_create_content()
        assert should_create is False
    
    async def test_content_creation_cooldown(self, casual_user):
        """Test content creation cooldown mechanism."""
        # Initially no cooldown
        assert casual_user._content_creation_cooldown == 0.0
        
        # Set cooldown
        casual_user._set_content_creation_cooldown()
        
        # Should have cooldown now
        assert casual_user._content_creation_cooldown > 0
        
        # Should not create content while on cooldown
        should_create = await casual_user._should_create_content()
        assert should_create is False
    
    def test_content_text_generation(self, casual_user, influencer_user, bot_user, activist_user):
        """Test content text generation for different personas."""
        casual_content = casual_user._generate_content_text()
        influencer_content = influencer_user._generate_content_text()
        bot_content = bot_user._generate_content_text()
        activist_content = activist_user._generate_content_text()
        
        # All should generate non-empty strings
        assert isinstance(casual_content, str)
        assert len(casual_content) > 0
        assert isinstance(influencer_content, str)
        assert len(influencer_content) > 0
        assert isinstance(bot_content, str)
        assert len(bot_content) > 0
        assert isinstance(activist_content, str)
        assert len(activist_content) > 0
        
        # Content should be different (with high probability)
        # Run multiple times to check variety
        casual_contents = [casual_user._generate_content_text() for _ in range(10)]
        assert len(set(casual_contents)) > 1  # Should have variety
    
    @patch('random.random')
    async def test_engagement_decision(self, mock_random, casual_user):
        """Test engagement decision logic."""
        # Mock random to return low value (should engage)
        mock_random.return_value = 0.01
        
        should_engage = await casual_user._should_engage_with_content()
        assert should_engage is True
        
        # Mock random to return high value (should not engage)
        mock_random.return_value = 0.99
        
        should_engage = await casual_user._should_engage_with_content()
        assert should_engage is False
    
    def test_engagement_type_decision(self, casual_user):
        """Test engagement type decision logic."""
        # Run multiple times to test different outcomes
        engagement_types = []
        for _ in range(100):
            engagement_type = casual_user._decide_engagement_type()
            engagement_types.append(engagement_type)
        
        # Should have variety in engagement types
        unique_types = set(engagement_types)
        assert "like" in unique_types
        
        # Like should be most common
        like_count = engagement_types.count("like")
        assert like_count > len(engagement_types) * 0.3  # At least 30% likes
    
    async def test_create_content_action(self, casual_user, mock_event_manager):
        """Test content creation action."""
        await casual_user._create_content()
        
        # Should have published content creation event
        mock_event_manager.publish.assert_called_once()
        
        # Check event details
        published_event = mock_event_manager.publish.call_args[0][0]
        assert published_event.event_type == "content_created"
        assert published_event.source_agent == casual_user.agent_id
        assert "content_id" in published_event.payload
        assert "text_content" in published_event.payload
        assert published_event.payload["created_by"] == casual_user.agent_id
        
        # Should have set cooldown
        assert casual_user._content_creation_cooldown > 0
    
    async def test_engage_with_content_action(self, casual_user, mock_event_manager):
        """Test content engagement action."""
        await casual_user._engage_with_content()
        
        # Should have published interaction event
        mock_event_manager.publish.assert_called_once()
        
        # Check event details
        published_event = mock_event_manager.publish.call_args[0][0]
        assert published_event.event_type == "content_interaction"
        assert published_event.source_agent == casual_user.agent_id
        assert "interaction_id" in published_event.payload
        assert "user_id" in published_event.payload
        assert "content_id" in published_event.payload
        assert "interaction_type" in published_event.payload
        assert published_event.payload["user_id"] == casual_user.agent_id
    
    async def test_grow_network_action(self, casual_user, mock_event_manager):
        """Test network growth action."""
        initial_following = casual_user.user_data.following_count
        
        await casual_user._grow_network()
        
        # Should have published network connection event
        mock_event_manager.publish.assert_called_once()
        
        # Check event details
        published_event = mock_event_manager.publish.call_args[0][0]
        assert published_event.event_type == "network_connection_created"
        assert published_event.source_agent == casual_user.agent_id
        assert "connection_id" in published_event.payload
        assert "user_a_id" in published_event.payload
        assert "user_b_id" in published_event.payload
        assert published_event.payload["user_a_id"] == casual_user.agent_id
        
        # Should have updated following count
        assert casual_user.user_data.following_count == initial_following + 1
    
    def test_persona_compatibility(self, casual_user):
        """Test persona compatibility calculation."""
        # Test compatibility with same persona
        same_compatibility = casual_user._get_persona_compatibility(PersonaType.CASUAL.value)
        assert same_compatibility > 1.0  # Should be higher than baseline
        
        # Test compatibility with influencer (should be positive)
        influencer_compatibility = casual_user._get_persona_compatibility(PersonaType.INFLUENCER.value)
        assert influencer_compatibility > 1.0
        
        # Test compatibility with bot (should be lower)
        bot_compatibility = casual_user._get_persona_compatibility(PersonaType.BOT.value)
        assert bot_compatibility < 1.0
        
        # Test with invalid persona
        invalid_compatibility = casual_user._get_persona_compatibility("invalid_persona")
        assert invalid_compatibility == 1.0  # Should default to neutral
        
        # Test with None
        none_compatibility = casual_user._get_persona_compatibility(None)
        assert none_compatibility == 1.0
    
    async def test_content_created_event_handling(self, casual_user):
        """Test handling of content creation events from other agents."""
        # Create event from another agent
        event = AgentEvent(
            event_type="content_created",
            source_agent="other-agent",
            target_agents=[],
            payload={
                "content_id": "test-content-1",
                "created_by": "other-agent",
                "text_content": "Test content",
                "persona_type": PersonaType.INFLUENCER.value
            },
            timestamp=datetime.utcnow(),
            correlation_id="test-correlation"
        )
        
        # Handle the event
        await casual_user._handle_content_created(event)
        
        # Should potentially add to engagement targets
        # (This is probabilistic, so we can't assert definitively)
        # But we can check that the method runs without error
        assert True  # If we get here, no exception was raised
    
    async def test_network_connection_event_handling(self, casual_user):
        """Test handling of network connection events."""
        initial_followers = casual_user.user_data.follower_count
        
        # Create event where someone follows this user
        event = AgentEvent(
            event_type="network_connection_created",
            source_agent="other-agent",
            target_agents=[],
            payload={
                "connection_id": "test-connection-1",
                "user_a_id": "other-agent",
                "user_b_id": casual_user.agent_id,
                "connection_type": "follow"
            },
            timestamp=datetime.utcnow(),
            correlation_id="test-correlation"
        )
        
        # Handle the event
        await casual_user._handle_network_connection(event)
        
        # Should have increased follower count
        assert casual_user.user_data.follower_count == initial_followers + 1
    
    async def test_agent_tick_processing(self, casual_user):
        """Test agent tick processing."""
        # Mock the decision methods to return specific actions
        with patch.object(casual_user, '_decide_actions') as mock_decide:
            mock_decide.return_value = [
                {"type": "create_content"},
                {"type": "engage_content"}
            ]
            
            with patch.object(casual_user, '_execute_action') as mock_execute:
                await casual_user._process_tick()
                
                # Should have called decide_actions
                mock_decide.assert_called_once()
                
                # Should have executed both actions
                assert mock_execute.call_count == 2
                mock_execute.assert_any_call({"type": "create_content"})
                mock_execute.assert_any_call({"type": "engage_content"})
    
    def test_cooldown_updates(self, casual_user):
        """Test cooldown update mechanism."""
        # Set initial cooldown
        casual_user._content_creation_cooldown = 10.0
        
        # Update cooldowns
        casual_user._update_cooldowns()
        
        # Cooldown should be reduced by tick interval
        tick_interval = casual_user._get_tick_interval()
        expected_cooldown = 10.0 - tick_interval
        assert casual_user._content_creation_cooldown == max(0, expected_cooldown)
    
    def test_user_stats(self, casual_user):
        """Test user statistics retrieval."""
        stats = casual_user.get_user_stats()
        
        # Check required fields
        required_fields = [
            "agent_id", "persona_type", "behavior_params",
            "follower_count", "following_count", "influence_score",
            "credibility_score", "network_connections",
            "content_creation_cooldown", "last_active",
            "is_active", "is_running"
        ]
        
        for field in required_fields:
            assert field in stats
        
        # Check data types
        assert isinstance(stats["agent_id"], str)
        assert isinstance(stats["persona_type"], str)
        assert isinstance(stats["behavior_params"], dict)
        assert isinstance(stats["follower_count"], int)
        assert isinstance(stats["is_active"], bool)
    
    def test_behavior_params_update(self, casual_user):
        """Test behavior parameters update."""
        initial_posting_freq = casual_user.behavior_params["posting_frequency"]
        
        new_params = {
            "posting_frequency": 10.0,
            "new_custom_param": 0.8
        }
        
        casual_user.update_behavior_params(new_params)
        
        # Should have updated existing parameter
        assert casual_user.behavior_params["posting_frequency"] == 10.0
        assert casual_user.user_data.posting_frequency == 10.0
        
        # Should have added new parameter
        assert casual_user.behavior_params["new_custom_param"] == 0.8
        
        # Should have preserved other parameters
        assert "engagement_likelihood" in casual_user.behavior_params
    
    async def test_agent_lifecycle(self, casual_user):
        """Test complete agent lifecycle."""
        # Start agent
        await casual_user.start()
        assert casual_user.is_running
        assert casual_user.is_active
        
        # Let it run for a short time
        await asyncio.sleep(0.1)
        
        # Stop agent
        await casual_user.stop()
        assert not casual_user.is_running
        assert not casual_user.is_active
    
    async def test_error_handling_in_tick(self, casual_user):
        """Test error handling during tick processing."""
        # Mock _decide_actions to raise an exception
        with patch.object(casual_user, '_decide_actions') as mock_decide:
            mock_decide.side_effect = Exception("Test error")
            
            # Should not raise exception, but handle it gracefully
            await casual_user._process_tick()
            
            # Should have attempted to call _decide_actions
            mock_decide.assert_called_once()
    
    async def test_event_handling_errors(self, casual_user):
        """Test error handling in event processing."""
        # Create malformed event
        event = AgentEvent(
            event_type="content_created",
            source_agent="other-agent",
            target_agents=[],
            payload={},  # Missing required fields
            timestamp=datetime.utcnow(),
            correlation_id="test-correlation"
        )
        
        # Should handle gracefully without raising exception
        await casual_user._on_event_received(event)
        
        # If we get here, error was handled gracefully
        assert True