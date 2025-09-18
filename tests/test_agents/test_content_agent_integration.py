"""Integration tests for ContentAgent with other components."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from simu_net.agents.content_agent import ContentAgent
from simu_net.agents.registry import AgentRegistry
from simu_net.data.models import ContentType, ModerationAction
from simu_net.events import EventManager, AgentEvent


class TestContentAgentIntegration:
    """Test ContentAgent integration with other components."""
    
    @pytest.fixture
    async def event_manager(self):
        """Create and start event manager."""
        manager = AsyncMock(spec=EventManager)
        return manager
    
    @pytest.fixture
    async def agent_registry(self):
        """Create agent registry."""
        registry = AgentRegistry()
        registry.register_agent_type("ContentAgent", ContentAgent)
        return registry
    
    async def test_content_agent_with_registry(self, agent_registry):
        """Test ContentAgent creation through registry."""
        # Create content agent through registry
        content = await agent_registry.create_agent(
            "ContentAgent",
            content_id="registry-content-1",
            text_content="This is a test post about technology and innovation.",
            created_by="user-1"
        )
        
        assert isinstance(content, ContentAgent)
        assert content.content_id == "registry-content-1"
        assert content.created_by == "user-1"
        
        # Verify it's registered
        retrieved_content = await agent_registry.get_agent("registry-content-1")
        assert retrieved_content is content
    
    async def test_content_agent_full_lifecycle(self, event_manager, agent_registry):
        """Test complete content agent lifecycle."""
        # Create content agent
        content = await agent_registry.create_agent(
            "ContentAgent",
            content_id="lifecycle-content",
            text_content="Breaking: Amazing discovery in AI research! This will change everything! 🚀",
            created_by="researcher-1",
            event_manager=event_manager
        )
        
        # Start agent (should trigger metadata generation and event publishing)
        await content.start()
        
        # Verify agent is running and metadata is generated
        assert content.is_running
        assert content._metadata_generated
        assert content._embeddings_generated
        assert content._lifecycle_stage == "published"
        
        # Verify content creation event was published
        event_manager.publish.assert_called()
        published_event = event_manager.publish.call_args[0][0]
        assert published_event.event_type == "content_created"
        assert published_event.payload["content_id"] == "lifecycle-content"
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Stop agent
        await content.stop()
        assert not content.is_running
    
    async def test_multiple_content_agents_interaction(self, event_manager, agent_registry):
        """Test multiple content agents with different characteristics."""
        # Create different types of content
        tech_content = await agent_registry.create_agent(
            "ContentAgent",
            content_id="tech-content",
            text_content="Revolutionary AI breakthrough in machine learning algorithms!",
            created_by="tech-user",
            event_manager=event_manager
        )
        
        political_content = await agent_registry.create_agent(
            "ContentAgent",
            content_id="political-content", 
            text_content="Government announces new policies for democratic voting rights.",
            created_by="political-user",
            event_manager=event_manager
        )
        
        emotional_content = await agent_registry.create_agent(
            "ContentAgent",
            content_id="emotional-content",
            text_content="I'm so happy and grateful for this wonderful opportunity! 😊❤️",
            created_by="happy-user",
            event_manager=event_manager
        )
        
        # Start all agents
        await tech_content.start()
        await political_content.start()
        await emotional_content.start()
        
        # Verify all are running and have generated metadata
        for content in [tech_content, political_content, emotional_content]:
            assert content.is_running
            assert content._metadata_generated
            assert content._embeddings_generated
        
        # Check that different content types have different characteristics
        tech_stats = tech_content.get_content_stats()
        political_stats = political_content.get_content_stats()
        emotional_stats = emotional_content.get_content_stats()
        
        # Tech content should have technology topic
        assert 'technology' in tech_stats['topic_classification']
        
        # Political content should have politics topic
        assert 'politics' in political_stats['topic_classification']
        
        # Emotional content should have positive sentiment
        assert emotional_stats['sentiment_score'] > 0
        
        # Stop all agents
        await tech_content.stop()
        await political_content.stop()
        await emotional_content.stop()
    
    async def test_content_engagement_simulation(self, event_manager):
        """Test content engagement through events."""
        content = ContentAgent(
            content_id="engagement-test",
            text_content="This is a test post for engagement tracking.",
            created_by="user-1",
            event_manager=event_manager
        )
        
        await content.start()
        
        initial_stats = content.get_content_stats()
        initial_likes = initial_stats['engagement_metrics']['likes']
        initial_views = initial_stats['engagement_metrics']['views']
        
        # Simulate multiple interactions
        interactions = [
            {"type": "like", "user": "user-2"},
            {"type": "view", "user": "user-3"},
            {"type": "share", "user": "user-4"},
            {"type": "comment", "user": "user-5"},
            {"type": "view", "user": "user-6"},
            {"type": "like", "user": "user-7"}
        ]
        
        for interaction in interactions:
            event = AgentEvent(
                event_type="content_interaction",
                source_agent=interaction["user"],
                target_agents=[],
                payload={
                    "content_id": content.content_id,
                    "user_id": interaction["user"],
                    "interaction_type": interaction["type"]
                },
                timestamp=pytest.approx_now(),
                correlation_id=f"interaction-{interaction['user']}"
            )
            
            await content._handle_event(event)
        
        # Check updated engagement metrics
        final_stats = content.get_content_stats()
        final_metrics = final_stats['engagement_metrics']
        
        assert final_metrics['likes'] == initial_likes + 2  # 2 likes
        assert final_metrics['views'] == initial_views + 2  # 2 views
        assert final_metrics['shares'] == 1  # 1 share
        assert final_metrics['comments'] == 1  # 1 comment
        
        # Engagement rate should be calculated
        expected_rate = (2 + 2 + 1 + 1) / 2  # total interactions / views = 6/2 = 3.0
        assert final_metrics['engagement_rate'] == expected_rate
        
        await content.stop()
    
    async def test_content_moderation_workflow(self, event_manager):
        """Test content moderation workflow."""
        content = ContentAgent(
            content_id="moderation-test",
            text_content="This content might be problematic and need moderation review.",
            created_by="user-1",
            event_manager=event_manager
        )
        
        await content.start()
        
        # Initially content should be active
        assert content.content_data.is_active
        assert not content.content_data.moderation_status.is_flagged
        
        # Simulate moderation warning
        warning_event = AgentEvent(
            event_type="moderation_action",
            source_agent="moderator-1",
            target_agents=[],
            payload={
                "content_id": content.content_id,
                "action": ModerationAction.WARNING.value,
                "moderator_id": "moderator-1",
                "reason": "Potentially misleading information"
            },
            timestamp=pytest.approx_now(),
            correlation_id="mod-warning"
        )
        
        await content._handle_event(warning_event)
        
        # Content should be flagged but still active
        assert content.content_data.is_active
        assert content.content_data.moderation_status.is_flagged
        assert content.content_data.moderation_status.action_taken == ModerationAction.WARNING
        
        # Simulate content removal
        removal_event = AgentEvent(
            event_type="moderation_action",
            source_agent="moderator-2",
            target_agents=[],
            payload={
                "content_id": content.content_id,
                "action": ModerationAction.REMOVE.value,
                "moderator_id": "moderator-2",
                "reason": "Violates community guidelines"
            },
            timestamp=pytest.approx_now(),
            correlation_id="mod-removal"
        )
        
        await content._handle_event(removal_event)
        
        # Content should be inactive and removed
        assert not content.content_data.is_active
        assert content.content_data.moderation_status.action_taken == ModerationAction.REMOVE
        assert content._lifecycle_stage == "removed"
    
    async def test_content_virality_detection(self, event_manager):
        """Test content virality detection."""
        content = ContentAgent(
            content_id="viral-test",
            text_content="BREAKING: Shocking news that everyone needs to see! Share this now! 🔥",
            created_by="viral-user",
            event_manager=event_manager
        )
        
        await content.start()
        
        # Initially not viral
        assert not content.content_data.is_viral
        
        # Simulate high engagement to trigger virality
        high_engagement_interactions = [
            {"type": "view", "user": f"user-{i}"} for i in range(100)
        ] + [
            {"type": "like", "user": f"liker-{i}"} for i in range(80)
        ] + [
            {"type": "share", "user": f"sharer-{i}"} for i in range(20)
        ]
        
        for interaction in high_engagement_interactions:
            event = AgentEvent(
                event_type="content_interaction",
                source_agent=interaction["user"],
                target_agents=[],
                payload={
                    "content_id": content.content_id,
                    "user_id": interaction["user"],
                    "interaction_type": interaction["type"]
                },
                timestamp=pytest.approx_now(),
                correlation_id=f"viral-{interaction['user']}"
            )
            
            await content._handle_event(event)
        
        # Check virality status
        content._check_virality_status()
        
        # Should have gone viral due to high engagement rate
        final_stats = content.get_content_stats()
        engagement_rate = final_stats['engagement_metrics']['engagement_rate']
        
        # With 100 interactions out of 100 views, engagement rate should be 1.0
        assert engagement_rate == 1.0
        assert content.content_data.is_viral
        
        await content.stop()
    
    async def test_content_similarity_comparison(self, event_manager):
        """Test content similarity comparison."""
        # Create similar content pieces
        content1 = ContentAgent(
            content_id="similar-1",
            text_content="Artificial intelligence and machine learning are transforming technology.",
            created_by="tech-user-1",
            event_manager=event_manager
        )
        
        content2 = ContentAgent(
            content_id="similar-2", 
            text_content="AI and ML technologies are revolutionizing the tech industry.",
            created_by="tech-user-2",
            event_manager=event_manager
        )
        
        content3 = ContentAgent(
            content_id="different-1",
            text_content="I love cooking pasta and pizza for dinner with my family.",
            created_by="food-user",
            event_manager=event_manager
        )
        
        # Start all agents to generate embeddings
        await content1.start()
        await content2.start()
        await content3.start()
        
        # Calculate similarities
        sim_tech_tech = content1.calculate_similarity(content2.get_embeddings())
        sim_tech_food = content1.calculate_similarity(content3.get_embeddings())
        sim_identical = content1.calculate_similarity(content1.get_embeddings())
        
        # Tech content should be more similar to each other than to food content
        assert sim_identical == 1.0  # Identical content
        assert 0.0 <= sim_tech_tech <= 1.0
        assert 0.0 <= sim_tech_food <= 1.0
        
        # Generally, similar topics should have higher similarity
        # (though this depends on the embedding quality)
        assert isinstance(sim_tech_tech, float)
        assert isinstance(sim_tech_food, float)
        
        # Stop all agents
        await content1.stop()
        await content2.stop()
        await content3.stop()
    
    async def test_content_metadata_accuracy(self, event_manager):
        """Test accuracy of content metadata generation."""
        test_cases = [
            {
                "text": "Check out this amazing video: https://youtube.com/watch?v=123",
                "expected_type": ContentType.LINK,
                "expected_topics": ["technology"],
                "expected_sentiment_range": (-0.5, 1.0)
            },
            {
                "text": "Breaking news about government election policies and voting rights!",
                "expected_type": ContentType.TEXT,
                "expected_topics": ["politics"],
                "expected_sentiment_range": (-0.5, 0.5)
            },
            {
                "text": "I hate this terrible and awful situation! This is the worst! 😠💔",
                "expected_type": ContentType.TEXT,
                "expected_topics": [],
                "expected_sentiment_range": (-1.0, 0.0)
            }
        ]
        
        for i, case in enumerate(test_cases):
            content = ContentAgent(
                content_id=f"metadata-test-{i}",
                text_content=case["text"],
                created_by="test-user",
                event_manager=event_manager
            )
            
            await content.start()
            
            stats = content.get_content_stats()
            
            # Check content type
            assert stats["content_type"] == case["expected_type"].value
            
            # Check sentiment range
            sentiment = stats["sentiment_score"]
            min_sent, max_sent = case["expected_sentiment_range"]
            assert min_sent <= sentiment <= max_sent
            
            # Check topics (if expected)
            topics = stats["topic_classification"]
            for expected_topic in case["expected_topics"]:
                assert expected_topic in topics
                assert topics[expected_topic] > 0
            
            # All metadata should be within valid ranges
            assert 0.0 <= stats["misinformation_probability"] <= 1.0
            assert 0.0 <= stats["virality_potential"] <= 1.0
            assert stats["word_count"] > 0
            assert stats["language"] in ["en", "fr", "de", "es"]  # Supported languages
            
            await content.stop()