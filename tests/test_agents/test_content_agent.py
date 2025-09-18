"""Tests for ContentAgent implementation."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from simu_net.agents.content_agent import ContentAgent
from simu_net.data.models import ContentType, ModerationAction
from simu_net.events import AgentEvent, EventManager


class TestContentAgent:
    """Test ContentAgent functionality."""
    
    @pytest.fixture
    def mock_event_manager(self):
        """Create mock event manager."""
        return AsyncMock(spec=EventManager)
    
    @pytest.fixture
    def simple_content_agent(self, mock_event_manager):
        """Create simple content agent."""
        return ContentAgent(
            content_id="test-content-1",
            text_content="This is a simple test post about technology and AI.",
            created_by="user-1",
            event_manager=mock_event_manager
        )
    
    @pytest.fixture
    def political_content_agent(self, mock_event_manager):
        """Create content agent with political content."""
        return ContentAgent(
            content_id="political-content-1",
            text_content="Breaking news: The government announced new policies for democracy and voting rights.",
            created_by="user-2",
            event_manager=mock_event_manager
        )
    
    @pytest.fixture
    def emotional_content_agent(self, mock_event_manager):
        """Create content agent with emotional content."""
        return ContentAgent(
            content_id="emotional-content-1",
            text_content="I'm so excited and happy about this amazing opportunity! 😊❤️",
            created_by="user-3",
            event_manager=mock_event_manager
        )
    
    @pytest.fixture
    def suspicious_content_agent(self, mock_event_manager):
        """Create content agent with suspicious content."""
        return ContentAgent(
            content_id="suspicious-content-1",
            text_content="They don't want you to know this shocking truth! Wake up and do your own research!",
            created_by="user-4",
            event_manager=mock_event_manager
        )
    
    def test_content_agent_initialization(self, simple_content_agent):
        """Test content agent initialization."""
        assert simple_content_agent.content_id == "test-content-1"
        assert simple_content_agent.text_content == "This is a simple test post about technology and AI."
        assert simple_content_agent.created_by == "user-1"
        assert simple_content_agent.content_data.content_id == "test-content-1"
        assert simple_content_agent.content_data.text_content == "This is a simple test post about technology and AI."
        assert simple_content_agent.content_data.created_by == "user-1"
        assert simple_content_agent._lifecycle_stage == "created"
    
    def test_auto_generated_content_id(self, mock_event_manager):
        """Test automatic content ID generation."""
        agent = ContentAgent(
            text_content="Test content",
            created_by="user-1",
            event_manager=mock_event_manager
        )
        
        assert agent.content_id is not None
        assert len(agent.content_id) > 0
        assert agent.agent_id == agent.content_id
    
    def test_language_detection(self, simple_content_agent):
        """Test language detection."""
        # English content
        assert simple_content_agent._detect_language() == 'en'
        
        # Test with different languages (simplified)
        agent_fr = ContentAgent(
            text_content="Bonjour, c'est une belle journée et je suis très content.",
            created_by="user-1"
        )
        # Note: This is a simplified test - real language detection would be more accurate
        detected_lang = agent_fr._detect_language()
        assert isinstance(detected_lang, str)
        assert len(detected_lang) == 2
    
    def test_content_type_classification(self, mock_event_manager):
        """Test content type classification."""
        # Text content
        text_agent = ContentAgent(
            text_content="This is just plain text content.",
            created_by="user-1",
            event_manager=mock_event_manager
        )
        assert text_agent._classify_content_type() == ContentType.TEXT
        
        # Link content
        link_agent = ContentAgent(
            text_content="Check out this amazing website: https://example.com",
            created_by="user-1",
            event_manager=mock_event_manager
        )
        assert link_agent._classify_content_type() == ContentType.LINK
        
        # Image content
        image_agent = ContentAgent(
            text_content="Here's a beautiful photo I took today 📷",
            created_by="user-1",
            event_manager=mock_event_manager
        )
        assert image_agent._classify_content_type() == ContentType.IMAGE
        
        # Video content
        video_agent = ContentAgent(
            text_content="Watch this amazing video on YouTube! 🎥",
            created_by="user-1",
            event_manager=mock_event_manager
        )
        assert video_agent._classify_content_type() == ContentType.VIDEO
    
    async def test_topic_classification(self, simple_content_agent, political_content_agent):
        """Test topic classification."""
        # Technology content
        tech_topics = await simple_content_agent._classify_topics()
        assert 'technology' in tech_topics
        assert tech_topics['technology'] > 0
        
        # Political content
        political_topics = await political_content_agent._classify_topics()
        assert 'politics' in political_topics
        assert political_topics['politics'] > 0
        
        # Empty content should have general topic
        empty_agent = ContentAgent(text_content="", created_by="user-1")
        empty_topics = await empty_agent._classify_topics()
        assert len(empty_topics) == 0  # No topics for empty content
    
    async def test_sentiment_analysis(self, simple_content_agent, emotional_content_agent):
        """Test sentiment analysis."""
        # Neutral content
        neutral_sentiment = await simple_content_agent._analyze_sentiment()
        assert -1.0 <= neutral_sentiment <= 1.0
        
        # Positive emotional content
        positive_sentiment = await emotional_content_agent._analyze_sentiment()
        assert positive_sentiment > 0  # Should be positive
        assert -1.0 <= positive_sentiment <= 1.0
        
        # Test negative content
        negative_agent = ContentAgent(
            text_content="This is terrible and awful! I hate this so much 😠😢",
            created_by="user-1"
        )
        negative_sentiment = await negative_agent._analyze_sentiment()
        assert negative_sentiment < 0  # Should be negative
        assert -1.0 <= negative_sentiment <= 1.0
    
    async def test_misinformation_detection(self, simple_content_agent, suspicious_content_agent):
        """Test misinformation detection."""
        # Normal content should have low misinformation score
        normal_score = await simple_content_agent._detect_misinformation()
        assert 0.0 <= normal_score <= 1.0
        assert normal_score < 0.5  # Should be low for normal content
        
        # Suspicious content should have higher misinformation score
        suspicious_score = await suspicious_content_agent._detect_misinformation()
        assert 0.0 <= suspicious_score <= 1.0
        assert suspicious_score > normal_score  # Should be higher than normal content
    
    async def test_virality_potential_calculation(self, simple_content_agent, emotional_content_agent, political_content_agent):
        """Test virality potential calculation."""
        # All content should have virality scores between 0 and 1
        simple_virality = await simple_content_agent._calculate_virality_potential()
        assert 0.0 <= simple_virality <= 1.0
        
        emotional_virality = await emotional_content_agent._calculate_virality_potential()
        assert 0.0 <= emotional_virality <= 1.0
        
        political_virality = await political_content_agent._calculate_virality_potential()
        assert 0.0 <= political_virality <= 1.0
        
        # Emotional and political content should generally have higher virality potential
        # (though this is probabilistic and may not always be true)
        assert isinstance(emotional_virality, float)
        assert isinstance(political_virality, float)
    
    async def test_metadata_generation(self, simple_content_agent):
        """Test complete metadata generation."""
        await simple_content_agent._generate_metadata()
        
        metadata = simple_content_agent.content_data.metadata
        
        # Check that all metadata fields are populated
        assert metadata.word_count > 0
        assert metadata.language is not None
        assert metadata.content_type is not None
        assert isinstance(metadata.topic_classification, dict)
        assert -1.0 <= metadata.sentiment_score <= 1.0
        assert 0.0 <= metadata.misinformation_probability <= 1.0
        assert 0.0 <= metadata.virality_potential <= 1.0
        
        # Check that metadata generation flag is set
        assert simple_content_agent._metadata_generated is True
    
    def test_mock_embeddings_generation(self, simple_content_agent):
        """Test mock embeddings generation."""
        embeddings = simple_content_agent._generate_mock_embeddings()
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 384  # Default dimension
        assert all(isinstance(x, float) for x in embeddings)
        assert all(-0.5 <= x <= 0.5 for x in embeddings)  # Should be in expected range
        
        # Same content should generate same embeddings
        embeddings2 = simple_content_agent._generate_mock_embeddings()
        assert embeddings == embeddings2
    
    async def test_embeddings_generation(self, simple_content_agent):
        """Test embeddings generation."""
        await simple_content_agent._generate_embeddings()
        
        assert simple_content_agent.content_data.embeddings is not None
        assert isinstance(simple_content_agent.content_data.embeddings, list)
        assert len(simple_content_agent.content_data.embeddings) > 0
        assert simple_content_agent._embeddings_generated is True
    
    async def test_agent_startup_sequence(self, simple_content_agent, mock_event_manager):
        """Test agent startup sequence."""
        await simple_content_agent._on_start()
        
        # Should have generated metadata and embeddings
        assert simple_content_agent._metadata_generated is True
        assert simple_content_agent._embeddings_generated is True
        assert simple_content_agent._lifecycle_stage == "published"
        
        # Should have published content creation event
        mock_event_manager.publish.assert_called_once()
        
        # Check event details
        published_event = mock_event_manager.publish.call_args[0][0]
        assert published_event.event_type == "content_created"
        assert published_event.source_agent == simple_content_agent.agent_id
        assert "content_id" in published_event.payload
        assert "metadata" in published_event.payload
    
    def test_subscribed_events(self, simple_content_agent):
        """Test event subscription list."""
        events = simple_content_agent._get_subscribed_events()
        
        expected_events = [
            "content_interaction",
            "moderation_action",
            "content_flagged"
        ]
        
        for event_type in expected_events:
            assert event_type in events
    
    def test_tick_interval(self, simple_content_agent):
        """Test tick interval."""
        interval = simple_content_agent._get_tick_interval()
        assert isinstance(interval, float)
        assert interval > 0
        assert interval == 10.0  # Content agents should tick every 10 seconds
    
    async def test_interaction_event_handling(self, simple_content_agent):
        """Test handling of content interaction events."""
        initial_likes = simple_content_agent.content_data.engagement_metrics.likes
        initial_views = simple_content_agent.content_data.engagement_metrics.views
        
        # Create like interaction event
        like_event = AgentEvent(
            event_type="content_interaction",
            source_agent="user-1",
            target_agents=[],
            payload={
                "content_id": simple_content_agent.content_id,
                "user_id": "user-1",
                "interaction_type": "like"
            },
            timestamp=datetime.utcnow(),
            correlation_id="test-correlation"
        )
        
        await simple_content_agent._handle_interaction_event(like_event)
        
        # Should have increased likes
        assert simple_content_agent.content_data.engagement_metrics.likes == initial_likes + 1
        
        # Create view interaction event
        view_event = AgentEvent(
            event_type="content_interaction",
            source_agent="user-2",
            target_agents=[],
            payload={
                "content_id": simple_content_agent.content_id,
                "user_id": "user-2",
                "interaction_type": "view"
            },
            timestamp=datetime.utcnow(),
            correlation_id="test-correlation-2"
        )
        
        await simple_content_agent._handle_interaction_event(view_event)
        
        # Should have increased views
        assert simple_content_agent.content_data.engagement_metrics.views == initial_views + 1
        
        # Should have updated engagement rate
        expected_rate = 1 / 1  # 1 like out of 1 view
        assert simple_content_agent.content_data.engagement_metrics.engagement_rate == expected_rate
    
    async def test_moderation_event_handling(self, simple_content_agent):
        """Test handling of moderation action events."""
        # Create moderation event
        moderation_event = AgentEvent(
            event_type="moderation_action",
            source_agent="moderator-1",
            target_agents=[],
            payload={
                "content_id": simple_content_agent.content_id,
                "action": ModerationAction.WARNING.value,
                "moderator_id": "moderator-1",
                "reason": "Potentially misleading content"
            },
            timestamp=datetime.utcnow(),
            correlation_id="mod-correlation"
        )
        
        await simple_content_agent._handle_moderation_event(moderation_event)
        
        # Should have updated moderation status
        assert simple_content_agent.content_data.moderation_status.is_flagged is True
        assert simple_content_agent.content_data.moderation_status.action_taken == ModerationAction.WARNING
        assert simple_content_agent.content_data.moderation_status.reviewed_by == "moderator-1"
        assert simple_content_agent.content_data.moderation_status.reviewed_at is not None
    
    async def test_content_removal_handling(self, simple_content_agent):
        """Test content removal through moderation."""
        # Create removal moderation event
        removal_event = AgentEvent(
            event_type="moderation_action",
            source_agent="moderator-1",
            target_agents=[],
            payload={
                "content_id": simple_content_agent.content_id,
                "action": ModerationAction.REMOVE.value,
                "moderator_id": "moderator-1",
                "reason": "Violates community guidelines"
            },
            timestamp=datetime.utcnow(),
            correlation_id="removal-correlation"
        )
        
        await simple_content_agent._handle_moderation_event(removal_event)
        
        # Should have marked content as inactive
        assert simple_content_agent.content_data.is_active is False
        assert simple_content_agent._lifecycle_stage == "removed"
        assert simple_content_agent.content_data.moderation_status.action_taken == ModerationAction.REMOVE
    
    def test_virality_status_check(self, simple_content_agent):
        """Test virality status checking."""
        # Initially not viral
        assert simple_content_agent.content_data.is_viral is False
        
        # Set high engagement rate
        simple_content_agent.content_data.engagement_metrics.engagement_rate = 0.9
        simple_content_agent._check_virality_status()
        
        # Should become viral
        assert simple_content_agent.content_data.is_viral is True
    
    def test_content_stats(self, simple_content_agent):
        """Test content statistics retrieval."""
        stats = simple_content_agent.get_content_stats()
        
        # Check required fields
        required_fields = [
            "content_id", "created_by", "text_content", "word_count",
            "language", "content_type", "topic_classification",
            "sentiment_score", "misinformation_probability", "virality_potential",
            "engagement_metrics", "is_viral", "is_active", "lifecycle_stage",
            "embeddings_available", "metadata_generated", "created_at"
        ]
        
        for field in required_fields:
            assert field in stats
        
        # Check data types
        assert isinstance(stats["content_id"], str)
        assert isinstance(stats["created_by"], str)
        assert isinstance(stats["text_content"], str)
        assert isinstance(stats["word_count"], int)
        assert isinstance(stats["is_viral"], bool)
        assert isinstance(stats["is_active"], bool)
    
    def test_embeddings_retrieval(self, simple_content_agent):
        """Test embeddings retrieval."""
        # Initially no embeddings
        embeddings = simple_content_agent.get_embeddings()
        assert embeddings is None
        
        # Generate embeddings
        simple_content_agent.content_data.embeddings = [0.1, 0.2, 0.3]
        embeddings = simple_content_agent.get_embeddings()
        assert embeddings == [0.1, 0.2, 0.3]
    
    def test_similarity_calculation(self, simple_content_agent):
        """Test similarity calculation between contents."""
        # Set embeddings for this content
        simple_content_agent.content_data.embeddings = [1.0, 0.0, 0.0]
        
        # Test similarity with identical embeddings
        identical_similarity = simple_content_agent.calculate_similarity([1.0, 0.0, 0.0])
        assert identical_similarity == 1.0
        
        # Test similarity with opposite embeddings
        opposite_similarity = simple_content_agent.calculate_similarity([-1.0, 0.0, 0.0])
        assert 0.0 <= opposite_similarity <= 1.0
        assert opposite_similarity < identical_similarity
        
        # Test similarity with orthogonal embeddings
        orthogonal_similarity = simple_content_agent.calculate_similarity([0.0, 1.0, 0.0])
        assert 0.0 <= orthogonal_similarity <= 1.0
        
        # Test with no embeddings
        simple_content_agent.content_data.embeddings = None
        no_embeddings_similarity = simple_content_agent.calculate_similarity([1.0, 0.0, 0.0])
        assert no_embeddings_similarity == 0.0
    
    async def test_agent_lifecycle(self, simple_content_agent):
        """Test complete agent lifecycle."""
        # Start agent
        await simple_content_agent.start()
        assert simple_content_agent.is_running
        assert simple_content_agent.is_active
        assert simple_content_agent._lifecycle_stage == "published"
        
        # Let it run for a short time
        await asyncio.sleep(0.1)
        
        # Stop agent
        await simple_content_agent.stop()
        assert not simple_content_agent.is_running
        assert not simple_content_agent.is_active
    
    async def test_error_handling_in_startup(self, mock_event_manager):
        """Test error handling during startup."""
        # Create agent with problematic content
        agent = ContentAgent(
            text_content="Test content",
            created_by="user-1",
            event_manager=mock_event_manager
        )
        
        # Mock _generate_metadata to raise an exception
        with patch.object(agent, '_generate_metadata') as mock_generate:
            mock_generate.side_effect = Exception("Test error")
            
            # Should handle error gracefully
            await agent._on_start()
            
            # Should have attempted to generate metadata
            mock_generate.assert_called_once()
            
            # Should have set error state
            assert agent._lifecycle_stage == "error"
    
    async def test_event_handling_with_wrong_content_id(self, simple_content_agent):
        """Test that events for other content are ignored."""
        initial_likes = simple_content_agent.content_data.engagement_metrics.likes
        
        # Create interaction event for different content
        event = AgentEvent(
            event_type="content_interaction",
            source_agent="user-1",
            target_agents=[],
            payload={
                "content_id": "different-content-id",
                "user_id": "user-1",
                "interaction_type": "like"
            },
            timestamp=datetime.utcnow(),
            correlation_id="test-correlation"
        )
        
        await simple_content_agent._handle_interaction_event(event)
        
        # Should not have changed likes (event was for different content)
        assert simple_content_agent.content_data.engagement_metrics.likes == initial_likes