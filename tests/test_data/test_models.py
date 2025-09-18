"""Tests for data models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from simu_net.data.models import (
    PersonaType,
    ContentType,
    ModerationAction,
    ContentMetadata,
    EngagementMetrics,
    ModerationStatus,
    UserAgent,
    ContentAgent,
    InteractionEvent,
    NetworkConnection
)


class TestEnums:
    """Test enum definitions."""
    
    def test_persona_type_enum(self):
        """Test PersonaType enum values."""
        assert PersonaType.CASUAL == "casual"
        assert PersonaType.INFLUENCER == "influencer"
        assert PersonaType.BOT == "bot"
        assert PersonaType.ACTIVIST == "activist"
    
    def test_content_type_enum(self):
        """Test ContentType enum values."""
        assert ContentType.TEXT == "text"
        assert ContentType.IMAGE == "image"
        assert ContentType.VIDEO == "video"
        assert ContentType.LINK == "link"
    
    def test_moderation_action_enum(self):
        """Test ModerationAction enum values."""
        assert ModerationAction.NONE == "none"
        assert ModerationAction.WARNING == "warning"
        assert ModerationAction.SHADOW_BAN == "shadow_ban"
        assert ModerationAction.REMOVE == "remove"


class TestContentMetadata:
    """Test ContentMetadata model."""
    
    def test_content_metadata_creation(self):
        """Test basic content metadata creation."""
        metadata = ContentMetadata(
            topic_classification={"politics": 0.8, "technology": 0.2},
            sentiment_score=0.5,
            misinformation_probability=0.1,
            virality_potential=0.7,
            content_type=ContentType.TEXT,
            language="en",
            word_count=150
        )
        
        assert metadata.topic_classification == {"politics": 0.8, "technology": 0.2}
        assert metadata.sentiment_score == 0.5
        assert metadata.misinformation_probability == 0.1
        assert metadata.virality_potential == 0.7
        assert metadata.content_type == ContentType.TEXT
        assert metadata.language == "en"
        assert metadata.word_count == 150
    
    def test_content_metadata_defaults(self):
        """Test content metadata with default values."""
        metadata = ContentMetadata()
        
        assert metadata.topic_classification == {}
        assert metadata.sentiment_score == 0.0
        assert metadata.misinformation_probability == 0.0
        assert metadata.virality_potential == 0.0
        assert metadata.content_type == ContentType.TEXT
        assert metadata.language == "en"
        assert metadata.word_count == 0
    
    def test_content_metadata_validation(self):
        """Test content metadata validation."""
        # Sentiment score out of range
        with pytest.raises(ValidationError):
            ContentMetadata(sentiment_score=2.0)
        
        with pytest.raises(ValidationError):
            ContentMetadata(sentiment_score=-2.0)
        
        # Misinformation probability out of range
        with pytest.raises(ValidationError):
            ContentMetadata(misinformation_probability=1.5)
        
        with pytest.raises(ValidationError):
            ContentMetadata(misinformation_probability=-0.1)
        
        # Negative word count
        with pytest.raises(ValidationError):
            ContentMetadata(word_count=-1)


class TestEngagementMetrics:
    """Test EngagementMetrics model."""
    
    def test_engagement_metrics_creation(self):
        """Test basic engagement metrics creation."""
        metrics = EngagementMetrics(
            likes=100,
            shares=25,
            comments=50,
            views=1000,
            engagement_rate=0.175,
            propagation_depth=3,
            reach=5000
        )
        
        assert metrics.likes == 100
        assert metrics.shares == 25
        assert metrics.comments == 50
        assert metrics.views == 1000
        assert metrics.engagement_rate == 0.175
        assert metrics.propagation_depth == 3
        assert metrics.reach == 5000
    
    def test_engagement_metrics_defaults(self):
        """Test engagement metrics with default values."""
        metrics = EngagementMetrics()
        
        assert metrics.likes == 0
        assert metrics.shares == 0
        assert metrics.comments == 0
        assert metrics.views == 0
        assert metrics.engagement_rate == 0.0
        assert metrics.propagation_depth == 0
        assert metrics.reach == 0
    
    def test_calculate_engagement_rate(self):
        """Test engagement rate calculation."""
        metrics = EngagementMetrics(
            likes=50,
            shares=10,
            comments=15,
            views=1000
        )
        
        # Calculate engagement rate
        rate = metrics.calculate_engagement_rate()
        expected_rate = (50 + 10 + 15) / 1000  # 0.075
        
        assert rate == expected_rate
        assert metrics.engagement_rate == expected_rate
    
    def test_calculate_engagement_rate_zero_views(self):
        """Test engagement rate calculation with zero views."""
        metrics = EngagementMetrics(
            likes=10,
            shares=5,
            comments=3,
            views=0
        )
        
        rate = metrics.calculate_engagement_rate()
        assert rate == 0.0
        assert metrics.engagement_rate == 0.0
    
    def test_engagement_metrics_validation(self):
        """Test engagement metrics validation."""
        # Negative values should fail
        with pytest.raises(ValidationError):
            EngagementMetrics(likes=-1)
        
        with pytest.raises(ValidationError):
            EngagementMetrics(shares=-1)
        
        with pytest.raises(ValidationError):
            EngagementMetrics(views=-1)


class TestModerationStatus:
    """Test ModerationStatus model."""
    
    def test_moderation_status_creation(self):
        """Test basic moderation status creation."""
        timestamp = datetime.utcnow()
        
        status = ModerationStatus(
            is_flagged=True,
            violation_types=["hate_speech", "misinformation"],
            confidence_scores={"hate_speech": 0.9, "misinformation": 0.7},
            action_taken=ModerationAction.REMOVE,
            reviewed_by="moderator-1",
            reviewed_at=timestamp,
            appeal_status="pending"
        )
        
        assert status.is_flagged is True
        assert status.violation_types == ["hate_speech", "misinformation"]
        assert status.confidence_scores == {"hate_speech": 0.9, "misinformation": 0.7}
        assert status.action_taken == ModerationAction.REMOVE
        assert status.reviewed_by == "moderator-1"
        assert status.reviewed_at == timestamp
        assert status.appeal_status == "pending"
    
    def test_moderation_status_defaults(self):
        """Test moderation status with default values."""
        status = ModerationStatus()
        
        assert status.is_flagged is False
        assert status.violation_types == []
        assert status.confidence_scores == {}
        assert status.action_taken == ModerationAction.NONE
        assert status.reviewed_by is None
        assert status.reviewed_at is None
        assert status.appeal_status is None


class TestUserAgent:
    """Test UserAgent model."""
    
    def test_user_agent_creation(self):
        """Test basic user agent creation."""
        created_at = datetime.utcnow()
        last_active = datetime.utcnow()
        
        user = UserAgent(
            agent_id="user-1",
            persona_type=PersonaType.INFLUENCER,
            behavior_params={"posting_freq": 2.5, "engagement_rate": 0.8},
            network_connections=["user-2", "user-3"],
            engagement_history=["content-1", "content-2"],
            created_at=created_at,
            last_active=last_active,
            follower_count=1000,
            following_count=500,
            influence_score=0.8,
            credibility_score=0.9
        )
        
        assert user.agent_id == "user-1"
        assert user.persona_type == PersonaType.INFLUENCER
        assert user.behavior_params == {"posting_freq": 2.5, "engagement_rate": 0.8}
        assert user.network_connections == ["user-2", "user-3"]
        assert user.engagement_history == ["content-1", "content-2"]
        assert user.created_at == created_at
        assert user.last_active == last_active
        assert user.follower_count == 1000
        assert user.following_count == 500
        assert user.influence_score == 0.8
        assert user.credibility_score == 0.9
    
    def test_user_agent_defaults(self):
        """Test user agent with default values."""
        created_at = datetime.utcnow()
        last_active = datetime.utcnow()
        
        user = UserAgent(
            agent_id="user-1",
            persona_type=PersonaType.CASUAL,
            created_at=created_at,
            last_active=last_active
        )
        
        assert user.behavior_params == {}
        assert user.network_connections == []
        assert user.engagement_history == []
        assert user.follower_count == 0
        assert user.following_count == 0
        assert user.influence_score == 0.0
        assert user.credibility_score == 0.5
        assert user.posting_frequency == 1.0
        assert user.engagement_likelihood == 0.1
        assert user.misinformation_susceptibility == 0.1
    
    def test_user_agent_validation(self):
        """Test user agent validation."""
        created_at = datetime.utcnow()
        last_active = datetime.utcnow()
        
        # Negative follower count
        with pytest.raises(ValidationError):
            UserAgent(
                agent_id="user-1",
                persona_type=PersonaType.CASUAL,
                created_at=created_at,
                last_active=last_active,
                follower_count=-1
            )
        
        # Influence score out of range
        with pytest.raises(ValidationError):
            UserAgent(
                agent_id="user-1",
                persona_type=PersonaType.CASUAL,
                created_at=created_at,
                last_active=last_active,
                influence_score=1.5
            )


class TestContentAgent:
    """Test ContentAgent model."""
    
    def test_content_agent_creation(self):
        """Test basic content agent creation."""
        created_at = datetime.utcnow()
        
        content = ContentAgent(
            content_id="content-1",
            text_content="This is a test post about technology.",
            embeddings=[0.1, 0.2, 0.3, 0.4, 0.5],
            created_by="user-1",
            created_at=created_at,
            is_active=True,
            is_viral=False
        )
        
        assert content.content_id == "content-1"
        assert content.text_content == "This is a test post about technology."
        assert content.embeddings == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert content.created_by == "user-1"
        assert content.created_at == created_at
        assert content.is_active is True
        assert content.is_viral is False
        assert isinstance(content.metadata, ContentMetadata)
        assert isinstance(content.engagement_metrics, EngagementMetrics)
        assert isinstance(content.moderation_status, ModerationStatus)
    
    def test_content_agent_defaults(self):
        """Test content agent with default values."""
        created_at = datetime.utcnow()
        
        content = ContentAgent(
            content_id="content-1",
            text_content="Test content",
            created_by="user-1",
            created_at=created_at
        )
        
        assert content.embeddings is None
        assert content.updated_at is None
        assert content.is_active is True
        assert content.is_viral is False
        assert content.viral_threshold == 0.8
        assert content.parent_content_id is None
        assert content.child_content_ids == []
    
    def test_add_engagement(self):
        """Test adding engagement to content."""
        created_at = datetime.utcnow()
        
        content = ContentAgent(
            content_id="content-1",
            text_content="Test content",
            created_by="user-1",
            created_at=created_at
        )
        
        # Add engagement
        content.add_engagement(likes=10, shares=5, comments=3, views=100)
        
        assert content.engagement_metrics.likes == 10
        assert content.engagement_metrics.shares == 5
        assert content.engagement_metrics.comments == 3
        assert content.engagement_metrics.views == 100
        assert content.engagement_metrics.engagement_rate == 0.18  # (10+5+3)/100
        assert content.updated_at is not None
    
    def test_update_virality_status(self):
        """Test virality status update."""
        created_at = datetime.utcnow()
        
        content = ContentAgent(
            content_id="content-1",
            text_content="Test content",
            created_by="user-1",
            created_at=created_at,
            viral_threshold=0.5
        )
        
        # Initially not viral
        assert content.is_viral is False
        
        # Set high engagement rate
        content.engagement_metrics.engagement_rate = 0.6
        result = content.update_virality_status()
        
        assert result is True
        assert content.is_viral is True
        
        # Test with high virality potential
        content2 = ContentAgent(
            content_id="content-2",
            text_content="Test content 2",
            created_by="user-1",
            created_at=created_at,
            viral_threshold=0.5
        )
        
        content2.metadata.virality_potential = 0.7
        result2 = content2.update_virality_status()
        
        assert result2 is True
        assert content2.is_viral is True


class TestInteractionEvent:
    """Test InteractionEvent model."""
    
    def test_interaction_event_creation(self):
        """Test basic interaction event creation."""
        timestamp = datetime.utcnow()
        
        interaction = InteractionEvent(
            interaction_id="interaction-1",
            user_id="user-1",
            content_id="content-1",
            interaction_type="like",
            timestamp=timestamp,
            metadata={"source": "mobile_app", "duration": 5.2}
        )
        
        assert interaction.interaction_id == "interaction-1"
        assert interaction.user_id == "user-1"
        assert interaction.content_id == "content-1"
        assert interaction.interaction_type == "like"
        assert interaction.timestamp == timestamp
        assert interaction.metadata == {"source": "mobile_app", "duration": 5.2}
    
    def test_interaction_event_defaults(self):
        """Test interaction event with default values."""
        timestamp = datetime.utcnow()
        
        interaction = InteractionEvent(
            interaction_id="interaction-1",
            user_id="user-1",
            content_id="content-1",
            interaction_type="view",
            timestamp=timestamp
        )
        
        assert interaction.metadata == {}


class TestNetworkConnection:
    """Test NetworkConnection model."""
    
    def test_network_connection_creation(self):
        """Test basic network connection creation."""
        created_at = datetime.utcnow()
        
        connection = NetworkConnection(
            connection_id="connection-1",
            user_a_id="user-1",
            user_b_id="user-2",
            connection_type="follow",
            strength=0.8,
            created_at=created_at,
            is_active=True
        )
        
        assert connection.connection_id == "connection-1"
        assert connection.user_a_id == "user-1"
        assert connection.user_b_id == "user-2"
        assert connection.connection_type == "follow"
        assert connection.strength == 0.8
        assert connection.created_at == created_at
        assert connection.is_active is True
    
    def test_network_connection_defaults(self):
        """Test network connection with default values."""
        created_at = datetime.utcnow()
        
        connection = NetworkConnection(
            connection_id="connection-1",
            user_a_id="user-1",
            user_b_id="user-2",
            created_at=created_at
        )
        
        assert connection.connection_type == "follow"
        assert connection.strength == 1.0
        assert connection.is_active is True
    
    def test_network_connection_validation(self):
        """Test network connection validation."""
        created_at = datetime.utcnow()
        
        # Strength out of range
        with pytest.raises(ValidationError):
            NetworkConnection(
                connection_id="connection-1",
                user_a_id="user-1",
                user_b_id="user-2",
                created_at=created_at,
                strength=1.5
            )
        
        with pytest.raises(ValidationError):
            NetworkConnection(
                connection_id="connection-1",
                user_a_id="user-1",
                user_b_id="user-2",
                created_at=created_at,
                strength=-0.1
            )