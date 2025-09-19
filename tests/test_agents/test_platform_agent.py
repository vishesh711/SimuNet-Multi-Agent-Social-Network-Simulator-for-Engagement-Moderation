"""Tests for Platform Agent."""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from simu_net.agents.platform_agent import (
    PlatformAgent,
    FeedRankingMode,
    ExperimentStatus,
    ABTestExperiment
)
from simu_net.data.models import (
    ContentAgent as ContentAgentModel,
    UserAgent as UserAgentModel,
    PersonaType,
    EngagementMetrics,
    ContentMetadata,
    ModerationStatus,
    InteractionEvent
)
from simu_net.events import AgentEvent


class TestPlatformAgent:
    """Test suite for PlatformAgent class."""
    
    @pytest.fixture
    def mock_similarity_engine(self):
        """Create mock similarity search engine."""
        engine = Mock()
        engine.get_content_recommendations = AsyncMock(return_value=[
            ("content_001", 0.8, {}),
            ("content_002", 0.6, {}),
            ("content_003", 0.4, {})
        ])
        engine.search_content = AsyncMock(return_value=[
            ("content_001", 0.9, {}),
            ("content_002", 0.7, {})
        ])
        return engine
    
    @pytest.fixture
    def platform_agent(self, mock_similarity_engine):
        """Create a test platform agent."""
        return PlatformAgent(
            agent_id="platform_001",
            similarity_engine=mock_similarity_engine
        )
    
    @pytest.fixture
    def sample_user(self):
        """Create a sample user agent model."""
        return UserAgentModel(
            agent_id="user_001",
            persona_type=PersonaType.CASUAL,
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_content(self):
        """Create a sample content agent model."""
        return ContentAgentModel(
            content_id="content_001",
            text_content="This is a test post about technology",
            created_by="user_002",
            created_at=datetime.utcnow(),
            embeddings=[0.1] * 384,
            metadata=ContentMetadata(
                topic_classification={"technology": 0.8, "general": 0.2},
                sentiment_score=0.6,
                misinformation_probability=0.1,
                virality_potential=0.7
            ),
            engagement_metrics=EngagementMetrics(
                likes=10,
                shares=3,
                comments=2,
                views=50,
                engagement_rate=0.3
            ),
            moderation_status=ModerationStatus(
                is_flagged=False,
                confidence_scores={"toxicity": 0.1}
            )
        )
    
    def test_platform_agent_initialization(self, platform_agent):
        """Test platform agent initialization."""
        assert platform_agent.agent_id == "platform_001"
        assert platform_agent.ranking_mode == FeedRankingMode.BALANCED
        assert platform_agent.engagement_weight == 0.6
        assert platform_agent.safety_weight == 0.4
        assert platform_agent.personalization_strength == 0.5
        assert len(platform_agent.content_registry) == 0
        assert len(platform_agent.user_registry) == 0
        assert len(platform_agent.active_experiments) == 0
    
    @pytest.mark.asyncio
    async def test_content_registration(self, platform_agent, sample_content):
        """Test content registration through events."""
        # Create content creation event
        event = AgentEvent(
            event_id=str(uuid.uuid4()),
            event_type="content_created",
            source_agent="user_002",
            target_agents=["platform_001"],
            payload={
                "content_id": "content_001",
                "text_content": "Test content",
                "created_by": "user_002",
                "created_at": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            correlation_id=str(uuid.uuid4())
        )
        
        # Handle the event
        await platform_agent._handle_content_created(event)
        
        # Verify content was registered
        assert "content_001" in platform_agent.content_registry
        content = platform_agent.content_registry["content_001"]
        assert content.content_id == "content_001"
        assert content.created_by == "user_002"
    
    @pytest.mark.asyncio
    async def test_user_registration(self, platform_agent, sample_user):
        """Test user registration through events."""
        # Create user registration event
        event = AgentEvent(
            event_id=str(uuid.uuid4()),
            event_type="user_registered",
            source_agent="user_001",
            target_agents=["platform_001"],
            payload={
                "agent_id": "user_001",
                "persona_type": "casual",
                "created_at": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            correlation_id=str(uuid.uuid4())
        )
        
        # Handle the event
        await platform_agent._handle_user_registered(event)
        
        # Verify user was registered
        assert "user_001" in platform_agent.user_registry
        assert "user_001" in platform_agent.user_feeds
        user = platform_agent.user_registry["user_001"]
        assert user.agent_id == "user_001"
        assert user.persona_type == PersonaType.CASUAL
    
    @pytest.mark.asyncio
    async def test_content_interaction_handling(self, platform_agent, sample_content):
        """Test content interaction event handling."""
        # Register content first
        platform_agent.content_registry["content_001"] = sample_content
        
        # Create interaction event
        event = AgentEvent(
            event_id=str(uuid.uuid4()),
            event_type="content_interaction",
            source_agent="user_001",
            target_agents=["platform_001"],
            payload={
                "user_id": "user_001",
                "content_id": "content_001",
                "interaction_type": "like",
                "timestamp": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            correlation_id=str(uuid.uuid4())
        )
        
        # Handle the event
        await platform_agent._handle_content_interaction(event)
        
        # Verify interaction was recorded
        assert "user_001" in platform_agent.engagement_history
        interactions = platform_agent.engagement_history["user_001"]
        assert len(interactions) == 1
        assert interactions[0].interaction_type == "like"
        assert interactions[0].content_id == "content_001"
        
        # Verify content engagement metrics were updated
        content = platform_agent.content_registry["content_001"]
        assert content.engagement_metrics.likes == 11  # Original 10 + 1
    
    def test_engagement_score_calculation(self, platform_agent, sample_content):
        """Test engagement score calculation."""
        score = platform_agent._calculate_engagement_score(sample_content)
        
        assert 0.0 <= score <= 1.0
        assert score > 0  # Should have positive score due to engagement
    
    def test_safety_score_calculation(self, platform_agent, sample_content):
        """Test safety score calculation."""
        score = platform_agent._calculate_safety_score(sample_content)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should have high safety score (not flagged, low misinformation)
    
    @pytest.mark.asyncio
    async def test_personalization_score_calculation(self, platform_agent, sample_user, sample_content):
        """Test personalization score calculation."""
        # Add some engagement history
        platform_agent.engagement_history["user_001"] = [
            InteractionEvent(
                interaction_id=str(uuid.uuid4()),
                user_id="user_001",
                content_id="content_002",
                interaction_type="like",
                timestamp=datetime.utcnow()
            )
        ]
        
        score = await platform_agent._calculate_personalization_score(sample_user, sample_content)
        
        assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_content_score_calculation(self, platform_agent, sample_user, sample_content):
        """Test overall content score calculation."""
        score = await platform_agent._calculate_content_score(
            sample_user,
            sample_content,
            FeedRankingMode.BALANCED,
            0.6,
            0.4
        )
        
        assert 0.0 <= score <= 1.0
        assert score > 0  # Should have positive score
    
    @pytest.mark.asyncio
    async def test_feed_ranking_modes(self, platform_agent, sample_user, sample_content):
        """Test different feed ranking modes."""
        # Test engagement-focused ranking
        engagement_score = await platform_agent._calculate_content_score(
            sample_user,
            sample_content,
            FeedRankingMode.ENGAGEMENT_FOCUSED,
            0.8,
            0.2
        )
        
        # Test safety-focused ranking
        safety_score = await platform_agent._calculate_content_score(
            sample_user,
            sample_content,
            FeedRankingMode.SAFETY_FOCUSED,
            0.2,
            0.8
        )
        
        # Test chronological ranking
        chrono_score = await platform_agent._calculate_content_score(
            sample_user,
            sample_content,
            FeedRankingMode.CHRONOLOGICAL,
            0.5,
            0.5
        )
        
        assert all(0.0 <= score <= 1.0 for score in [engagement_score, safety_score, chrono_score])
        
        # Chronological should be based on time, others on content features
        assert chrono_score != engagement_score or chrono_score != safety_score
    
    def test_viral_content_detection(self, platform_agent, sample_content):
        """Test viral content detection logic."""
        # Test non-viral content
        assert not platform_agent._is_content_viral(sample_content)
        
        # Make content viral by increasing metrics
        sample_content.engagement_metrics.engagement_rate = 0.15
        sample_content.engagement_metrics.shares = 15
        sample_content.engagement_metrics.reach = 150
        sample_content.metadata.virality_potential = 0.8
        
        # Should now be detected as viral
        assert platform_agent._is_content_viral(sample_content)
    
    def test_trending_score_calculation(self, platform_agent, sample_content):
        """Test trending score calculation."""
        score = platform_agent._calculate_trending_score(sample_content)
        
        assert score >= 0.0
        assert isinstance(score, float)
    
    @pytest.mark.asyncio
    async def test_user_feed_generation(self, platform_agent, sample_user, sample_content):
        """Test user feed generation."""
        # Register user and content
        platform_agent.user_registry["user_001"] = sample_user
        platform_agent.content_registry["content_001"] = sample_content
        
        # Update user feed
        await platform_agent._update_user_feed("user_001")
        
        # Verify feed was generated
        assert "user_001" in platform_agent.user_feeds
        feed = platform_agent.user_feeds["user_001"]
        assert len(feed) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in feed)
        assert all(isinstance(item[0], str) and isinstance(item[1], float) for item in feed)
    
    @pytest.mark.asyncio
    async def test_get_user_feed_api(self, platform_agent):
        """Test get user feed API method."""
        # Set up test feed
        platform_agent.user_feeds["user_001"] = [
            ("content_001", 0.9),
            ("content_002", 0.8),
            ("content_003", 0.7)
        ]
        
        # Test getting feed
        feed = await platform_agent.get_user_feed("user_001", limit=2)
        
        assert len(feed) == 2
        assert feed[0] == ("content_001", 0.9)
        assert feed[1] == ("content_002", 0.8)
    
    @pytest.mark.asyncio
    async def test_experiment_creation(self, platform_agent):
        """Test A/B experiment creation."""
        experiment_id = await platform_agent.create_experiment(
            name="Test Experiment",
            description="Testing engagement vs safety",
            control_config={"engagement_weight": 0.8, "safety_weight": 0.2},
            treatment_config={"engagement_weight": 0.4, "safety_weight": 0.6},
            traffic_split=0.5,
            duration_hours=24
        )
        
        assert experiment_id in platform_agent.active_experiments
        experiment = platform_agent.active_experiments[experiment_id]
        assert experiment.name == "Test Experiment"
        assert experiment.status == ExperimentStatus.DRAFT
        assert experiment.traffic_split == 0.5
    
    @pytest.mark.asyncio
    async def test_experiment_lifecycle(self, platform_agent):
        """Test complete A/B experiment lifecycle."""
        # Create experiment
        experiment_id = await platform_agent.create_experiment(
            name="Test Experiment",
            description="Test description",
            control_config={"engagement_weight": 0.8},
            treatment_config={"engagement_weight": 0.4}
        )
        
        # Start experiment
        success = await platform_agent.start_experiment(experiment_id)
        assert success
        
        experiment = platform_agent.active_experiments[experiment_id]
        assert experiment.status == ExperimentStatus.RUNNING
        assert experiment.start_time is not None
        assert experiment.end_time is not None
        
        # Stop experiment
        success = await platform_agent.stop_experiment(experiment_id)
        assert success
        
        experiment = platform_agent.active_experiments[experiment_id]
        assert experiment.status == ExperimentStatus.PAUSED
    
    @pytest.mark.asyncio
    async def test_experiment_user_assignment(self, platform_agent):
        """Test user assignment to experiment groups."""
        # Create and start experiment
        experiment_id = await platform_agent.create_experiment(
            name="Test Experiment",
            description="Test description",
            control_config={"engagement_weight": 0.8},
            treatment_config={"engagement_weight": 0.4},
            traffic_split=0.5
        )
        await platform_agent.start_experiment(experiment_id)
        
        # Get experiment config for multiple users
        configs = []
        for i in range(100):
            config = await platform_agent._get_user_experiment_config(f"user_{i:03d}")
            configs.append(config)
        
        # Should have both control and treatment configs
        control_configs = [c for c in configs if c and c.get("engagement_weight") == 0.8]
        treatment_configs = [c for c in configs if c and c.get("engagement_weight") == 0.4]
        
        assert len(control_configs) > 0
        assert len(treatment_configs) > 0
        
        # Traffic split should be approximately 50/50
        total_assigned = len(control_configs) + len(treatment_configs)
        control_ratio = len(control_configs) / total_assigned
        assert 0.3 < control_ratio < 0.7  # Allow some variance due to randomness
    
    @pytest.mark.asyncio
    async def test_experiment_results(self, platform_agent):
        """Test experiment results retrieval."""
        # Create experiment
        experiment_id = await platform_agent.create_experiment(
            name="Test Experiment",
            description="Test description",
            control_config={"engagement_weight": 0.8},
            treatment_config={"engagement_weight": 0.4}
        )
        
        # Get results
        results = await platform_agent.get_experiment_results(experiment_id)
        
        assert results is not None
        assert results["experiment_id"] == experiment_id
        assert results["name"] == "Test Experiment"
        assert results["status"] == ExperimentStatus.DRAFT.value
        assert "control_config" in results
        assert "treatment_config" in results
    
    @pytest.mark.asyncio
    async def test_platform_metrics(self, platform_agent, sample_user, sample_content):
        """Test platform metrics collection."""
        # Add some test data
        platform_agent.user_registry["user_001"] = sample_user
        platform_agent.content_registry["content_001"] = sample_content
        platform_agent.viral_content.append("content_001")
        platform_agent.trending_content.append("content_001")
        
        # Get metrics
        metrics = await platform_agent.get_platform_metrics()
        
        assert "active_users" in metrics
        assert "total_content" in metrics
        assert "viral_content_count" in metrics
        assert "trending_content_count" in metrics
        assert metrics["active_users"] == 1
        assert metrics["total_content"] == 1
        assert metrics["viral_content_count"] == 1
        assert metrics["trending_content_count"] == 1
    
    @pytest.mark.asyncio
    async def test_ranking_configuration_update(self, platform_agent):
        """Test ranking configuration updates."""
        # Update configuration
        await platform_agent.update_ranking_configuration(
            ranking_mode=FeedRankingMode.ENGAGEMENT_FOCUSED,
            engagement_weight=0.9,
            safety_weight=0.1
        )
        
        assert platform_agent.ranking_mode == FeedRankingMode.ENGAGEMENT_FOCUSED
        assert platform_agent.engagement_weight == 0.9
        assert platform_agent.safety_weight == 0.1
    
    @pytest.mark.asyncio
    async def test_moderation_action_handling(self, platform_agent, sample_content):
        """Test moderation action event handling."""
        # Register content
        platform_agent.content_registry["content_001"] = sample_content
        platform_agent.viral_content.append("content_001")
        platform_agent.trending_content.append("content_001")
        
        # Create moderation action event
        event = AgentEvent(
            event_id=str(uuid.uuid4()),
            event_type="moderation_action",
            source_agent="moderator_001",
            target_agents=["platform_001"],
            payload={
                "content_id": "content_001",
                "action": "remove"
            },
            timestamp=datetime.utcnow(),
            correlation_id=str(uuid.uuid4())
        )
        
        # Handle the event
        await platform_agent._handle_moderation_action(event)
        
        # Verify content was deactivated and removed from lists
        content = platform_agent.content_registry["content_001"]
        assert not content.is_active
        assert "content_001" not in platform_agent.viral_content
        assert "content_001" not in platform_agent.trending_content
    
    @pytest.mark.asyncio
    async def test_statistical_significance_calculation(self, platform_agent):
        """Test statistical significance calculation for experiments."""
        # Create experiment with mock data
        experiment = ABTestExperiment(
            experiment_id="test_exp",
            name="Test",
            description="Test",
            control_config={},
            treatment_config={}
        )
        
        # Add mock metrics and user assignments
        experiment.control_metrics = {"engagement_rate": 0.10}
        experiment.treatment_metrics = {"engagement_rate": 0.15}
        experiment.user_assignments = {
            f"user_{i:03d}": "control" if i < 50 else "treatment"
            for i in range(100)
        }
        
        # Calculate significance
        significance = await platform_agent._calculate_statistical_significance(experiment)
        
        assert "control_rate" in significance
        assert "treatment_rate" in significance
        assert "difference" in significance
        assert "p_value" in significance
        assert "significant" in significance
        assert significance["control_rate"] == 0.10
        assert significance["treatment_rate"] == 0.15
        assert significance["difference"] == 0.05
    
    def test_get_subscribed_events(self, platform_agent):
        """Test that platform agent subscribes to correct events."""
        events = platform_agent._get_subscribed_events()
        
        expected_events = [
            "content_created",
            "content_interaction",
            "user_registered",
            "moderation_action"
        ]
        
        for event in expected_events:
            assert event in events
    
    def test_get_tick_interval(self, platform_agent):
        """Test platform agent tick interval."""
        interval = platform_agent._get_tick_interval()
        assert interval == 10.0  # Platform agent should tick every 10 seconds


class TestABTestExperiment:
    """Test suite for ABTestExperiment class."""
    
    def test_experiment_initialization(self):
        """Test experiment initialization."""
        experiment = ABTestExperiment(
            experiment_id="test_001",
            name="Test Experiment",
            description="Testing something",
            control_config={"param": "control"},
            treatment_config={"param": "treatment"},
            traffic_split=0.6,
            duration_hours=48
        )
        
        assert experiment.experiment_id == "test_001"
        assert experiment.name == "Test Experiment"
        assert experiment.traffic_split == 0.6
        assert experiment.duration_hours == 48
        assert experiment.status == ExperimentStatus.DRAFT
        assert experiment.start_time is None
        assert experiment.end_time is None
        assert len(experiment.user_assignments) == 0


@pytest.mark.parametrize("ranking_mode,expected_weight_focus", [
    (FeedRankingMode.ENGAGEMENT_FOCUSED, "engagement"),
    (FeedRankingMode.SAFETY_FOCUSED, "safety"),
    (FeedRankingMode.BALANCED, "balanced"),
    (FeedRankingMode.CHRONOLOGICAL, "time"),
    (FeedRankingMode.PERSONALIZED, "personalization")
])
@pytest.mark.asyncio
async def test_ranking_mode_behavior(ranking_mode, expected_weight_focus):
    """Test that different ranking modes behave as expected."""
    platform_agent = PlatformAgent(agent_id="test_platform")
    
    # Create test user and content
    user = UserAgentModel(
        agent_id="test_user",
        persona_type=PersonaType.CASUAL,
        created_at=datetime.utcnow(),
        last_active=datetime.utcnow()
    )
    
    content = ContentAgentModel(
        content_id="test_content",
        text_content="Test content",
        created_by="other_user",
        created_at=datetime.utcnow(),
        metadata=ContentMetadata(
            sentiment_score=0.5,
            misinformation_probability=0.1,
            virality_potential=0.6
        ),
        engagement_metrics=EngagementMetrics(
            likes=10,
            shares=2,
            comments=1,
            views=50,
            engagement_rate=0.26
        )
    )
    
    # Calculate score with the ranking mode
    score = await platform_agent._calculate_content_score(
        user, content, ranking_mode, 0.6, 0.4
    )
    
    # Score should be valid
    assert 0.0 <= score <= 1.0
    
    # For chronological mode, score should be time-based
    if ranking_mode == FeedRankingMode.CHRONOLOGICAL:
        # Recent content should have high score
        assert score > 0.8