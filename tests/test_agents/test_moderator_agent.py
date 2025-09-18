"""Tests for moderator agent functionality."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from simu_net.agents.moderator_agent import (
    ModeratorAgent,
    PolicyConfig,
    ModerationResult
)
from simu_net.data.models import ModerationAction
from simu_net.events import EventManager, AgentEvent


class TestPolicyConfig:
    """Test policy configuration."""
    
    def test_default_policy_config(self):
        """Test default policy configuration."""
        config = PolicyConfig()
        
        assert config.strictness_level == "moderate"
        assert config.toxicity_threshold == 0.7
        assert config.hate_speech_threshold == 0.8
        assert config.misinformation_threshold == 0.6
        assert config.confidence_threshold == 0.5
        assert config.enable_shadow_ban is True
        assert config.enable_warnings is True
    
    def test_strict_policy_config(self):
        """Test strict policy configuration adjustments."""
        config = PolicyConfig(strictness_level="strict")
        
        # Thresholds should be lowered for strict enforcement
        assert config.toxicity_threshold < 0.7
        assert config.hate_speech_threshold < 0.8
        assert config.misinformation_threshold < 0.6
        assert config.confidence_threshold < 0.5
    
    def test_lenient_policy_config(self):
        """Test lenient policy configuration adjustments."""
        config = PolicyConfig(strictness_level="lenient")
        
        # Thresholds should be raised for lenient enforcement
        assert config.toxicity_threshold > 0.7
        assert config.hate_speech_threshold > 0.8
        assert config.misinformation_threshold > 0.6
        assert config.confidence_threshold > 0.5
    
    def test_custom_policy_config(self):
        """Test custom policy configuration."""
        config = PolicyConfig(
            strictness_level="moderate",
            toxicity_threshold=0.5,
            hate_speech_threshold=0.6,
            misinformation_threshold=0.4,
            confidence_threshold=0.3
        )
        
        assert config.toxicity_threshold == 0.5
        assert config.hate_speech_threshold == 0.6
        assert config.misinformation_threshold == 0.4
        assert config.confidence_threshold == 0.3


class TestModerationResult:
    """Test moderation result."""
    
    def test_moderation_result_creation(self):
        """Test moderation result creation."""
        violations = {"toxicity": 0.8, "hate_speech": 0.6}
        result = ModerationResult(
            content_id="test-content",
            violations=violations,
            recommended_action=ModerationAction.REMOVE,
            confidence=0.9,
            reasoning="High toxicity detected"
        )
        
        assert result.content_id == "test-content"
        assert result.violations == violations
        assert result.recommended_action == ModerationAction.REMOVE
        assert result.confidence == 0.9
        assert result.reasoning == "High toxicity detected"
        assert isinstance(result.timestamp, datetime)


class TestModeratorAgent:
    """Test moderator agent functionality."""
    
    @pytest.fixture
    def mock_event_manager(self):
        """Create mock event manager."""
        manager = AsyncMock(spec=EventManager)
        return manager
    
    @pytest.fixture
    def policy_config(self):
        """Create test policy configuration."""
        return PolicyConfig(
            strictness_level="moderate",
            toxicity_threshold=0.7,
            hate_speech_threshold=0.8,
            confidence_threshold=0.5
        )
    
    @pytest.fixture
    def moderator_agent(self, mock_event_manager, policy_config):
        """Create moderator agent for testing."""
        return ModeratorAgent(
            agent_id="test-moderator",
            event_manager=mock_event_manager,
            policy_config=policy_config
        )
    
    def test_moderator_agent_initialization(self, moderator_agent, policy_config):
        """Test moderator agent initialization."""
        assert moderator_agent.agent_id == "test-moderator"
        assert moderator_agent.policy_config.strictness_level == "moderate"
        assert moderator_agent.stats["content_analyzed"] == 0
        assert moderator_agent.stats["violations_detected"] == 0
        assert len(moderator_agent._content_queue) == 0
        assert len(moderator_agent._audit_log) == 0
    
    def test_auto_id_generation(self, mock_event_manager):
        """Test automatic agent ID generation."""
        agent = ModeratorAgent(event_manager=mock_event_manager)
        assert agent.agent_id is not None
        assert len(agent.agent_id) > 0
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, moderator_agent):
        """Test moderator agent lifecycle."""
        # Initially not running
        assert not moderator_agent.is_running
        
        # Start agent
        with patch.object(moderator_agent, '_load_models', new_callable=AsyncMock):
            await moderator_agent.start()
        
        assert moderator_agent.is_running
        assert moderator_agent.is_active
        
        # Stop agent
        await moderator_agent.stop()
        assert not moderator_agent.is_running
    
    @patch('simu_net.agents.moderator_agent._TRANSFORMERS_AVAILABLE', False)
    @pytest.mark.asyncio
    async def test_model_loading_fallback(self, moderator_agent):
        """Test model loading with transformers unavailable."""
        await moderator_agent._load_models()
        
        # Should handle gracefully without transformers
        assert moderator_agent._models_loaded is True
        assert moderator_agent._toxicity_classifier is None
        assert moderator_agent._hate_speech_classifier is None
    
    @pytest.mark.asyncio
    async def test_content_queue_processing(self, moderator_agent):
        """Test content queue processing."""
        # Add content to queue
        content_data = {
            "content_id": "test-content-1",
            "text_content": "This is test content",
            "created_by": "user-1"
        }
        
        moderator_agent._content_queue.append(content_data)
        assert len(moderator_agent._content_queue) == 1
        
        # Mock the analysis method
        with patch.object(moderator_agent, '_analyze_content', new_callable=AsyncMock) as mock_analyze:
            await moderator_agent._process_content_queue()
            
            # Should have processed the content
            mock_analyze.assert_called_once_with(content_data)
            assert len(moderator_agent._content_queue) == 0
    
    @pytest.mark.asyncio
    async def test_toxicity_detection_fallback(self, moderator_agent):
        """Test fallback toxicity detection."""
        # Test clean content
        clean_text = "This is a nice and friendly message"
        toxicity_score = moderator_agent._fallback_toxicity_detection(clean_text)
        assert toxicity_score == 0.0
        
        # Test toxic content
        toxic_text = "You are stupid and pathetic"
        toxicity_score = moderator_agent._fallback_toxicity_detection(toxic_text)
        assert toxicity_score > 0.0
        assert toxicity_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_hate_speech_detection_fallback(self, moderator_agent):
        """Test fallback hate speech detection."""
        # Test clean content
        clean_text = "I love this community"
        hate_score = moderator_agent._fallback_hate_speech_detection(clean_text)
        assert hate_score == 0.0
        
        # Test hate speech content
        hate_text = "I hate all of them, they are scum"
        hate_score = moderator_agent._fallback_hate_speech_detection(hate_text)
        assert hate_score > 0.0
        assert hate_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_misinformation_detection_fallback(self, moderator_agent):
        """Test fallback misinformation detection."""
        # Test normal content
        normal_text = "Here are some facts about climate change"
        misinfo_score = moderator_agent._fallback_misinformation_detection(normal_text)
        assert misinfo_score == 0.0
        
        # Test suspicious content
        suspicious_text = "They don't want you to know the truth about vaccines"
        misinfo_score = moderator_agent._fallback_misinformation_detection(suspicious_text)
        assert misinfo_score > 0.0
        assert misinfo_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_spam_detection(self, moderator_agent):
        """Test spam detection."""
        # Test normal content
        normal_text = "I enjoyed reading this article"
        spam_score = await moderator_agent._detect_spam(normal_text)
        assert spam_score < 0.1  # Allow for small floating point variations
        
        # Test spam content
        spam_text = "CLICK HERE NOW!!! BUY NOW!!! LIMITED TIME OFFER!!!"
        spam_score = await moderator_agent._detect_spam(spam_text)
        assert spam_score > 0.0
        assert spam_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_content_analysis_clean_content(self, moderator_agent):
        """Test content analysis with clean content."""
        clean_text = "This is a wonderful day and I'm feeling great!"
        
        result = await moderator_agent._perform_content_analysis("test-content", clean_text)
        
        assert result.content_id == "test-content"
        assert len(result.violations) == 0
        assert result.recommended_action == ModerationAction.NONE
        assert result.confidence > 0.5
        assert "No policy violations" in result.reasoning
    
    @pytest.mark.asyncio
    async def test_content_analysis_toxic_content(self, moderator_agent):
        """Test content analysis with toxic content."""
        toxic_text = "You are all stupid idiots and I hate you"
        
        result = await moderator_agent._perform_content_analysis("test-content", toxic_text)
        
        assert result.content_id == "test-content"
        assert len(result.violations) > 0
        assert result.recommended_action != ModerationAction.NONE
        assert result.confidence > 0.0
        assert "detected" in result.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_moderation_decision_low_confidence(self, moderator_agent):
        """Test moderation decision with low confidence."""
        result = ModerationResult(
            content_id="test-content",
            violations={"toxicity": 0.3},
            recommended_action=ModerationAction.WARNING,
            confidence=0.2,  # Below threshold
            reasoning="Low confidence detection"
        )
        
        action = moderator_agent._make_moderation_decision(result)
        assert action == ModerationAction.NONE
    
    @pytest.mark.asyncio
    async def test_moderation_decision_high_confidence(self, moderator_agent):
        """Test moderation decision with high confidence."""
        result = ModerationResult(
            content_id="test-content",
            violations={"toxicity": 0.8},
            recommended_action=ModerationAction.REMOVE,
            confidence=0.9,  # Above threshold
            reasoning="High confidence detection"
        )
        
        action = moderator_agent._make_moderation_decision(result)
        assert action == ModerationAction.REMOVE
    
    @pytest.mark.asyncio
    async def test_execute_moderation_action(self, moderator_agent, mock_event_manager):
        """Test executing moderation actions."""
        result = ModerationResult(
            content_id="test-content",
            violations={"toxicity": 0.8},
            recommended_action=ModerationAction.REMOVE,
            confidence=0.9,
            reasoning="High toxicity detected"
        )
        
        await moderator_agent._execute_moderation_action(result, ModerationAction.REMOVE)
        
        # Should publish moderation event
        mock_event_manager.publish.assert_called_once()
        
        # Check event details
        published_event = mock_event_manager.publish.call_args[0][0]
        assert published_event.event_type == "moderation_action"
        assert published_event.payload["content_id"] == "test-content"
        assert published_event.payload["action"] == "remove"
        
        # Should update statistics
        assert moderator_agent.stats["actions_taken"]["removals"] == 1
    
    @pytest.mark.asyncio
    async def test_audit_log_creation(self, moderator_agent):
        """Test audit log creation."""
        result = ModerationResult(
            content_id="test-content",
            violations={"toxicity": 0.8},
            recommended_action=ModerationAction.WARNING,
            confidence=0.9,
            reasoning="Toxicity detected"
        )
        
        initial_log_size = len(moderator_agent._audit_log)
        moderator_agent._log_moderation_action(result, ModerationAction.WARNING)
        
        assert len(moderator_agent._audit_log) == initial_log_size + 1
        
        # Check audit log entry
        log_entry = moderator_agent._audit_log[-1]
        assert log_entry["content_id"] == "test-content"
        assert log_entry["action"] == "warning"
        assert log_entry["confidence"] == 0.9
        assert "toxicity" in log_entry["violations"]
    
    @pytest.mark.asyncio
    async def test_content_created_event_handling(self, moderator_agent):
        """Test handling content creation events."""
        content_data = {
            "content_id": "new-content",
            "text_content": "This is new content",
            "created_by": "user-1"
        }
        
        event = AgentEvent(
            event_type="content_created",
            source_agent="user-1",
            target_agents=[],
            payload=content_data,
            timestamp=datetime.utcnow(),
            correlation_id="test-correlation"
        )
        
        initial_queue_size = len(moderator_agent._content_queue)
        await moderator_agent._handle_content_created(event)
        
        # Should add content to queue
        assert len(moderator_agent._content_queue) == initial_queue_size + 1
        assert moderator_agent._content_queue[-1]["content_id"] == "new-content"
    
    @pytest.mark.asyncio
    async def test_content_flagged_event_handling(self, moderator_agent):
        """Test handling content flagging events."""
        # Add content to queue first
        content_data = {
            "content_id": "flagged-content",
            "text_content": "This content was flagged",
            "created_by": "user-1"
        }
        moderator_agent._content_queue.append(content_data)
        
        # Create flagging event
        flag_data = {
            "content_id": "flagged-content",
            "reason": "inappropriate",
            "flagged_by": "user-2"
        }
        
        event = AgentEvent(
            event_type="content_flagged",
            source_agent="user-2",
            target_agents=[],
            payload=flag_data,
            timestamp=datetime.utcnow(),
            correlation_id="flag-correlation"
        )
        
        await moderator_agent._handle_content_flagged(event)
        
        # Should move flagged content to front of queue
        assert moderator_agent._content_queue[0]["content_id"] == "flagged-content"
        assert moderator_agent._content_queue[0]["user_flagged"] is True
    
    @pytest.mark.asyncio
    async def test_moderation_appeal_handling(self, moderator_agent):
        """Test handling moderation appeals."""
        # Add entry to audit log first
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "content_id": "appealed-content",
            "moderator_id": moderator_agent.agent_id,
            "action": "remove",
            "violations": {"toxicity": 0.8},
            "confidence": 0.9,
            "reasoning": "High toxicity"
        }
        moderator_agent._audit_log.append(audit_entry)
        
        # Create appeal event
        appeal_data = {
            "content_id": "appealed-content",
            "appeal_reason": "False positive",
            "appealed_by": "user-1"
        }
        
        event = AgentEvent(
            event_type="moderation_appeal",
            source_agent="user-1",
            target_agents=[],
            payload=appeal_data,
            timestamp=datetime.utcnow(),
            correlation_id="appeal-correlation"
        )
        
        initial_appeals = moderator_agent.stats["appeals_processed"]
        await moderator_agent._handle_moderation_appeal(event)
        
        # Should increment appeals processed
        assert moderator_agent.stats["appeals_processed"] == initial_appeals + 1
    
    def test_get_moderation_stats(self, moderator_agent):
        """Test getting moderation statistics."""
        stats = moderator_agent.get_moderation_stats()
        
        assert stats["agent_id"] == moderator_agent.agent_id
        assert "policy_config" in stats
        assert "statistics" in stats
        assert stats["models_loaded"] is not None
        assert stats["queue_size"] == 0
        assert stats["audit_log_size"] == 0
    
    def test_get_audit_log(self, moderator_agent):
        """Test getting audit log."""
        # Add some entries to audit log
        for i in range(5):
            moderator_agent._audit_log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "content_id": f"content-{i}",
                "action": "warning"
            })
        
        # Get limited audit log
        audit_log = moderator_agent.get_audit_log(limit=3)
        assert len(audit_log) == 3
        
        # Should return most recent entries
        assert audit_log[-1]["content_id"] == "content-4"
    
    def test_update_policy_config(self, moderator_agent):
        """Test updating policy configuration."""
        new_config = PolicyConfig(
            strictness_level="strict",
            toxicity_threshold=0.5,
            hate_speech_threshold=0.6
        )
        
        old_strictness = moderator_agent.policy_config.strictness_level
        moderator_agent.update_policy_config(new_config)
        
        assert moderator_agent.policy_config.strictness_level == "strict"
        assert moderator_agent.policy_config.toxicity_threshold == new_config.toxicity_threshold
        
        # Should log policy change
        assert len(moderator_agent._audit_log) > 0
        log_entry = moderator_agent._audit_log[-1]
        assert log_entry["event_type"] == "policy_update"
        assert log_entry["old_strictness"] == old_strictness
        assert log_entry["new_strictness"] == "strict"
    
    def test_subscribed_events(self, moderator_agent):
        """Test subscribed event types."""
        events = moderator_agent._get_subscribed_events()
        
        expected_events = [
            "content_created",
            "content_flagged",
            "moderation_appeal"
        ]
        
        for event_type in expected_events:
            assert event_type in events
    
    def test_tick_interval(self, moderator_agent):
        """Test tick interval."""
        interval = moderator_agent._get_tick_interval()
        assert isinstance(interval, float)
        assert interval > 0
    
    @pytest.mark.asyncio
    async def test_statistics_update(self, moderator_agent):
        """Test statistics updates."""
        # Simulate some moderation activity
        moderator_agent.stats["content_analyzed"] = 10
        moderator_agent.stats["actions_taken"]["warnings"] = 2
        moderator_agent.stats["actions_taken"]["removals"] = 1
        
        moderator_agent._update_statistics()
        
        # Should calculate action rate
        assert "action_rate" in moderator_agent.stats
        assert moderator_agent.stats["action_rate"] == 0.3  # 3 actions / 10 analyzed
    
    @pytest.mark.asyncio
    async def test_batch_processing_limit(self, moderator_agent):
        """Test batch processing limits."""
        # Add more than batch size to queue
        for i in range(10):
            content_data = {
                "content_id": f"content-{i}",
                "text_content": f"Content {i}",
                "created_by": "user-1"
            }
            moderator_agent._content_queue.append(content_data)
        
        # Mock analysis to avoid actual processing
        with patch.object(moderator_agent, '_analyze_content', new_callable=AsyncMock) as mock_analyze:
            await moderator_agent._process_content_queue()
            
            # Should process at most 5 items (batch size)
            assert mock_analyze.call_count <= 5
            assert len(moderator_agent._content_queue) >= 5
    
    @pytest.mark.asyncio
    async def test_error_handling_in_analysis(self, moderator_agent):
        """Test error handling during content analysis."""
        content_data = {
            "content_id": "error-content",
            "text_content": "This will cause an error",
            "created_by": "user-1"
        }
        
        # Mock analysis to raise an error
        with patch.object(moderator_agent, '_perform_content_analysis', side_effect=Exception("Test error")):
            # Should not raise exception but should handle it gracefully
            try:
                await moderator_agent._analyze_content(content_data)
            except Exception:
                pytest.fail("Exception should have been caught and handled")
            
            # Statistics should not be updated due to error
            assert moderator_agent.stats["content_analyzed"] == 0
    
    @pytest.mark.asyncio
    async def test_empty_content_handling(self, moderator_agent):
        """Test handling of empty or invalid content."""
        # Test with empty content
        empty_content = {
            "content_id": "empty-content",
            "text_content": "",
            "created_by": "user-1"
        }
        
        # Should handle gracefully
        await moderator_agent._analyze_content(empty_content)
        
        # Test with missing content_id
        invalid_content = {
            "text_content": "Some content",
            "created_by": "user-1"
        }
        
        # Should handle gracefully
        await moderator_agent._analyze_content(invalid_content)
    
    @pytest.mark.asyncio
    async def test_audit_log_size_limit(self, moderator_agent):
        """Test audit log size limiting."""
        # Add more than the limit (1000) entries
        for i in range(1100):
            moderator_agent._audit_log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "content_id": f"content-{i}",
                "action": "warning"
            })
        
        # Trigger log cleanup by adding another entry
        result = ModerationResult(
            content_id="new-content",
            violations={},
            recommended_action=ModerationAction.NONE,
            confidence=0.5,
            reasoning="Test"
        )
        moderator_agent._log_moderation_action(result, ModerationAction.NONE)
        
        # Should keep only last 1000 entries
        assert len(moderator_agent._audit_log) == 1000