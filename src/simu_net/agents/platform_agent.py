"""Platform agent for algorithmic content distribution and feed management."""

import asyncio
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import structlog

from .base import SimuNetAgent
from ..data.models import (
    ContentAgent as ContentAgentModel,
    UserAgent as UserAgentModel,
    PersonaType,
    EngagementMetrics,
    InteractionEvent
)
from ..events import AgentEvent
from ..vector.similarity_search import SimilaritySearchEngine, SearchMode


class FeedRankingMode(str, Enum):
    """Feed ranking modes."""
    ENGAGEMENT_FOCUSED = "engagement_focused"
    SAFETY_FOCUSED = "safety_focused"
    BALANCED = "balanced"
    CHRONOLOGICAL = "chronological"
    PERSONALIZED = "personalized"


class ExperimentStatus(str, Enum):
    """A/B experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"


class ABTestExperiment:
    """A/B test experiment configuration."""
    
    def __init__(
        self,
        experiment_id: str,
        name: str,
        description: str,
        control_config: Dict[str, Any],
        treatment_config: Dict[str, Any],
        traffic_split: float = 0.5,
        duration_hours: int = 168  # 1 week
    ):
        self.experiment_id = experiment_id
        self.name = name
        self.description = description
        self.control_config = control_config
        self.treatment_config = treatment_config
        self.traffic_split = traffic_split
        self.duration_hours = duration_hours
        
        self.status = ExperimentStatus.DRAFT
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Metrics tracking
        self.control_metrics: Dict[str, float] = {}
        self.treatment_metrics: Dict[str, float] = {}
        self.user_assignments: Dict[str, str] = {}  # user_id -> group


class PlatformAgent(SimuNetAgent):
    """Platform agent for algorithmic content distribution and feed management."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        similarity_engine: Optional[SimilaritySearchEngine] = None,
        **kwargs
    ):
        """Initialize platform agent.
        
        Args:
            agent_id: Unique agent identifier
            similarity_engine: Similarity search engine for recommendations
            **kwargs: Additional agent parameters
        """
        super().__init__(agent_id=agent_id, **kwargs)
        
        self.similarity_engine = similarity_engine
        
        # Feed ranking configuration
        self.ranking_mode = FeedRankingMode.BALANCED
        self.engagement_weight = 0.6
        self.safety_weight = 0.4
        self.personalization_strength = 0.5
        
        # Content tracking
        self.content_registry: Dict[str, ContentAgentModel] = {}
        self.user_registry: Dict[str, UserAgentModel] = {}
        self.engagement_history: Dict[str, List[InteractionEvent]] = {}
        
        # Feed management
        self.user_feeds: Dict[str, List[Tuple[str, float]]] = {}  # user_id -> [(content_id, score)]
        self.viral_content: List[str] = []
        self.trending_content: List[str] = []
        
        # A/B testing framework
        self.active_experiments: Dict[str, ABTestExperiment] = {}
        self.experiment_history: List[ABTestExperiment] = []
        
        # Performance metrics
        self.metrics = {
            "total_content_processed": 0,
            "total_recommendations_served": 0,
            "average_engagement_rate": 0.0,
            "viral_content_detected": 0,
            "experiments_completed": 0
        }
        
        self.logger.info("Platform agent initialized")
    
    async def _process_tick(self) -> None:
        """Process a single agent tick."""
        try:
            # Update feed rankings for all users
            await self._update_all_feeds()
            
            # Detect viral content
            await self._detect_viral_content()
            
            # Update trending content
            await self._update_trending_content()
            
            # Process A/B experiments
            await self._process_experiments()
            
            # Update metrics
            await self._update_metrics()
            
        except Exception as e:
            self.logger.error("Error in platform agent tick", error=str(e))
    
    async def _update_all_feeds(self) -> None:
        """Update feed rankings for all registered users."""
        for user_id in self.user_registry.keys():
            try:
                await self._update_user_feed(user_id)
            except Exception as e:
                self.logger.error("Error updating user feed", user_id=user_id, error=str(e))
    
    async def _update_user_feed(self, user_id: str) -> None:
        """Update feed ranking for a specific user.
        
        Args:
            user_id: User ID to update feed for
        """
        user = self.user_registry.get(user_id)
        if not user:
            return
        
        # Get available content (excluding user's own content)
        available_content = [
            content for content_id, content in self.content_registry.items()
            if content.created_by != user_id and content.is_active
        ]
        
        if not available_content:
            return
        
        # Check if user is in an A/B experiment
        experiment_config = await self._get_user_experiment_config(user_id)
        
        # Rank content based on current mode and experiment config
        ranked_content = await self._rank_content_for_user(
            user,
            available_content,
            experiment_config
        )
        
        # Update user's feed
        self.user_feeds[user_id] = ranked_content[:50]  # Top 50 items
        
        self.logger.debug(
            "Updated user feed",
            user_id=user_id,
            feed_size=len(self.user_feeds[user_id])
        )
    
    async def _rank_content_for_user(
        self,
        user: UserAgentModel,
        content_list: List[ContentAgentModel],
        experiment_config: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """Rank content for a specific user.
        
        Args:
            user: User model
            content_list: List of available content
            experiment_config: A/B test configuration if applicable
            
        Returns:
            List of (content_id, score) tuples sorted by score
        """
        ranked_content = []
        
        # Use experiment config if available
        ranking_mode = FeedRankingMode(experiment_config.get("ranking_mode", self.ranking_mode))
        engagement_weight = experiment_config.get("engagement_weight", self.engagement_weight)
        safety_weight = experiment_config.get("safety_weight", self.safety_weight)
        
        for content in content_list:
            score = await self._calculate_content_score(
                user,
                content,
                ranking_mode,
                engagement_weight,
                safety_weight
            )
            
            if score > 0:
                ranked_content.append((content.content_id, score))
        
        # Sort by score (descending)
        ranked_content.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_content
    
    async def _calculate_content_score(
        self,
        user: UserAgentModel,
        content: ContentAgentModel,
        ranking_mode: FeedRankingMode,
        engagement_weight: float,
        safety_weight: float
    ) -> float:
        """Calculate content score for a user.
        
        Args:
            user: User model
            content: Content model
            ranking_mode: Ranking mode to use
            engagement_weight: Weight for engagement factors
            safety_weight: Weight for safety factors
            
        Returns:
            Content score (0.0 to 1.0)
        """
        if ranking_mode == FeedRankingMode.CHRONOLOGICAL:
            # Simple time-based scoring
            age_hours = (datetime.utcnow() - content.created_at).total_seconds() / 3600
            return max(0.0, 1.0 - (age_hours / 168))  # Decay over 1 week
        
        # Base engagement score
        engagement_score = self._calculate_engagement_score(content)
        
        # Safety score (inverse of risk)
        safety_score = self._calculate_safety_score(content)
        
        # Personalization score
        personalization_score = await self._calculate_personalization_score(user, content)
        
        # Viral boost
        viral_boost = 1.2 if content.content_id in self.viral_content else 1.0
        
        # Trending boost
        trending_boost = 1.1 if content.content_id in self.trending_content else 1.0
        
        if ranking_mode == FeedRankingMode.ENGAGEMENT_FOCUSED:
            base_score = engagement_score * 0.8 + safety_score * 0.2
        elif ranking_mode == FeedRankingMode.SAFETY_FOCUSED:
            base_score = engagement_score * 0.2 + safety_score * 0.8
        elif ranking_mode == FeedRankingMode.PERSONALIZED:
            base_score = (
                engagement_score * 0.4 +
                safety_score * 0.3 +
                personalization_score * 0.3
            )
        else:  # BALANCED
            base_score = (
                engagement_score * engagement_weight +
                safety_score * safety_weight
            )
        
        # Apply boosts
        final_score = base_score * viral_boost * trending_boost
        
        # Add small random factor to prevent identical rankings
        final_score += random.uniform(-0.01, 0.01)
        
        return max(0.0, min(1.0, final_score))
    
    def _calculate_engagement_score(self, content: ContentAgentModel) -> float:
        """Calculate engagement score for content.
        
        Args:
            content: Content model
            
        Returns:
            Engagement score (0.0 to 1.0)
        """
        metrics = content.engagement_metrics
        
        # Normalize metrics (simplified approach)
        like_score = min(1.0, metrics.likes / 100.0)
        share_score = min(1.0, metrics.shares / 20.0)
        comment_score = min(1.0, metrics.comments / 10.0)
        
        # Weighted combination
        engagement_score = (
            like_score * 0.4 +
            share_score * 0.4 +
            comment_score * 0.2
        )
        
        # Boost based on virality potential
        virality_boost = 1.0 + (content.metadata.virality_potential * 0.3)
        
        return min(1.0, engagement_score * virality_boost)
    
    def _calculate_safety_score(self, content: ContentAgentModel) -> float:
        """Calculate safety score for content.
        
        Args:
            content: Content model
            
        Returns:
            Safety score (0.0 to 1.0)
        """
        moderation = content.moderation_status
        
        # Base safety score (inverse of risk)
        safety_score = 1.0
        
        # Reduce score for flagged content
        if moderation.is_flagged:
            safety_score *= 0.3
        
        # Reduce score based on misinformation probability
        misinformation_penalty = content.metadata.misinformation_probability
        safety_score *= (1.0 - misinformation_penalty * 0.7)
        
        # Reduce score based on violation confidence
        if moderation.confidence_scores:
            max_confidence = max(moderation.confidence_scores.values())
            safety_score *= (1.0 - max_confidence * 0.5)
        
        return max(0.0, safety_score)
    
    async def _calculate_personalization_score(
        self,
        user: UserAgentModel,
        content: ContentAgentModel
    ) -> float:
        """Calculate personalization score for user-content pair.
        
        Args:
            user: User model
            content: Content model
            
        Returns:
            Personalization score (0.0 to 1.0)
        """
        if not self.similarity_engine:
            return 0.5  # Neutral score if no similarity engine
        
        # Get user's engagement history
        user_history = self.engagement_history.get(user.agent_id, [])
        if not user_history:
            return 0.5  # Neutral score for new users
        
        # Get content IDs user has engaged with
        engaged_content_ids = [event.content_id for event in user_history[-20:]]  # Last 20 interactions
        
        try:
            # Get similarity-based recommendations
            recommendations = await self.similarity_engine.get_content_recommendations(
                user_id=user.agent_id,
                user_history=engaged_content_ids,
                persona_type=user.persona_type,
                k=100
            )
            
            # Check if current content is in recommendations
            for content_id, score, _ in recommendations:
                if content_id == content.content_id:
                    return score
            
            # If not in recommendations, calculate direct similarity
            if engaged_content_ids and content.embeddings:
                # Use most recent engagement for similarity
                recent_content_id = engaged_content_ids[-1]
                similarity_results = await self.similarity_engine.search_content(
                    query=recent_content_id,
                    mode=SearchMode.SIMILARITY,
                    k=50
                )
                
                for content_id, score, _ in similarity_results:
                    if content_id == content.content_id:
                        return score
            
        except Exception as e:
            self.logger.error("Error calculating personalization score", error=str(e))
        
        return 0.3  # Low default score
    
    async def _detect_viral_content(self) -> None:
        """Detect and track viral content."""
        new_viral_content = []
        
        for content_id, content in self.content_registry.items():
            if content_id not in self.viral_content:
                # Check if content has gone viral
                if self._is_content_viral(content):
                    new_viral_content.append(content_id)
                    self.viral_content.append(content_id)
                    
                    # Publish viral content event
                    await self.publish_event(
                        event_type="content_viral",
                        payload={
                            "content_id": content_id,
                            "engagement_rate": content.engagement_metrics.engagement_rate,
                            "virality_score": content.metadata.virality_potential,
                            "detected_at": datetime.utcnow().isoformat()
                        }
                    )
        
        if new_viral_content:
            self.metrics["viral_content_detected"] += len(new_viral_content)
            self.logger.info("Viral content detected", count=len(new_viral_content))
    
    def _is_content_viral(self, content: ContentAgentModel) -> bool:
        """Check if content meets viral criteria.
        
        Args:
            content: Content model
            
        Returns:
            True if content is viral
        """
        metrics = content.engagement_metrics
        
        # Viral criteria
        high_engagement = metrics.engagement_rate > 0.1
        high_shares = metrics.shares > 10
        high_reach = metrics.reach > 100
        high_virality_potential = content.metadata.virality_potential > 0.7
        
        # Content must meet at least 2 criteria
        criteria_met = sum([high_engagement, high_shares, high_reach, high_virality_potential])
        
        return criteria_met >= 2
    
    async def _update_trending_content(self) -> None:
        """Update trending content list."""
        # Calculate trending scores for all content
        trending_scores = []
        
        for content_id, content in self.content_registry.items():
            if content.is_active:
                trending_score = self._calculate_trending_score(content)
                trending_scores.append((content_id, trending_score))
        
        # Sort by trending score and take top 20
        trending_scores.sort(key=lambda x: x[1], reverse=True)
        self.trending_content = [content_id for content_id, _ in trending_scores[:20]]
        
        self.logger.debug("Updated trending content", count=len(self.trending_content))
    
    def _calculate_trending_score(self, content: ContentAgentModel) -> float:
        """Calculate trending score for content.
        
        Args:
            content: Content model
            
        Returns:
            Trending score
        """
        # Time decay factor
        age_hours = (datetime.utcnow() - content.created_at).total_seconds() / 3600
        time_decay = max(0.1, 1.0 - (age_hours / 24))  # Decay over 24 hours
        
        # Engagement velocity (engagement per hour)
        engagement_velocity = 0.0
        if age_hours > 0:
            total_engagement = (
                content.engagement_metrics.likes +
                content.engagement_metrics.shares * 2 +
                content.engagement_metrics.comments * 1.5
            )
            engagement_velocity = total_engagement / age_hours
        
        # Trending score
        trending_score = engagement_velocity * time_decay
        
        # Boost for viral content
        if content.content_id in self.viral_content:
            trending_score *= 1.5
        
        return trending_score
    
    async def _process_experiments(self) -> None:
        """Process active A/B experiments."""
        current_time = datetime.utcnow()
        
        for experiment_id, experiment in list(self.active_experiments.items()):
            try:
                if experiment.status == ExperimentStatus.RUNNING:
                    # Check if experiment should end
                    if (experiment.end_time and current_time >= experiment.end_time):
                        await self._complete_experiment(experiment_id)
                    else:
                        # Update experiment metrics
                        await self._update_experiment_metrics(experiment)
                
            except Exception as e:
                self.logger.error("Error processing experiment", experiment_id=experiment_id, error=str(e))
    
    async def _update_experiment_metrics(self, experiment: ABTestExperiment) -> None:
        """Update metrics for an A/B experiment.
        
        Args:
            experiment: Experiment to update
        """
        # Calculate metrics for control and treatment groups
        control_users = [
            user_id for user_id, group in experiment.user_assignments.items()
            if group == "control"
        ]
        treatment_users = [
            user_id for user_id, group in experiment.user_assignments.items()
            if group == "treatment"
        ]
        
        # Calculate engagement rates for each group
        if control_users:
            control_engagement = await self._calculate_group_engagement(control_users)
            experiment.control_metrics["engagement_rate"] = control_engagement
        
        if treatment_users:
            treatment_engagement = await self._calculate_group_engagement(treatment_users)
            experiment.treatment_metrics["engagement_rate"] = treatment_engagement
    
    async def _calculate_group_engagement(self, user_ids: List[str]) -> float:
        """Calculate average engagement rate for a group of users.
        
        Args:
            user_ids: List of user IDs
            
        Returns:
            Average engagement rate
        """
        if not user_ids:
            return 0.0
        
        total_engagement = 0.0
        total_users = 0
        
        for user_id in user_ids:
            user_history = self.engagement_history.get(user_id, [])
            if user_history:
                # Calculate engagement rate for last 24 hours
                recent_history = [
                    event for event in user_history
                    if (datetime.utcnow() - event.timestamp).total_seconds() < 86400
                ]
                
                if recent_history:
                    engagement_rate = len(recent_history) / 24.0  # Interactions per hour
                    total_engagement += engagement_rate
                    total_users += 1
        
        return total_engagement / max(1, total_users)
    
    async def _complete_experiment(self, experiment_id: str) -> None:
        """Complete an A/B experiment.
        
        Args:
            experiment_id: Experiment ID to complete
        """
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            return
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_time = datetime.utcnow()
        
        # Move to history
        self.experiment_history.append(experiment)
        del self.active_experiments[experiment_id]
        
        # Publish experiment completion event
        await self.publish_event(
            event_type="experiment_completed",
            payload={
                "experiment_id": experiment_id,
                "name": experiment.name,
                "control_metrics": experiment.control_metrics,
                "treatment_metrics": experiment.treatment_metrics,
                "duration_hours": (experiment.end_time - experiment.start_time).total_seconds() / 3600,
                "completed_at": experiment.end_time.isoformat()
            }
        )
        
        self.metrics["experiments_completed"] += 1
        self.logger.info("Experiment completed", experiment_id=experiment_id)
    
    async def _update_metrics(self) -> None:
        """Update platform metrics."""
        # Calculate average engagement rate
        if self.content_registry:
            total_engagement = sum(
                content.engagement_metrics.engagement_rate
                for content in self.content_registry.values()
            )
            self.metrics["average_engagement_rate"] = total_engagement / len(self.content_registry)
        
        # Update content processed count
        self.metrics["total_content_processed"] = len(self.content_registry)
        
        # Update recommendations served (estimated)
        self.metrics["total_recommendations_served"] = len(self.user_feeds) * 10
    
    async def _get_user_experiment_config(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment configuration for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Experiment configuration or None
        """
        for experiment in self.active_experiments.values():
            if experiment.status == ExperimentStatus.RUNNING:
                if user_id not in experiment.user_assignments:
                    # Assign user to experiment group
                    group = "treatment" if random.random() < experiment.traffic_split else "control"
                    experiment.user_assignments[user_id] = group
                
                group = experiment.user_assignments[user_id]
                if group == "treatment":
                    return experiment.treatment_config
                else:
                    return experiment.control_config
        
        return None
    
    async def _on_event_received(self, event: AgentEvent) -> None:
        """Handle received events.
        
        Args:
            event: The received event
        """
        try:
            if event.event_type == "content_created":
                await self._handle_content_created(event)
            elif event.event_type == "content_interaction":
                await self._handle_content_interaction(event)
            elif event.event_type == "user_registered":
                await self._handle_user_registered(event)
            elif event.event_type == "moderation_action":
                await self._handle_moderation_action(event)
            
        except Exception as e:
            self.logger.error("Error handling event", event_type=event.event_type, error=str(e))
    
    async def _handle_content_created(self, event: AgentEvent) -> None:
        """Handle content creation events.
        
        Args:
            event: Content creation event
        """
        content_data = event.payload
        content_id = content_data.get("content_id")
        
        if content_id:
            # Create content model from event data
            content = ContentAgentModel(
                content_id=content_id,
                text_content=content_data.get("text_content", ""),
                created_by=content_data.get("created_by", ""),
                created_at=datetime.fromisoformat(content_data.get("created_at", datetime.utcnow().isoformat()))
            )
            
            # Register content
            self.content_registry[content_id] = content
            
            self.logger.debug("Content registered", content_id=content_id)
    
    async def _handle_content_interaction(self, event: AgentEvent) -> None:
        """Handle content interaction events.
        
        Args:
            event: Content interaction event
        """
        interaction_data = event.payload
        user_id = interaction_data.get("user_id")
        content_id = interaction_data.get("content_id")
        interaction_type = interaction_data.get("interaction_type")
        
        if user_id and content_id:
            # Create interaction event
            interaction = InteractionEvent(
                interaction_id=interaction_data.get("interaction_id", str(uuid.uuid4())),
                user_id=user_id,
                content_id=content_id,
                interaction_type=interaction_type,
                timestamp=datetime.fromisoformat(interaction_data.get("timestamp", datetime.utcnow().isoformat()))
            )
            
            # Add to engagement history
            if user_id not in self.engagement_history:
                self.engagement_history[user_id] = []
            self.engagement_history[user_id].append(interaction)
            
            # Update content engagement metrics
            if content_id in self.content_registry:
                content = self.content_registry[content_id]
                
                if interaction_type == "like":
                    content.add_engagement(likes=1, views=1)
                elif interaction_type == "share":
                    content.add_engagement(shares=1, views=1)
                elif interaction_type == "comment":
                    content.add_engagement(comments=1, views=1)
                else:
                    content.add_engagement(views=1)
            
            self.logger.debug(
                "Interaction processed",
                user_id=user_id,
                content_id=content_id,
                interaction_type=interaction_type
            )
    
    async def _handle_user_registered(self, event: AgentEvent) -> None:
        """Handle user registration events.
        
        Args:
            event: User registration event
        """
        user_data = event.payload
        user_id = user_data.get("agent_id")
        
        if user_id:
            # Create user model from event data
            user = UserAgentModel(
                agent_id=user_id,
                persona_type=PersonaType(user_data.get("persona_type", "casual")),
                created_at=datetime.fromisoformat(user_data.get("created_at", datetime.utcnow().isoformat())),
                last_active=datetime.utcnow()
            )
            
            # Register user
            self.user_registry[user_id] = user
            
            # Initialize empty feed
            self.user_feeds[user_id] = []
            
            self.logger.debug("User registered", user_id=user_id)
    
    async def _handle_moderation_action(self, event: AgentEvent) -> None:
        """Handle moderation action events.
        
        Args:
            event: Moderation action event
        """
        moderation_data = event.payload
        content_id = moderation_data.get("content_id")
        action = moderation_data.get("action")
        
        if content_id in self.content_registry:
            content = self.content_registry[content_id]
            
            # Update content based on moderation action
            if action == "remove":
                content.is_active = False
                # Remove from viral and trending lists
                if content_id in self.viral_content:
                    self.viral_content.remove(content_id)
                if content_id in self.trending_content:
                    self.trending_content.remove(content_id)
            
            self.logger.debug("Moderation action processed", content_id=content_id, action=action)
    
    def _get_tick_interval(self) -> float:
        """Get agent tick interval in seconds."""
        return 10.0  # Platform agent ticks every 10 seconds
    
    def _get_subscribed_events(self) -> List[str]:
        """Get list of event types this agent subscribes to."""
        return [
            "content_created",
            "content_interaction",
            "user_registered",
            "moderation_action"
        ]
    
    # Public API methods
    
    async def get_user_feed(self, user_id: str, limit: int = 20) -> List[Tuple[str, float]]:
        """Get feed for a specific user.
        
        Args:
            user_id: User ID to get feed for
            limit: Maximum number of items to return
            
        Returns:
            List of (content_id, score) tuples
        """
        user_feed = self.user_feeds.get(user_id, [])
        return user_feed[:limit]
    
    async def get_trending_content(self, limit: int = 20) -> List[str]:
        """Get current trending content.
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of trending content IDs
        """
        return self.trending_content[:limit]
    
    async def get_viral_content(self, limit: int = 20) -> List[str]:
        """Get current viral content.
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of viral content IDs
        """
        return self.viral_content[:limit]
    
    async def create_experiment(
        self,
        name: str,
        description: str,
        control_config: Dict[str, Any],
        treatment_config: Dict[str, Any],
        traffic_split: float = 0.5,
        duration_hours: int = 168
    ) -> str:
        """Create a new A/B experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            control_config: Configuration for control group
            treatment_config: Configuration for treatment group
            traffic_split: Fraction of users in treatment group
            duration_hours: Experiment duration in hours
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        
        experiment = ABTestExperiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            control_config=control_config,
            treatment_config=treatment_config,
            traffic_split=traffic_split,
            duration_hours=duration_hours
        )
        
        self.active_experiments[experiment_id] = experiment
        
        self.logger.info("Experiment created", experiment_id=experiment_id, name=name)
        
        return experiment_id
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B experiment.
        
        Args:
            experiment_id: Experiment ID to start
            
        Returns:
            True if started successfully
        """
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            self.logger.error("Experiment not found", experiment_id=experiment_id)
            return False
        
        if experiment.status != ExperimentStatus.DRAFT:
            self.logger.error("Experiment not in draft status", experiment_id=experiment_id)
            return False
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.utcnow()
        experiment.end_time = experiment.start_time + timedelta(hours=experiment.duration_hours)
        
        # Publish experiment start event
        await self.publish_event(
            event_type="experiment_started",
            payload={
                "experiment_id": experiment_id,
                "name": experiment.name,
                "start_time": experiment.start_time.isoformat(),
                "end_time": experiment.end_time.isoformat(),
                "traffic_split": experiment.traffic_split
            }
        )
        
        self.logger.info("Experiment started", experiment_id=experiment_id)
        
        return True
    
    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an A/B experiment early.
        
        Args:
            experiment_id: Experiment ID to stop
            
        Returns:
            True if stopped successfully
        """
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            self.logger.error("Experiment not found", experiment_id=experiment_id)
            return False
        
        if experiment.status != ExperimentStatus.RUNNING:
            self.logger.error("Experiment not running", experiment_id=experiment_id)
            return False
        
        experiment.status = ExperimentStatus.PAUSED
        
        self.logger.info("Experiment stopped", experiment_id=experiment_id)
        
        return True
    
    async def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get results for an A/B experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment results or None if not found
        """
        # Check active experiments
        experiment = self.active_experiments.get(experiment_id)
        
        # Check experiment history if not active
        if not experiment:
            for hist_exp in self.experiment_history:
                if hist_exp.experiment_id == experiment_id:
                    experiment = hist_exp
                    break
        
        if not experiment:
            return None
        
        # Calculate statistical significance if experiment is completed
        statistical_significance = None
        if experiment.status == ExperimentStatus.COMPLETED:
            statistical_significance = await self._calculate_statistical_significance(experiment)
        
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "description": experiment.description,
            "status": experiment.status.value,
            "start_time": experiment.start_time.isoformat() if experiment.start_time else None,
            "end_time": experiment.end_time.isoformat() if experiment.end_time else None,
            "traffic_split": experiment.traffic_split,
            "control_config": experiment.control_config,
            "treatment_config": experiment.treatment_config,
            "control_metrics": experiment.control_metrics,
            "treatment_metrics": experiment.treatment_metrics,
            "user_assignments": len(experiment.user_assignments),
            "statistical_significance": statistical_significance
        }
    
    async def _calculate_statistical_significance(self, experiment: ABTestExperiment) -> Dict[str, Any]:
        """Calculate statistical significance for experiment results.
        
        Args:
            experiment: Experiment to analyze
            
        Returns:
            Statistical significance results
        """
        try:
            # Get engagement rates for both groups
            control_rate = experiment.control_metrics.get("engagement_rate", 0.0)
            treatment_rate = experiment.treatment_metrics.get("engagement_rate", 0.0)
            
            # Count users in each group
            control_users = sum(1 for group in experiment.user_assignments.values() if group == "control")
            treatment_users = sum(1 for group in experiment.user_assignments.values() if group == "treatment")
            
            if control_users == 0 or treatment_users == 0:
                return {"error": "Insufficient data for statistical analysis"}
            
            # Simple statistical test (in practice, would use proper statistical libraries)
            # This is a simplified implementation for demonstration
            pooled_rate = (control_rate * control_users + treatment_rate * treatment_users) / (control_users + treatment_users)
            pooled_variance = pooled_rate * (1 - pooled_rate)
            
            if pooled_variance == 0:
                return {"error": "No variance in data"}
            
            standard_error = (pooled_variance * (1/control_users + 1/treatment_users)) ** 0.5
            
            if standard_error == 0:
                return {"error": "Zero standard error"}
            
            z_score = (treatment_rate - control_rate) / standard_error
            
            # Approximate p-value calculation (simplified)
            import math
            p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2))))
            
            return {
                "control_rate": control_rate,
                "treatment_rate": treatment_rate,
                "difference": treatment_rate - control_rate,
                "relative_change": ((treatment_rate - control_rate) / control_rate * 100) if control_rate > 0 else 0,
                "z_score": z_score,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "control_sample_size": control_users,
                "treatment_sample_size": treatment_users
            }
            
        except Exception as e:
            self.logger.error("Error calculating statistical significance", error=str(e))
            return {"error": f"Statistical calculation failed: {str(e)}"}
    
    async def get_platform_metrics(self) -> Dict[str, Any]:
        """Get current platform metrics.
        
        Returns:
            Dictionary of platform metrics
        """
        return {
            **self.metrics,
            "active_users": len(self.user_registry),
            "total_content": len(self.content_registry),
            "active_content": sum(1 for content in self.content_registry.values() if content.is_active),
            "viral_content_count": len(self.viral_content),
            "trending_content_count": len(self.trending_content),
            "active_experiments": len(self.active_experiments),
            "completed_experiments": len(self.experiment_history),
            "user_feeds_generated": len(self.user_feeds)
        }
    
    async def update_ranking_configuration(
        self,
        ranking_mode: Optional[FeedRankingMode] = None,
        engagement_weight: Optional[float] = None,
        safety_weight: Optional[float] = None,
        personalization_strength: Optional[float] = None
    ) -> None:
        """Update feed ranking configuration.
        
        Args:
            ranking_mode: New ranking mode
            engagement_weight: New engagement weight
            safety_weight: New safety weight
            personalization_strength: New personalization strength
        """
        if ranking_mode:
            self.ranking_mode = ranking_mode
        if engagement_weight is not None:
            self.engagement_weight = engagement_weight
        if safety_weight is not None:
            self.safety_weight = safety_weight
        if personalization_strength is not None:
            self.personalization_strength = personalization_strength
        
        # Ensure weights sum to 1.0
        if engagement_weight is not None or safety_weight is not None:
            total_weight = self.engagement_weight + self.safety_weight
            if total_weight > 0:
                self.engagement_weight /= total_weight
                self.safety_weight /= total_weight
        
        self.logger.info(
            "Ranking configuration updated",
            ranking_mode=self.ranking_mode.value,
            engagement_weight=self.engagement_weight,
            safety_weight=self.safety_weight
        ):
            user_id: User ID
            limit: Maximum number of items to return
            
        Returns:
            List of (content_id, score) tuples
        """
        feed = self.user_feeds.get(user_id, [])
        return feed[:limit]
    
    async def get_content_recommendations(
        self,
        user_id: str,
        k: int = 10,
        mode: SearchMode = SearchMode.PERSONALIZED
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Get content recommendations for a user.
        
        Args:
            user_id: User ID
            k: Number of recommendations
            mode: Search mode to use
            
        Returns:
            List of (content_id, score, metadata) tuples
        """
        if not self.similarity_engine:
            return []
        
        user = self.user_registry.get(user_id)
        if not user:
            return []
        
        # Get user's engagement history
        user_history = self.engagement_history.get(user_id, [])
        engaged_content_ids = [event.content_id for event in user_history[-20:]]
        
        try:
            recommendations = await self.similarity_engine.get_content_recommendations(
                user_id=user_id,
                user_history=engaged_content_ids,
                persona_type=user.persona_type,
                k=k
            )
            
            return recommendations
            
        except Exception as e:
            self.logger.error("Error getting recommendations", user_id=user_id, error=str(e))
            return []
    
    def create_ab_experiment(
        self,
        name: str,
        description: str,
        control_config: Dict[str, Any],
        treatment_config: Dict[str, Any],
        traffic_split: float = 0.5,
        duration_hours: int = 168
    ) -> str:
        """Create a new A/B test experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            control_config: Control group configuration
            treatment_config: Treatment group configuration
            traffic_split: Fraction of users in treatment group
            duration_hours: Experiment duration in hours
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        
        experiment = ABTestExperiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            control_config=control_config,
            treatment_config=treatment_config,
            traffic_split=traffic_split,
            duration_hours=duration_hours
        )
        
        self.active_experiments[experiment_id] = experiment
        
        self.logger.info("A/B experiment created", experiment_id=experiment_id, name=name)
        
        return experiment_id
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if started successfully
        """
        experiment = self.active_experiments.get(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.DRAFT:
            return False
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.utcnow()
        experiment.end_time = experiment.start_time + timedelta(hours=experiment.duration_hours)
        
        # Publish experiment start event
        await self.publish_event(
            event_type="experiment_started",
            payload={
                "experiment_id": experiment_id,
                "name": experiment.name,
                "start_time": experiment.start_time.isoformat(),
                "end_time": experiment.end_time.isoformat()
            }
        )
        
        self.logger.info("A/B experiment started", experiment_id=experiment_id)
        
        return True
    
    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get results for an A/B experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment results or None if not found
        """
        # Check active experiments
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            # Check experiment history
            for hist_exp in self.experiment_history:
                if hist_exp.experiment_id == experiment_id:
                    experiment = hist_exp
                    break
        
        if not experiment:
            return None
        
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "description": experiment.description,
            "status": experiment.status.value,
            "start_time": experiment.start_time.isoformat() if experiment.start_time else None,
            "end_time": experiment.end_time.isoformat() if experiment.end_time else None,
            "control_config": experiment.control_config,
            "treatment_config": experiment.treatment_config,
            "control_metrics": experiment.control_metrics,
            "treatment_metrics": experiment.treatment_metrics,
            "user_assignments": len(experiment.user_assignments),
            "traffic_split": experiment.traffic_split
        }
    
    def get_platform_stats(self) -> Dict[str, Any]:
        """Get platform statistics.
        
        Returns:
            Dictionary with platform statistics
        """
        return {
            "agent_id": self.agent_id,
            "ranking_mode": self.ranking_mode.value,
            "engagement_weight": self.engagement_weight,
            "safety_weight": self.safety_weight,
            "registered_users": len(self.user_registry),
            "registered_content": len(self.content_registry),
            "viral_content_count": len(self.viral_content),
            "trending_content_count": len(self.trending_content),
            "active_experiments": len(self.active_experiments),
            "completed_experiments": len(self.experiment_history),
            "metrics": self.metrics,
            "is_active": self.is_active,
            "is_running": self.is_running
        }
    
    def update_ranking_config(
        self,
        ranking_mode: Optional[FeedRankingMode] = None,
        engagement_weight: Optional[float] = None,
        safety_weight: Optional[float] = None,
        personalization_strength: Optional[float] = None
    ) -> None:
        """Update feed ranking configuration.
        
        Args:
            ranking_mode: New ranking mode
            engagement_weight: New engagement weight
            safety_weight: New safety weight
            personalization_strength: New personalization strength
        """
        if ranking_mode:
            self.ranking_mode = ranking_mode
        if engagement_weight is not None:
            self.engagement_weight = engagement_weight
        if safety_weight is not None:
            self.safety_weight = safety_weight
        if personalization_strength is not None:
            self.personalization_strength = personalization_strength
        
        # Ensure weights sum to 1.0
        if engagement_weight is not None or safety_weight is not None:
            total_weight = self.engagement_weight + self.safety_weight
            if total_weight > 0:
                self.engagement_weight /= total_weight
                self.safety_weight /= total_weight
        
        self.logger.info(
            "Ranking configuration updated",
            ranking_mode=self.ranking_mode.value,
            engagement_weight=self.engagement_weight,
            safety_weight=self.safety_weight
        )