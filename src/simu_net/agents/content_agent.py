"""Content agent implementation with rich metadata generation."""

import asyncio
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog

from .base import SimuNetAgent
from ..data.models import (
    ContentAgent as ContentAgentModel,
    ContentMetadata,
    EngagementMetrics,
    ModerationStatus,
    ContentType,
    ModerationAction
)
from ..events import AgentEvent

# Optional ML imports - graceful degradation if not available
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None


class ContentAgent(SimuNetAgent):
    """Content agent with rich metadata generation and lifecycle management."""
    
    def __init__(
        self,
        content_id: Optional[str] = None,
        text_content: str = "",
        created_by: str = "",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs
    ):
        """Initialize content agent.
        
        Args:
            content_id: Unique content identifier
            text_content: The text content to analyze
            created_by: ID of the user agent who created this content
            embedding_model_name: Name of the sentence transformer model
            **kwargs: Additional agent parameters
        """
        agent_id = content_id or str(uuid.uuid4())
        super().__init__(agent_id=agent_id, **kwargs)
        
        self.content_id = agent_id
        self.text_content = text_content
        self.created_by = created_by
        self.embedding_model_name = embedding_model_name
        
        # Initialize ML models (lazy loading)
        self._embedding_model: Optional[SentenceTransformer] = None
        self._models_loaded = False
        
        # Initialize content data model
        self.content_data = ContentAgentModel(
            content_id=self.content_id,
            text_content=text_content,
            created_by=created_by,
            created_at=datetime.utcnow()
        )
        
        # Processing state
        self._metadata_generated = False
        self._embeddings_generated = False
        self._lifecycle_stage = "created"  # created -> analyzed -> published -> archived
        
        self.logger.info(
            "Content agent initialized",
            content_id=self.content_id,
            created_by=created_by,
            text_length=len(text_content)
        )
    
    async def _on_start(self) -> None:
        """Called when agent starts - generate metadata and embeddings."""
        try:
            # Load ML models if available
            await self._load_models()
            
            # Generate content metadata
            await self._generate_metadata()
            
            # Generate embeddings
            await self._generate_embeddings()
            
            # Publish content creation event
            await self._publish_content_created()
            
            self._lifecycle_stage = "published"
            
        except Exception as e:
            self.logger.error("Error during content agent startup", error=str(e))
            self._lifecycle_stage = "error"
    
    async def _load_models(self) -> None:
        """Load ML models for content analysis."""
        if self._models_loaded:
            return
        
        try:
            if _SENTENCE_TRANSFORMERS_AVAILABLE:
                # Load in a separate thread to avoid blocking
                loop = asyncio.get_event_loop()
                self._embedding_model = await loop.run_in_executor(
                    None, 
                    lambda: SentenceTransformer(self.embedding_model_name)
                )
                self.logger.info("Embedding model loaded", model=self.embedding_model_name)
            else:
                self.logger.warning("Sentence transformers not available, using mock embeddings")
            
            self._models_loaded = True
            
        except Exception as e:
            self.logger.error("Error loading ML models", error=str(e))
            self._models_loaded = False
    
    async def _generate_metadata(self) -> None:
        """Generate rich metadata for the content."""
        if self._metadata_generated:
            return
        
        try:
            # Analyze text content
            metadata = ContentMetadata()
            
            # Basic text analysis
            metadata.word_count = len(self.text_content.split())
            metadata.language = self._detect_language()
            metadata.content_type = self._classify_content_type()
            
            # Topic classification
            metadata.topic_classification = await self._classify_topics()
            
            # Sentiment analysis
            metadata.sentiment_score = await self._analyze_sentiment()
            
            # Misinformation detection
            metadata.misinformation_probability = await self._detect_misinformation()
            
            # Virality potential
            metadata.virality_potential = await self._calculate_virality_potential()
            
            # Update content data
            self.content_data.metadata = metadata
            self._metadata_generated = True
            
            self.logger.info(
                "Content metadata generated",
                word_count=metadata.word_count,
                sentiment=metadata.sentiment_score,
                virality=metadata.virality_potential
            )
            
        except Exception as e:
            self.logger.error("Error generating metadata", error=str(e))
    
    async def _generate_embeddings(self) -> None:
        """Generate embeddings for the content."""
        if self._embeddings_generated or not self.text_content:
            return
        
        try:
            if self._embedding_model and _NUMPY_AVAILABLE:
                # Generate embeddings in executor to avoid blocking
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None,
                    lambda: self._embedding_model.encode([self.text_content])
                )
                
                # Convert to list for JSON serialization
                self.content_data.embeddings = embeddings[0].tolist()
                
                self.logger.debug(
                    "Embeddings generated",
                    embedding_dim=len(self.content_data.embeddings)
                )
            else:
                # Generate mock embeddings if models not available
                self.content_data.embeddings = self._generate_mock_embeddings()
                
                self.logger.debug("Mock embeddings generated")
            
            self._embeddings_generated = True
            
        except Exception as e:
            self.logger.error("Error generating embeddings", error=str(e))
            # Fallback to mock embeddings
            self.content_data.embeddings = self._generate_mock_embeddings()
    
    def _detect_language(self) -> str:
        """Detect content language (simplified implementation).
        
        Returns:
            Language code (default: 'en')
        """
        # Simple heuristic-based language detection
        # In a real implementation, you'd use a proper language detection library
        
        text_lower = self.text_content.lower()
        
        # Check for common non-English patterns
        if any(char in text_lower for char in ['ñ', 'ç', 'ü', 'ö', 'ä']):
            return 'es'  # Spanish/German/other
        elif any(word in text_lower for word in ['le', 'la', 'les', 'des', 'une', 'est']):
            return 'fr'  # French
        elif any(word in text_lower for word in ['der', 'die', 'das', 'und', 'ist', 'ein']):
            return 'de'  # German
        
        return 'en'  # Default to English
    
    def _classify_content_type(self) -> ContentType:
        """Classify the type of content.
        
        Returns:
            ContentType enum value
        """
        text_lower = self.text_content.lower()
        
        # Check for URLs (links)
        if re.search(r'https?://\S+|www\.\S+', self.text_content):
            return ContentType.LINK
        
        # Check for image indicators
        if any(word in text_lower for word in ['photo', 'image', 'picture', 'pic', '📷', '📸']):
            return ContentType.IMAGE
        
        # Check for video indicators
        if any(word in text_lower for word in ['video', 'watch', 'youtube', 'vimeo', '🎥', '📹']):
            return ContentType.VIDEO
        
        return ContentType.TEXT
    
    async def _classify_topics(self) -> Dict[str, float]:
        """Classify content topics using keyword-based approach.
        
        Returns:
            Dictionary of topic scores
        """
        # Simplified topic classification using keywords
        # In a real implementation, you'd use a trained topic classifier
        
        text_lower = self.text_content.lower()
        topics = {}
        
        # Define topic keywords
        topic_keywords = {
            'politics': ['government', 'election', 'vote', 'policy', 'politician', 'democracy', 'republican', 'democrat'],
            'technology': ['tech', 'ai', 'software', 'computer', 'digital', 'internet', 'app', 'coding'],
            'sports': ['game', 'team', 'player', 'score', 'match', 'championship', 'football', 'basketball'],
            'entertainment': ['movie', 'music', 'celebrity', 'show', 'actor', 'singer', 'film', 'concert'],
            'health': ['health', 'medical', 'doctor', 'hospital', 'medicine', 'fitness', 'wellness', 'diet'],
            'business': ['business', 'company', 'market', 'economy', 'finance', 'investment', 'startup', 'entrepreneur'],
            'science': ['research', 'study', 'scientist', 'discovery', 'experiment', 'theory', 'data', 'analysis'],
            'lifestyle': ['food', 'travel', 'fashion', 'home', 'family', 'relationship', 'lifestyle', 'personal']
        }
        
        # Calculate topic scores
        total_words = len(self.text_content.split())
        
        for topic, keywords in topic_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                # Normalize by content length and keyword list size
                score = min(matches / max(total_words * 0.1, 1), 1.0)
                topics[topic] = round(score, 3)
        
        # Ensure at least one topic if content is not empty
        if not topics and total_words > 0:
            topics['general'] = 0.5
        
        return topics
    
    async def _analyze_sentiment(self) -> float:
        """Analyze sentiment of the content.
        
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        # Simplified sentiment analysis using keyword-based approach
        # In a real implementation, you'd use a trained sentiment model
        
        text_lower = self.text_content.lower()
        
        # Define sentiment keywords
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
            'love', 'like', 'happy', 'joy', 'excited', 'thrilled', 'pleased', 'satisfied',
            'beautiful', 'perfect', 'brilliant', 'outstanding', 'superb', '😊', '😍', '🎉', '❤️'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'dislike',
            'angry', 'sad', 'disappointed', 'frustrated', 'annoyed', 'upset', 'worried',
            'wrong', 'failed', 'broken', 'stupid', 'ridiculous', '😠', '😢', '😡', '💔'
        ]
        
        # Count sentiment words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate sentiment score
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.0  # Neutral
        
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        return max(-1.0, min(1.0, sentiment_score))
    
    async def _detect_misinformation(self) -> float:
        """Detect potential misinformation in content.
        
        Returns:
            Misinformation probability between 0 and 1
        """
        # Simplified misinformation detection using heuristics
        # In a real implementation, you'd use a trained misinformation classifier
        
        text_lower = self.text_content.lower()
        
        # Misinformation indicators
        suspicious_phrases = [
            'they don\'t want you to know', 'the truth they\'re hiding',
            'doctors hate this', 'big pharma', 'government conspiracy',
            'wake up', 'do your own research', 'mainstream media lies',
            'secret cure', 'they\'re lying to you', 'cover up'
        ]
        
        # Absolute claims without evidence
        absolute_claims = [
            'always', 'never', 'all', 'none', 'every', 'completely',
            'totally', 'absolutely', 'definitely', 'certainly'
        ]
        
        # Emotional manipulation
        emotional_words = [
            'shocking', 'unbelievable', 'incredible', 'outrageous',
            'scandalous', 'terrifying', 'devastating', 'explosive'
        ]
        
        # Calculate misinformation score
        suspicious_count = sum(1 for phrase in suspicious_phrases if phrase in text_lower)
        absolute_count = sum(1 for word in absolute_claims if word in text_lower)
        emotional_count = sum(1 for word in emotional_words if word in text_lower)
        
        total_words = len(self.text_content.split())
        
        if total_words == 0:
            return 0.0
        
        # Weight different indicators
        score = (
            suspicious_count * 0.4 +
            absolute_count * 0.1 +
            emotional_count * 0.2
        ) / max(total_words * 0.1, 1)
        
        return min(score, 1.0)
    
    async def _calculate_virality_potential(self) -> float:
        """Calculate the virality potential of content.
        
        Returns:
            Virality potential score between 0 and 1
        """
        # Factors that contribute to virality
        text_lower = self.text_content.lower()
        
        # Emotional content tends to be more viral
        emotional_intensity = abs(self.content_data.metadata.sentiment_score)
        
        # Certain topics are more viral
        viral_topics = ['politics', 'entertainment', 'technology', 'sports']
        topic_virality = sum(
            self.content_data.metadata.topic_classification.get(topic, 0)
            for topic in viral_topics
        )
        
        # Visual content indicators
        visual_indicators = ['photo', 'image', 'video', 'watch', 'see', 'look']
        visual_score = sum(1 for word in visual_indicators if word in text_lower)
        visual_factor = min(visual_score / max(len(self.text_content.split()) * 0.1, 1), 0.3)
        
        # Call-to-action indicators
        cta_words = ['share', 'retweet', 'like', 'comment', 'tag', 'follow', 'subscribe']
        cta_score = sum(1 for word in cta_words if word in text_lower)
        cta_factor = min(cta_score / max(len(self.text_content.split()) * 0.1, 1), 0.2)
        
        # Trending/timely content
        trending_words = ['breaking', 'urgent', 'now', 'today', 'just', 'happening']
        trending_score = sum(1 for word in trending_words if word in text_lower)
        trending_factor = min(trending_score / max(len(self.text_content.split()) * 0.1, 1), 0.2)
        
        # Combine factors
        virality_score = (
            emotional_intensity * 0.3 +
            topic_virality * 0.3 +
            visual_factor +
            cta_factor +
            trending_factor
        )
        
        return min(virality_score, 1.0)
    
    def _generate_mock_embeddings(self, dim: int = 384) -> List[float]:
        """Generate mock embeddings when ML models are not available.
        
        Args:
            dim: Embedding dimension
            
        Returns:
            List of mock embedding values
        """
        # Generate deterministic mock embeddings based on content
        import hashlib
        
        # Use content hash to generate consistent embeddings
        content_hash = hashlib.md5(self.text_content.encode()).hexdigest()
        
        # Convert hash to numbers
        embeddings = []
        for i in range(0, min(len(content_hash), dim * 2), 2):
            hex_pair = content_hash[i:i+2]
            value = int(hex_pair, 16) / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
            embeddings.append(value)
        
        # Pad or truncate to desired dimension
        while len(embeddings) < dim:
            embeddings.append(0.0)
        
        return embeddings[:dim]
    
    async def _publish_content_created(self) -> None:
        """Publish content creation event."""
        try:
            content_data = {
                "content_id": self.content_id,
                "text_content": self.text_content,
                "created_by": self.created_by,
                "created_at": self.content_data.created_at.isoformat(),
                "metadata": self.content_data.metadata.model_dump(),
                "embeddings_available": self._embeddings_generated,
                "word_count": self.content_data.metadata.word_count,
                "sentiment_score": self.content_data.metadata.sentiment_score,
                "virality_potential": self.content_data.metadata.virality_potential,
                "topic_classification": self.content_data.metadata.topic_classification
            }
            
            await self.publish_event(
                event_type="content_created",
                payload=content_data
            )
            
            self.logger.info("Content creation event published")
            
        except Exception as e:
            self.logger.error("Error publishing content creation event", error=str(e))
    
    async def _process_tick(self) -> None:
        """Process agent tick - handle engagement updates and lifecycle."""
        try:
            # Update engagement metrics if we've received interactions
            await self._update_engagement_metrics()
            
            # Check for virality status changes
            self._check_virality_status()
            
            # Handle lifecycle transitions
            await self._handle_lifecycle()
            
        except Exception as e:
            self.logger.error("Error in content agent tick", error=str(e))
    
    async def _update_engagement_metrics(self) -> None:
        """Update engagement metrics based on received interactions."""
        # This would typically query a database for recent interactions
        # For now, we'll simulate some engagement growth
        pass
    
    def _check_virality_status(self) -> None:
        """Check and update virality status."""
        if not self.content_data.is_viral:
            # Check if content has become viral
            engagement_rate = self.content_data.engagement_metrics.engagement_rate
            virality_potential = self.content_data.metadata.virality_potential
            
            if (engagement_rate >= self.content_data.viral_threshold or
                virality_potential >= self.content_data.viral_threshold):
                
                self.content_data.is_viral = True
                self.logger.info("Content went viral!", content_id=self.content_id)
                
                # Publish viral event
                asyncio.create_task(self._publish_viral_event())
    
    async def _publish_viral_event(self) -> None:
        """Publish event when content goes viral."""
        try:
            viral_data = {
                "content_id": self.content_id,
                "created_by": self.created_by,
                "engagement_rate": self.content_data.engagement_metrics.engagement_rate,
                "virality_potential": self.content_data.metadata.virality_potential,
                "viral_at": datetime.utcnow().isoformat()
            }
            
            await self.publish_event(
                event_type="content_viral",
                payload=viral_data
            )
            
        except Exception as e:
            self.logger.error("Error publishing viral event", error=str(e))
    
    async def _handle_lifecycle(self) -> None:
        """Handle content lifecycle transitions."""
        # Content lifecycle: created -> analyzed -> published -> viral? -> archived
        
        if self._lifecycle_stage == "published":
            # Check if content should be archived (simplified logic)
            age_hours = (datetime.utcnow() - self.content_data.created_at).total_seconds() / 3600
            
            # Archive content after 24 hours if not viral
            if age_hours > 24 and not self.content_data.is_viral:
                self._lifecycle_stage = "archived"
                self.logger.info("Content archived", content_id=self.content_id)
                
                # Stop the agent
                await self.stop()
    
    async def _on_event_received(self, event: AgentEvent) -> None:
        """Handle received events.
        
        Args:
            event: The received event
        """
        try:
            if event.event_type == "content_interaction":
                await self._handle_interaction_event(event)
            elif event.event_type == "moderation_action":
                await self._handle_moderation_event(event)
                
        except Exception as e:
            self.logger.error("Error handling event", event_type=event.event_type, error=str(e))
    
    async def _handle_interaction_event(self, event: AgentEvent) -> None:
        """Handle content interaction events.
        
        Args:
            event: Interaction event
        """
        interaction_data = event.payload
        content_id = interaction_data.get("content_id")
        
        # Only handle interactions with this content
        if content_id != self.content_id:
            return
        
        interaction_type = interaction_data.get("interaction_type", "")
        
        # Update engagement metrics
        if interaction_type == "like":
            self.content_data.engagement_metrics.likes += 1
        elif interaction_type == "share":
            self.content_data.engagement_metrics.shares += 1
        elif interaction_type == "comment":
            self.content_data.engagement_metrics.comments += 1
        elif interaction_type == "view":
            self.content_data.engagement_metrics.views += 1
        
        # Recalculate engagement rate
        self.content_data.engagement_metrics.calculate_engagement_rate()
        
        # Update timestamp
        self.content_data.updated_at = datetime.utcnow()
        
        self.logger.debug(
            "Interaction processed",
            interaction_type=interaction_type,
            new_engagement_rate=self.content_data.engagement_metrics.engagement_rate
        )
    
    async def _handle_moderation_event(self, event: AgentEvent) -> None:
        """Handle moderation action events.
        
        Args:
            event: Moderation event
        """
        moderation_data = event.payload
        content_id = moderation_data.get("content_id")
        
        # Only handle moderation for this content
        if content_id != self.content_id:
            return
        
        # Update moderation status
        action = moderation_data.get("action", ModerationAction.NONE.value)
        self.content_data.moderation_status.action_taken = ModerationAction(action)
        self.content_data.moderation_status.is_flagged = action != ModerationAction.NONE.value
        self.content_data.moderation_status.reviewed_by = moderation_data.get("moderator_id")
        self.content_data.moderation_status.reviewed_at = datetime.utcnow()
        
        # Handle content removal
        if action == ModerationAction.REMOVE.value:
            self.content_data.is_active = False
            self._lifecycle_stage = "removed"
            
            self.logger.info("Content removed by moderation", content_id=self.content_id)
            
            # Stop the agent
            await self.stop()
    
    def _get_tick_interval(self) -> float:
        """Get agent tick interval in seconds."""
        # Content agents tick less frequently than user agents
        return 10.0
    
    def _get_subscribed_events(self) -> List[str]:
        """Get list of event types this agent subscribes to."""
        return [
            "content_interaction",
            "moderation_action",
            "content_flagged"
        ]
    
    def get_content_stats(self) -> Dict[str, Any]:
        """Get content statistics and current state.
        
        Returns:
            Dictionary with content statistics
        """
        return {
            "content_id": self.content_id,
            "created_by": self.created_by,
            "text_content": self.text_content,
            "word_count": self.content_data.metadata.word_count,
            "language": self.content_data.metadata.language,
            "content_type": self.content_data.metadata.content_type.value,
            "topic_classification": self.content_data.metadata.topic_classification,
            "sentiment_score": self.content_data.metadata.sentiment_score,
            "misinformation_probability": self.content_data.metadata.misinformation_probability,
            "virality_potential": self.content_data.metadata.virality_potential,
            "engagement_metrics": self.content_data.engagement_metrics.model_dump(),
            "is_viral": self.content_data.is_viral,
            "is_active": self.content_data.is_active,
            "lifecycle_stage": self._lifecycle_stage,
            "embeddings_available": self._embeddings_generated,
            "metadata_generated": self._metadata_generated,
            "created_at": self.content_data.created_at.isoformat(),
            "updated_at": self.content_data.updated_at.isoformat() if self.content_data.updated_at else None
        }
    
    def get_embeddings(self) -> Optional[List[float]]:
        """Get content embeddings.
        
        Returns:
            List of embedding values or None if not generated
        """
        return self.content_data.embeddings
    
    def calculate_similarity(self, other_embeddings: List[float]) -> float:
        """Calculate similarity with another content's embeddings.
        
        Args:
            other_embeddings: Other content's embeddings
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if not self.content_data.embeddings or not other_embeddings:
            return 0.0
        
        try:
            if _NUMPY_AVAILABLE:
                # Use numpy for efficient calculation
                a = np.array(self.content_data.embeddings)
                b = np.array(other_embeddings)
                
                # Cosine similarity
                dot_product = np.dot(a, b)
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                
                similarity = dot_product / (norm_a * norm_b)
                return max(0.0, min(1.0, (similarity + 1) / 2))  # Normalize to [0, 1]
            else:
                # Fallback implementation without numpy
                dot_product = sum(a * b for a, b in zip(self.content_data.embeddings, other_embeddings))
                norm_a = sum(a * a for a in self.content_data.embeddings) ** 0.5
                norm_b = sum(b * b for b in other_embeddings) ** 0.5
                
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                
                similarity = dot_product / (norm_a * norm_b)
                return max(0.0, min(1.0, (similarity + 1) / 2))
                
        except Exception as e:
            self.logger.error("Error calculating similarity", error=str(e))
            return 0.0