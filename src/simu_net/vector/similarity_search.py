"""Advanced similarity search engine for content discovery."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import structlog

from .vector_store import VectorStore
from ..data.models import ContentAgent as ContentAgentModel, PersonaType


class SearchMode(str, Enum):
    """Search modes for different use cases."""
    SIMILARITY = "similarity"
    TRENDING = "trending"
    DIVERSE = "diverse"
    TEMPORAL = "temporal"
    TOPIC_BASED = "topic_based"


class SimilaritySearchEngine:
    """Advanced similarity search engine for content discovery and recommendations."""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize similarity search engine.
        
        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store
        self.logger = structlog.get_logger().bind(component="SimilaritySearchEngine")
        
        # Search configuration
        self.default_k = 10
        self.default_threshold = 0.3
        self.max_results = 100
    
    async def search_content(
        self,
        query: Union[str, List[float], ContentAgentModel],
        mode: SearchMode = SearchMode.SIMILARITY,
        k: int = None,
        threshold: float = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Advanced content search with multiple modes.
        
        Args:
            query: Search query (content ID, embedding, or content model)
            mode: Search mode to use
            k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Additional filters to apply
            **kwargs: Mode-specific parameters
            
        Returns:
            List of (content_id, score, metadata) tuples
        """
        k = k or self.default_k
        threshold = threshold or self.default_threshold
        
        try:
            # Convert query to embedding if needed
            query_embedding = await self._prepare_query(query)
            if not query_embedding:
                return []
            
            # Execute search based on mode
            if mode == SearchMode.SIMILARITY:
                results = await self._similarity_search(query_embedding, k, threshold, filters)
            elif mode == SearchMode.TRENDING:
                results = await self._trending_search(query_embedding, k, **kwargs)
            elif mode == SearchMode.DIVERSE:
                results = await self._diverse_search(query_embedding, k, threshold, **kwargs)
            elif mode == SearchMode.TEMPORAL:
                results = await self._temporal_search(query_embedding, k, threshold, **kwargs)
            elif mode == SearchMode.TOPIC_BASED:
                results = await self._topic_based_search(query_embedding, k, threshold, **kwargs)
            else:
                self.logger.error("Unknown search mode", mode=mode)
                return []
            
            # Add metadata to results
            enriched_results = await self._enrich_results(results, mode)
            
            return enriched_results
            
        except Exception as e:
            self.logger.error("Error in content search", mode=mode, error=str(e))
            return []
    
    async def _prepare_query(self, query: Union[str, List[float], ContentAgentModel]) -> Optional[List[float]]:
        """Prepare query for search by converting to embedding.
        
        Args:
            query: Query in various formats
            
        Returns:
            Query embedding or None if conversion failed
        """
        try:
            if isinstance(query, list):
                # Already an embedding
                return query
            elif isinstance(query, ContentAgentModel):
                # Extract embedding from content model
                return query.embeddings
            elif isinstance(query, str):
                # Content ID - get embedding from cache or vector store
                embedding = self.vector_store._get_cached_embedding(query)
                if embedding:
                    return embedding
                else:
                    self.logger.warning("No embedding found for content ID", content_id=query)
                    return None
            else:
                self.logger.error("Unsupported query type", query_type=type(query))
                return None
                
        except Exception as e:
            self.logger.error("Error preparing query", error=str(e))
            return None
    
    async def _similarity_search(
        self,
        query_embedding: List[float],
        k: int,
        threshold: float,
        filters: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Basic similarity search.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results
            threshold: Similarity threshold
            filters: Additional filters
            
        Returns:
            List of (content_id, similarity_score) tuples
        """
        return await self.vector_store.find_similar_content(
            query_embedding=query_embedding,
            k=k,
            threshold=threshold,
            filters=filters
        )
    
    async def _trending_search(
        self,
        query_embedding: List[float],
        k: int,
        time_window_hours: int = 24,
        viral_boost: float = 1.5
    ) -> List[Tuple[str, float]]:
        """Search for trending content similar to query.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results
            time_window_hours: Time window for trending calculation
            viral_boost: Boost factor for viral content
            
        Returns:
            List of (content_id, trend_score) tuples
        """
        try:
            # Search in recent content index
            recent_results = await self.vector_store.find_similar_content(
                query_embedding=query_embedding,
                k=k * 2,  # Get more results for filtering
                index_name="content_recent",
                threshold=0.2
            )
            
            # Search in viral content index
            viral_results = await self.vector_store.find_similar_content(
                query_embedding=query_embedding,
                k=k,
                index_name="content_viral",
                threshold=0.2
            )
            
            # Combine and boost viral content
            combined_results = {}
            
            # Add recent results
            for content_id, score in recent_results:
                combined_results[content_id] = score
            
            # Add viral results with boost
            for content_id, score in viral_results:
                boosted_score = score * viral_boost
                if content_id in combined_results:
                    combined_results[content_id] = max(combined_results[content_id], boosted_score)
                else:
                    combined_results[content_id] = boosted_score
            
            # Sort by score and return top k
            sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
            return sorted_results[:k]
            
        except Exception as e:
            self.logger.error("Error in trending search", error=str(e))
            return []
    
    async def _diverse_search(
        self,
        query_embedding: List[float],
        k: int,
        threshold: float,
        diversity_factor: float = 0.4
    ) -> List[Tuple[str, float]]:
        """Search with diversity promotion.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results
            threshold: Similarity threshold
            diversity_factor: Factor to promote diversity
            
        Returns:
            List of (content_id, diversity_score) tuples
        """
        try:
            # Get more results than needed for diversity filtering
            initial_results = await self.vector_store.find_similar_content(
                query_embedding=query_embedding,
                k=k * 3,
                threshold=threshold
            )
            
            if not initial_results:
                return []
            
            # Apply diversity filtering
            diverse_results = await self._apply_diversity_selection(
                initial_results,
                k,
                diversity_factor
            )
            
            return diverse_results
            
        except Exception as e:
            self.logger.error("Error in diverse search", error=str(e))
            return []
    
    async def _temporal_search(
        self,
        query_embedding: List[float],
        k: int,
        threshold: float,
        time_decay_factor: float = 0.1,
        max_age_hours: int = 168  # 1 week
    ) -> List[Tuple[str, float]]:
        """Search with temporal relevance weighting.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results
            threshold: Similarity threshold
            time_decay_factor: Factor for time-based score decay
            max_age_hours: Maximum age of content to consider
            
        Returns:
            List of (content_id, temporal_score) tuples
        """
        try:
            # Search in recent content first
            recent_results = await self.vector_store.find_similar_content(
                query_embedding=query_embedding,
                k=k * 2,
                index_name="content_recent",
                threshold=threshold
            )
            
            # Apply temporal weighting (this would require timestamp metadata)
            # For now, we'll return recent results as-is
            # In a full implementation, you'd weight by recency
            
            return recent_results[:k]
            
        except Exception as e:
            self.logger.error("Error in temporal search", error=str(e))
            return []
    
    async def _topic_based_search(
        self,
        query_embedding: List[float],
        k: int,
        threshold: float,
        topic: Optional[str] = None,
        topic_boost: float = 1.3
    ) -> List[Tuple[str, float]]:
        """Search within specific topics.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results
            threshold: Similarity threshold
            topic: Specific topic to search within
            topic_boost: Boost factor for topic-specific results
            
        Returns:
            List of (content_id, topic_score) tuples
        """
        try:
            if topic:
                # Search in topic-specific index
                topic_index = f"content_topic_{topic}"
                if topic_index in self.vector_store.faiss_manager.get_available_indices():
                    topic_results = await self.vector_store.find_similar_content(
                        query_embedding=query_embedding,
                        k=k,
                        index_name=topic_index,
                        threshold=threshold
                    )
                    
                    # Apply topic boost
                    boosted_results = [
                        (content_id, score * topic_boost)
                        for content_id, score in topic_results
                    ]
                    
                    return boosted_results
            
            # Fallback to general search
            return await self._similarity_search(query_embedding, k, threshold, None)
            
        except Exception as e:
            self.logger.error("Error in topic-based search", error=str(e))
            return []
    
    async def _apply_diversity_selection(
        self,
        results: List[Tuple[str, float]],
        k: int,
        diversity_factor: float
    ) -> List[Tuple[str, float]]:
        """Apply diversity selection to search results.
        
        Args:
            results: Initial search results
            k: Number of results to select
            diversity_factor: Diversity factor (0.0 to 1.0)
            
        Returns:
            Diversified results
        """
        if diversity_factor <= 0 or len(results) <= k:
            return results[:k]
        
        selected = []
        remaining = results.copy()
        
        # Always select the top result
        if remaining:
            selected.append(remaining.pop(0))
        
        # Select remaining results with diversity consideration
        while remaining and len(selected) < k:
            best_candidate = None
            best_score = -1
            
            for i, (candidate_id, candidate_score) in enumerate(remaining):
                candidate_embedding = self.vector_store._get_cached_embedding(candidate_id)
                
                if not candidate_embedding:
                    continue
                
                # Calculate diversity score
                diversity_score = 0.0
                for selected_id, _ in selected:
                    selected_embedding = self.vector_store._get_cached_embedding(selected_id)
                    if selected_embedding:
                        similarity = self._cosine_similarity(candidate_embedding, selected_embedding)
                        diversity_score += (1.0 - similarity)
                
                diversity_score /= len(selected) if selected else 1
                
                # Combined score
                combined_score = (1 - diversity_factor) * candidate_score + diversity_factor * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = (i, (candidate_id, candidate_score))
            
            if best_candidate:
                idx, candidate = best_candidate
                selected.append(candidate)
                remaining.pop(idx)
            else:
                break
        
        return selected
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def _enrich_results(
        self,
        results: List[Tuple[str, float]],
        mode: SearchMode
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Enrich search results with additional metadata.
        
        Args:
            results: Search results
            mode: Search mode used
            
        Returns:
            Enriched results with metadata
        """
        enriched = []
        
        for content_id, score in results:
            metadata = {
                "search_mode": mode.value,
                "similarity_score": score,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add mode-specific metadata
            if mode == SearchMode.TRENDING:
                metadata["trend_factor"] = score / max(1.0, score)  # Normalized trend factor
            elif mode == SearchMode.DIVERSE:
                metadata["diversity_promoted"] = True
            elif mode == SearchMode.TEMPORAL:
                metadata["temporal_weighted"] = True
            
            enriched.append((content_id, score, metadata))
        
        return enriched
    
    async def get_content_recommendations(
        self,
        user_id: str,
        user_history: List[str],
        persona_type: Optional[PersonaType] = None,
        k: int = 10,
        diversity_factor: float = 0.3
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Get personalized content recommendations for a user.
        
        Args:
            user_id: User ID
            user_history: List of content IDs user has interacted with
            persona_type: User's persona type for personalization
            k: Number of recommendations
            diversity_factor: Diversity factor for recommendations
            
        Returns:
            List of (content_id, recommendation_score, metadata) tuples
        """
        try:
            if not user_history:
                # Cold start - return trending content
                trending_results = await self.vector_store.get_trending_content(k=k)
                return [
                    (content_id, score, {"recommendation_type": "cold_start_trending"})
                    for content_id, score in trending_results
                ]
            
            # Get recommendations based on user history
            recommendations = await self.vector_store.get_content_recommendations(
                user_history=user_history,
                k=k,
                diversity_factor=diversity_factor
            )
            
            # Personalize based on persona type
            if persona_type:
                recommendations = await self._personalize_recommendations(
                    recommendations,
                    persona_type
                )
            
            # Add metadata
            enriched_recommendations = [
                (content_id, score, {
                    "recommendation_type": "collaborative_filtering",
                    "persona_personalized": persona_type is not None,
                    "diversity_factor": diversity_factor
                })
                for content_id, score in recommendations
            ]
            
            return enriched_recommendations
            
        except Exception as e:
            self.logger.error("Error getting content recommendations", user_id=user_id, error=str(e))
            return []
    
    async def _personalize_recommendations(
        self,
        recommendations: List[Tuple[str, float]],
        persona_type: PersonaType
    ) -> List[Tuple[str, float]]:
        """Personalize recommendations based on user persona.
        
        Args:
            recommendations: Base recommendations
            persona_type: User's persona type
            
        Returns:
            Personalized recommendations
        """
        # Persona-based content preferences
        persona_preferences = {
            PersonaType.CASUAL: {
                "entertainment": 1.2,
                "lifestyle": 1.1,
                "general": 1.0
            },
            PersonaType.INFLUENCER: {
                "entertainment": 1.3,
                "technology": 1.2,
                "business": 1.1
            },
            PersonaType.BOT: {
                "technology": 1.4,
                "science": 1.2,
                "business": 1.1
            },
            PersonaType.ACTIVIST: {
                "politics": 1.5,
                "science": 1.2,
                "health": 1.1
            }
        }
        
        preferences = persona_preferences.get(persona_type, {})
        
        # Apply persona-based boosting (simplified)
        # In a full implementation, you'd analyze content topics and apply boosts
        personalized = []
        for content_id, score in recommendations:
            # For now, apply a small random boost based on persona
            # In practice, you'd analyze content metadata
            boost = 1.0 + (hash(content_id + persona_type.value) % 20) / 100.0
            personalized_score = score * boost
            personalized.append((content_id, personalized_score))
        
        # Re-sort by personalized score
        personalized.sort(key=lambda x: x[1], reverse=True)
        
        return personalized
    
    async def find_content_clusters(
        self,
        min_cluster_size: int = 5,
        similarity_threshold: float = 0.7
    ) -> Dict[str, List[str]]:
        """Find clusters of similar content.
        
        Args:
            min_cluster_size: Minimum size for a cluster
            similarity_threshold: Minimum similarity for clustering
            
        Returns:
            Dictionary mapping cluster names to content ID lists
        """
        try:
            # This would require implementing clustering algorithms
            # For now, return a placeholder
            
            self.logger.info("Content clustering requested")
            
            clusters = await self.vector_store.cluster_content(
                num_clusters=10
            )
            
            # Filter clusters by minimum size
            filtered_clusters = {
                f"cluster_{i}": content_ids
                for i, content_ids in clusters.items()
                if len(content_ids) >= min_cluster_size
            }
            
            return filtered_clusters
            
        except Exception as e:
            self.logger.error("Error finding content clusters", error=str(e))
            return {}
    
    async def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics and statistics.
        
        Returns:
            Dictionary with search analytics
        """
        try:
            vector_stats = await self.vector_store.get_stats()
            
            analytics = {
                "vector_store_stats": vector_stats,
                "search_config": {
                    "default_k": self.default_k,
                    "default_threshold": self.default_threshold,
                    "max_results": self.max_results
                },
                "available_search_modes": [mode.value for mode in SearchMode],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error("Error getting search analytics", error=str(e))
            return {"error": str(e)}