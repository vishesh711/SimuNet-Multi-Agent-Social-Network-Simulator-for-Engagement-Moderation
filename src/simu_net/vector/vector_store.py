"""High-level vector store interface for content embeddings."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import structlog

from .faiss_manager import FAISSManager
from ..data.models import ContentAgent as ContentAgentModel


class VectorStore:
    """High-level interface for vector storage and retrieval."""
    
    def __init__(
        self,
        faiss_manager: Optional[FAISSManager] = None,
        embedding_dim: int = 384
    ):
        """Initialize vector store.
        
        Args:
            faiss_manager: FAISS manager instance
            embedding_dim: Dimension of embeddings
        """
        self.faiss_manager = faiss_manager or FAISSManager(embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim
        self.logger = structlog.get_logger().bind(component="VectorStore")
        
        # Cache for frequently accessed embeddings
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_max_size = 1000
    
    async def initialize(self) -> None:
        """Initialize the vector store."""
        await self.faiss_manager.initialize()
        self.logger.info("Vector store initialized")
    
    async def store_content_embedding(
        self,
        content: ContentAgentModel,
        index_name: str = "content_all"
    ) -> bool:
        """Store content embedding in the vector database.
        
        Args:
            content: Content agent model with embeddings
            index_name: Name of the index to store in
            
        Returns:
            True if successful, False otherwise
        """
        if not content.embeddings:
            self.logger.warning("No embeddings found for content", content_id=content.content_id)
            return False
        
        try:
            # Store in main index
            success = await self.faiss_manager.add_embeddings(
                index_name=index_name,
                embeddings=[content.embeddings],
                content_ids=[content.content_id]
            )
            
            if success:
                # Also store in specialized indices based on content characteristics
                await self._store_in_specialized_indices(content)
                
                # Cache the embedding
                self._cache_embedding(content.content_id, content.embeddings)
                
                self.logger.debug("Content embedding stored", content_id=content.content_id)
            
            return success
            
        except Exception as e:
            self.logger.error("Error storing content embedding", content_id=content.content_id, error=str(e))
            return False
    
    async def _store_in_specialized_indices(self, content: ContentAgentModel) -> None:
        """Store content in specialized indices based on characteristics.
        
        Args:
            content: Content agent model
        """
        try:
            # Store in recent content index if created within last 24 hours
            if (datetime.utcnow() - content.created_at).total_seconds() < 86400:
                await self.faiss_manager.add_embeddings(
                    index_name="content_recent",
                    embeddings=[content.embeddings],
                    content_ids=[content.content_id]
                )
            
            # Store in viral content index if viral
            if content.is_viral:
                await self.faiss_manager.add_embeddings(
                    index_name="content_viral",
                    embeddings=[content.embeddings],
                    content_ids=[content.content_id]
                )
            
            # Store in topic-specific index based on primary topic
            primary_topic = self._get_primary_topic(content)
            if primary_topic:
                topic_index = f"content_topic_{primary_topic}"
                # Create topic index if it doesn't exist
                if topic_index not in self.faiss_manager.get_available_indices():
                    await self.faiss_manager._create_index(topic_index, f"Content for topic: {primary_topic}")
                
                await self.faiss_manager.add_embeddings(
                    index_name=topic_index,
                    embeddings=[content.embeddings],
                    content_ids=[content.content_id]
                )
            
        except Exception as e:
            self.logger.error("Error storing in specialized indices", content_id=content.content_id, error=str(e))
    
    def _get_primary_topic(self, content: ContentAgentModel) -> Optional[str]:
        """Get the primary topic for content.
        
        Args:
            content: Content agent model
            
        Returns:
            Primary topic name or None
        """
        if not content.metadata.topic_classification:
            return None
        
        # Find topic with highest score
        primary_topic = max(
            content.metadata.topic_classification.items(),
            key=lambda x: x[1]
        )
        
        # Only return if score is above threshold
        if primary_topic[1] > 0.3:
            return primary_topic[0]
        
        return None
    
    async def find_similar_content(
        self,
        query_embedding: List[float],
        k: int = 10,
        index_name: str = "content_all",
        threshold: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """Find similar content based on embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            index_name: Name of the index to search
            threshold: Minimum similarity threshold
            filters: Optional filters to apply
            
        Returns:
            List of (content_id, similarity_score) tuples
        """
        try:
            results = await self.faiss_manager.search_similar(
                index_name=index_name,
                query_embedding=query_embedding,
                k=k,
                threshold=threshold
            )
            
            # Apply additional filters if provided
            if filters:
                results = await self._apply_filters(results, filters)
            
            return results
            
        except Exception as e:
            self.logger.error("Error finding similar content", error=str(e))
            return []
    
    async def find_similar_to_content(
        self,
        content_id: str,
        k: int = 10,
        index_name: str = "content_all",
        threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Find content similar to a specific content item.
        
        Args:
            content_id: ID of the content to find similar items for
            k: Number of results to return
            index_name: Name of the index to search
            threshold: Minimum similarity threshold
            
        Returns:
            List of (content_id, similarity_score) tuples
        """
        # Get embedding for the content
        embedding = self._get_cached_embedding(content_id)
        if not embedding:
            self.logger.warning("No embedding found for content", content_id=content_id)
            return []
        
        # Find similar content
        results = await self.find_similar_content(
            query_embedding=embedding,
            k=k + 1,  # +1 to account for the query content itself
            index_name=index_name,
            threshold=threshold
        )
        
        # Remove the query content from results
        results = [(cid, score) for cid, score in results if cid != content_id]
        
        return results[:k]
    
    async def get_trending_content(
        self,
        k: int = 20,
        time_window_hours: int = 24
    ) -> List[Tuple[str, float]]:
        """Get trending content based on recent activity.
        
        Args:
            k: Number of results to return
            time_window_hours: Time window to consider for trending
            
        Returns:
            List of (content_id, trend_score) tuples
        """
        try:
            # Search in recent content index
            # For now, we'll return recent content with high similarity to viral content
            
            # Get some viral content embeddings as reference
            viral_results = await self.faiss_manager.search_similar(
                index_name="content_viral",
                query_embedding=[0.0] * self.embedding_dim,  # Dummy query
                k=5
            )
            
            if not viral_results:
                # Fallback to recent content
                return await self._get_recent_content(k)
            
            # Use viral content as queries to find trending content
            trending_content = set()
            for viral_content_id, _ in viral_results:
                viral_embedding = self._get_cached_embedding(viral_content_id)
                if viral_embedding:
                    similar_results = await self.find_similar_content(
                        query_embedding=viral_embedding,
                        k=k // len(viral_results) + 1,
                        index_name="content_recent",
                        threshold=0.3
                    )
                    trending_content.update(similar_results)
            
            # Sort by similarity score and return top k
            trending_list = list(trending_content)
            trending_list.sort(key=lambda x: x[1], reverse=True)
            
            return trending_list[:k]
            
        except Exception as e:
            self.logger.error("Error getting trending content", error=str(e))
            return []
    
    async def _get_recent_content(self, k: int) -> List[Tuple[str, float]]:
        """Get recent content as fallback for trending.
        
        Args:
            k: Number of results to return
            
        Returns:
            List of (content_id, score) tuples
        """
        # This is a simplified implementation
        # In practice, you'd query the recent index with various strategies
        return []
    
    async def cluster_content(
        self,
        index_name: str = "content_all",
        num_clusters: int = 10
    ) -> Dict[int, List[str]]:
        """Cluster content based on embeddings.
        
        Args:
            index_name: Name of the index to cluster
            num_clusters: Number of clusters to create
            
        Returns:
            Dictionary mapping cluster IDs to lists of content IDs
        """
        try:
            # This would require implementing k-means clustering on the embeddings
            # For now, we'll return a placeholder implementation
            
            self.logger.info("Content clustering requested", index_name=index_name, num_clusters=num_clusters)
            
            # Placeholder: return empty clusters
            return {i: [] for i in range(num_clusters)}
            
        except Exception as e:
            self.logger.error("Error clustering content", error=str(e))
            return {}
    
    async def get_content_recommendations(
        self,
        user_history: List[str],
        k: int = 10,
        diversity_factor: float = 0.3
    ) -> List[Tuple[str, float]]:
        """Get content recommendations based on user history.
        
        Args:
            user_history: List of content IDs the user has interacted with
            k: Number of recommendations to return
            diversity_factor: Factor to promote diversity (0.0 = no diversity, 1.0 = max diversity)
            
        Returns:
            List of (content_id, recommendation_score) tuples
        """
        try:
            if not user_history:
                return []
            
            # Get embeddings for user's history
            user_embeddings = []
            for content_id in user_history:
                embedding = self._get_cached_embedding(content_id)
                if embedding:
                    user_embeddings.append(embedding)
            
            if not user_embeddings:
                return []
            
            # Calculate user preference vector (average of history embeddings)
            user_preference = self._average_embeddings(user_embeddings)
            
            # Find similar content
            similar_content = await self.find_similar_content(
                query_embedding=user_preference,
                k=k * 3,  # Get more results for diversity filtering
                threshold=0.2
            )
            
            # Remove content already in user history
            filtered_content = [
                (cid, score) for cid, score in similar_content 
                if cid not in user_history
            ]
            
            # Apply diversity filtering if requested
            if diversity_factor > 0:
                filtered_content = await self._apply_diversity_filter(
                    filtered_content, 
                    diversity_factor
                )
            
            return filtered_content[:k]
            
        except Exception as e:
            self.logger.error("Error getting content recommendations", error=str(e))
            return []
    
    def _average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Calculate average of multiple embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Average embedding vector
        """
        if not embeddings:
            return [0.0] * self.embedding_dim
        
        avg_embedding = [0.0] * len(embeddings[0])
        for embedding in embeddings:
            for i, value in enumerate(embedding):
                avg_embedding[i] += value
        
        # Normalize
        for i in range(len(avg_embedding)):
            avg_embedding[i] /= len(embeddings)
        
        return avg_embedding
    
    async def _apply_diversity_filter(
        self,
        content_list: List[Tuple[str, float]],
        diversity_factor: float
    ) -> List[Tuple[str, float]]:
        """Apply diversity filtering to content recommendations.
        
        Args:
            content_list: List of (content_id, score) tuples
            diversity_factor: Diversity factor (0.0 to 1.0)
            
        Returns:
            Filtered list with improved diversity
        """
        if diversity_factor <= 0 or len(content_list) <= 1:
            return content_list
        
        # Simple diversity implementation: select content that's not too similar to already selected
        selected = []
        remaining = content_list.copy()
        
        # Always select the top result
        if remaining:
            selected.append(remaining.pop(0))
        
        while remaining and len(selected) < len(content_list):
            best_candidate = None
            best_score = -1
            
            for candidate in remaining:
                candidate_id, candidate_score = candidate
                candidate_embedding = self._get_cached_embedding(candidate_id)
                
                if not candidate_embedding:
                    continue
                
                # Calculate diversity score (average distance to selected items)
                diversity_score = 0.0
                for selected_id, _ in selected:
                    selected_embedding = self._get_cached_embedding(selected_id)
                    if selected_embedding:
                        similarity = self._cosine_similarity(candidate_embedding, selected_embedding)
                        diversity_score += (1.0 - similarity)  # Distance = 1 - similarity
                
                diversity_score /= len(selected)
                
                # Combined score: relevance + diversity
                combined_score = (1 - diversity_factor) * candidate_score + diversity_factor * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
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
    
    async def _apply_filters(
        self,
        results: List[Tuple[str, float]],
        filters: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """Apply additional filters to search results.
        
        Args:
            results: List of (content_id, score) tuples
            filters: Dictionary of filters to apply
            
        Returns:
            Filtered results
        """
        # This would require additional metadata storage
        # For now, return results as-is
        return results
    
    def _cache_embedding(self, content_id: str, embedding: List[float]) -> None:
        """Cache an embedding for quick access.
        
        Args:
            content_id: Content ID
            embedding: Embedding vector
        """
        if len(self._embedding_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        
        self._embedding_cache[content_id] = embedding
    
    def _get_cached_embedding(self, content_id: str) -> Optional[List[float]]:
        """Get cached embedding for content.
        
        Args:
            content_id: Content ID
            
        Returns:
            Embedding vector or None if not cached
        """
        return self._embedding_cache.get(content_id)
    
    async def remove_content(self, content_id: str) -> bool:
        """Remove content from all indices.
        
        Args:
            content_id: Content ID to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = True
            
            # Remove from all indices
            for index_name in self.faiss_manager.get_available_indices():
                result = await self.faiss_manager.remove_embeddings(
                    index_name=index_name,
                    content_ids=[content_id]
                )
                success = success and result
            
            # Remove from cache
            if content_id in self._embedding_cache:
                del self._embedding_cache[content_id]
            
            self.logger.debug("Content removed from vector store", content_id=content_id)
            return success
            
        except Exception as e:
            self.logger.error("Error removing content", content_id=content_id, error=str(e))
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = {
                "embedding_dim": self.embedding_dim,
                "cache_size": len(self._embedding_cache),
                "cache_max_size": self._cache_max_size,
                "available_indices": self.faiss_manager.get_available_indices(),
                "indices": {}
            }
            
            # Get stats for each index
            for index_name in self.faiss_manager.get_available_indices():
                index_stats = await self.faiss_manager.get_index_stats(index_name)
                stats["indices"][index_name] = index_stats
            
            return stats
            
        except Exception as e:
            self.logger.error("Error getting vector store stats", error=str(e))
            return {"error": str(e)}
    
    async def cleanup(self) -> None:
        """Clean up vector store resources."""
        try:
            # Clear cache
            self._embedding_cache.clear()
            
            # Clean up old indices
            await self.faiss_manager.cleanup_old_indices()
            
            self.logger.info("Vector store cleanup completed")
            
        except Exception as e:
            self.logger.error("Error during cleanup", error=str(e))