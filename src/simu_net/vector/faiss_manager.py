"""FAISS vector database manager for content embeddings."""

import os
import pickle
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import structlog

from ..config import get_settings

# Optional FAISS import - graceful degradation if not available
try:
    import faiss
    import numpy as np
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    np = None
    _FAISS_AVAILABLE = False


class FAISSManager:
    """FAISS-based vector database manager for content embeddings."""
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_dim: int = 384,
        index_type: str = "IVF",
        nlist: int = 100
    ):
        """Initialize FAISS manager.
        
        Args:
            index_path: Path to store FAISS indices
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index (Flat, IVF, HNSW)
            nlist: Number of clusters for IVF index
        """
        settings = get_settings()
        self.index_path = Path(index_path or settings.database.faiss_index_path)
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        
        self.logger = structlog.get_logger().bind(component="FAISSManager")
        
        # FAISS indices for different content types and time windows
        self._indices: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}  # Store metadata for each index
        self._id_mappings: Dict[str, Dict[int, str]] = {}  # Map FAISS IDs to content IDs
        self._reverse_mappings: Dict[str, Dict[str, int]] = {}  # Map content IDs to FAISS IDs
        
        # Ensure index directory exists
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        if not _FAISS_AVAILABLE:
            self.logger.warning("FAISS not available, using mock implementation")
    
    async def initialize(self) -> None:
        """Initialize FAISS indices."""
        try:
            if _FAISS_AVAILABLE:
                # Load existing indices or create new ones
                await self._load_or_create_indices()
            else:
                # Initialize mock indices
                await self._initialize_mock_indices()
            
            self.logger.info("FAISS manager initialized", index_path=str(self.index_path))
            
        except Exception as e:
            self.logger.error("Error initializing FAISS manager", error=str(e))
            raise
    
    async def _load_or_create_indices(self) -> None:
        """Load existing indices or create new ones."""
        index_configs = [
            ("content_all", "All content embeddings"),
            ("content_recent", "Recent content (last 24h)"),
            ("content_viral", "Viral content embeddings"),
            ("content_by_topic", "Content grouped by topic"),
        ]
        
        for index_name, description in index_configs:
            index_file = self.index_path / f"{index_name}.faiss"
            metadata_file = self.index_path / f"{index_name}_metadata.pkl"
            
            if index_file.exists() and metadata_file.exists():
                # Load existing index
                await self._load_index(index_name)
            else:
                # Create new index
                await self._create_index(index_name, description)
    
    async def _initialize_mock_indices(self) -> None:
        """Initialize mock indices when FAISS is not available."""
        mock_indices = [
            "content_all",
            "content_recent", 
            "content_viral",
            "content_by_topic"
        ]
        
        for index_name in mock_indices:
            self._indices[index_name] = MockFAISSIndex()
            self._metadata[index_name] = {
                "description": f"Mock {index_name} index",
                "created_at": datetime.utcnow(),
                "total_vectors": 0,
                "embedding_dim": self.embedding_dim
            }
            self._id_mappings[index_name] = {}
            self._reverse_mappings[index_name] = {}
    
    async def _load_index(self, index_name: str) -> None:
        """Load an existing FAISS index.
        
        Args:
            index_name: Name of the index to load
        """
        try:
            index_file = self.index_path / f"{index_name}.faiss"
            metadata_file = self.index_path / f"{index_name}_metadata.pkl"
            
            # Load FAISS index
            if _FAISS_AVAILABLE:
                loop = asyncio.get_event_loop()
                index = await loop.run_in_executor(
                    None, 
                    faiss.read_index, 
                    str(index_file)
                )
                self._indices[index_name] = index
            else:
                self._indices[index_name] = MockFAISSIndex()
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                self._metadata[index_name] = metadata
                self._id_mappings[index_name] = metadata.get('id_mappings', {})
                self._reverse_mappings[index_name] = metadata.get('reverse_mappings', {})
            
            self.logger.info("Index loaded", index_name=index_name)
            
        except Exception as e:
            self.logger.error("Error loading index", index_name=index_name, error=str(e))
            # Fallback to creating new index
            await self._create_index(index_name, f"Recreated {index_name}")
    
    async def _create_index(self, index_name: str, description: str) -> None:
        """Create a new FAISS index.
        
        Args:
            index_name: Name of the index to create
            description: Description of the index
        """
        try:
            if _FAISS_AVAILABLE:
                # Create appropriate FAISS index based on type
                if self.index_type == "Flat":
                    index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
                elif self.index_type == "IVF":
                    quantizer = faiss.IndexFlatIP(self.embedding_dim)
                    index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist)
                elif self.index_type == "HNSW":
                    index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
                else:
                    # Default to flat index
                    index = faiss.IndexFlatIP(self.embedding_dim)
                
                self._indices[index_name] = index
            else:
                self._indices[index_name] = MockFAISSIndex()
            
            # Initialize metadata
            self._metadata[index_name] = {
                "description": description,
                "created_at": datetime.utcnow(),
                "total_vectors": 0,
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type
            }
            self._id_mappings[index_name] = {}
            self._reverse_mappings[index_name] = {}
            
            self.logger.info("Index created", index_name=index_name, index_type=self.index_type)
            
        except Exception as e:
            self.logger.error("Error creating index", index_name=index_name, error=str(e))
            raise
    
    async def add_embeddings(
        self,
        index_name: str,
        embeddings: List[List[float]],
        content_ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Add embeddings to a FAISS index.
        
        Args:
            index_name: Name of the index
            embeddings: List of embedding vectors
            content_ids: List of content IDs corresponding to embeddings
            metadata: Optional metadata for each embedding
            
        Returns:
            True if successful, False otherwise
        """
        if index_name not in self._indices:
            self.logger.error("Index not found", index_name=index_name)
            return False
        
        if len(embeddings) != len(content_ids):
            self.logger.error("Embeddings and content_ids length mismatch")
            return False
        
        try:
            if _FAISS_AVAILABLE:
                # Convert to numpy array
                embeddings_array = np.array(embeddings, dtype=np.float32)
                
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings_array)
                
                # Add to index
                index = self._indices[index_name]
                start_id = index.ntotal
                
                # Train index if needed (for IVF)
                if hasattr(index, 'is_trained') and not index.is_trained:
                    if embeddings_array.shape[0] >= self.nlist:
                        index.train(embeddings_array)
                    else:
                        self.logger.warning(
                            "Not enough vectors to train IVF index",
                            required=self.nlist,
                            available=embeddings_array.shape[0]
                        )
                
                # Add vectors
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, index.add, embeddings_array)
                
                # Update mappings
                for i, content_id in enumerate(content_ids):
                    faiss_id = start_id + i
                    self._id_mappings[index_name][faiss_id] = content_id
                    self._reverse_mappings[index_name][content_id] = faiss_id
            else:
                # Mock implementation
                mock_index = self._indices[index_name]
                for i, (embedding, content_id) in enumerate(zip(embeddings, content_ids)):
                    mock_index.add_vector(embedding, content_id)
                    faiss_id = len(self._id_mappings[index_name])
                    self._id_mappings[index_name][faiss_id] = content_id
                    self._reverse_mappings[index_name][content_id] = faiss_id
            
            # Update metadata
            self._metadata[index_name]["total_vectors"] += len(embeddings)
            self._metadata[index_name]["last_updated"] = datetime.utcnow()
            
            # Save index and metadata
            await self._save_index(index_name)
            
            self.logger.info(
                "Embeddings added to index",
                index_name=index_name,
                count=len(embeddings)
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Error adding embeddings", index_name=index_name, error=str(e))
            return False
    
    async def search_similar(
        self,
        index_name: str,
        query_embedding: List[float],
        k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Search for similar embeddings.
        
        Args:
            index_name: Name of the index to search
            query_embedding: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (content_id, similarity_score) tuples
        """
        if index_name not in self._indices:
            self.logger.error("Index not found", index_name=index_name)
            return []
        
        try:
            if _FAISS_AVAILABLE:
                # Convert to numpy array and normalize
                query_array = np.array([query_embedding], dtype=np.float32)
                faiss.normalize_L2(query_array)
                
                # Search
                index = self._indices[index_name]
                loop = asyncio.get_event_loop()
                distances, indices = await loop.run_in_executor(
                    None,
                    index.search,
                    query_array,
                    k
                )
                
                # Convert results
                results = []
                for i, (distance, faiss_id) in enumerate(zip(distances[0], indices[0])):
                    if faiss_id == -1:  # No more results
                        break
                    
                    if distance >= threshold:
                        content_id = self._id_mappings[index_name].get(faiss_id)
                        if content_id:
                            results.append((content_id, float(distance)))
                
                return results
            else:
                # Mock implementation
                mock_index = self._indices[index_name]
                return mock_index.search(query_embedding, k, threshold)
                
        except Exception as e:
            self.logger.error("Error searching index", index_name=index_name, error=str(e))
            return []
    
    async def remove_embeddings(
        self,
        index_name: str,
        content_ids: List[str]
    ) -> bool:
        """Remove embeddings from index.
        
        Args:
            index_name: Name of the index
            content_ids: List of content IDs to remove
            
        Returns:
            True if successful, False otherwise
        """
        if index_name not in self._indices:
            self.logger.error("Index not found", index_name=index_name)
            return False
        
        try:
            if _FAISS_AVAILABLE:
                # FAISS doesn't support direct removal, so we need to rebuild
                # For now, we'll mark them as removed in metadata
                # A full rebuild can be done periodically
                
                removed_count = 0
                for content_id in content_ids:
                    if content_id in self._reverse_mappings[index_name]:
                        faiss_id = self._reverse_mappings[index_name][content_id]
                        del self._reverse_mappings[index_name][content_id]
                        del self._id_mappings[index_name][faiss_id]
                        removed_count += 1
                
                self._metadata[index_name]["total_vectors"] -= removed_count
                self._metadata[index_name]["last_updated"] = datetime.utcnow()
                
                # Save metadata
                await self._save_metadata(index_name)
                
                self.logger.info(
                    "Embeddings marked for removal",
                    index_name=index_name,
                    count=removed_count
                )
                
                return True
            else:
                # Mock implementation
                mock_index = self._indices[index_name]
                for content_id in content_ids:
                    mock_index.remove_vector(content_id)
                    if content_id in self._reverse_mappings[index_name]:
                        faiss_id = self._reverse_mappings[index_name][content_id]
                        del self._reverse_mappings[index_name][content_id]
                        del self._id_mappings[index_name][faiss_id]
                
                return True
                
        except Exception as e:
            self.logger.error("Error removing embeddings", index_name=index_name, error=str(e))
            return False
    
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics for an index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Dictionary with index statistics
        """
        if index_name not in self._indices:
            return {"error": "Index not found"}
        
        try:
            stats = self._metadata[index_name].copy()
            
            if _FAISS_AVAILABLE:
                index = self._indices[index_name]
                stats.update({
                    "faiss_total": index.ntotal,
                    "is_trained": getattr(index, 'is_trained', True),
                    "index_type": type(index).__name__
                })
            else:
                mock_index = self._indices[index_name]
                stats.update({
                    "mock_total": len(mock_index.vectors),
                    "is_trained": True,
                    "index_type": "MockFAISSIndex"
                })
            
            return stats
            
        except Exception as e:
            self.logger.error("Error getting index stats", index_name=index_name, error=str(e))
            return {"error": str(e)}
    
    async def rebuild_index(self, index_name: str) -> bool:
        """Rebuild an index to remove deleted vectors.
        
        Args:
            index_name: Name of the index to rebuild
            
        Returns:
            True if successful, False otherwise
        """
        if index_name not in self._indices:
            self.logger.error("Index not found", index_name=index_name)
            return False
        
        try:
            # This would involve re-adding all valid vectors to a new index
            # For now, we'll just log that a rebuild is needed
            self.logger.info("Index rebuild requested", index_name=index_name)
            
            # In a full implementation, you would:
            # 1. Create a new index
            # 2. Re-add all valid vectors
            # 3. Replace the old index
            # 4. Update all mappings
            
            return True
            
        except Exception as e:
            self.logger.error("Error rebuilding index", index_name=index_name, error=str(e))
            return False
    
    async def _save_index(self, index_name: str) -> None:
        """Save index and metadata to disk.
        
        Args:
            index_name: Name of the index to save
        """
        try:
            if _FAISS_AVAILABLE:
                # Save FAISS index
                index_file = self.index_path / f"{index_name}.faiss"
                index = self._indices[index_name]
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    faiss.write_index,
                    index,
                    str(index_file)
                )
            
            # Save metadata
            await self._save_metadata(index_name)
            
        except Exception as e:
            self.logger.error("Error saving index", index_name=index_name, error=str(e))
    
    async def _save_metadata(self, index_name: str) -> None:
        """Save metadata to disk.
        
        Args:
            index_name: Name of the index
        """
        try:
            metadata_file = self.index_path / f"{index_name}_metadata.pkl"
            
            metadata = self._metadata[index_name].copy()
            metadata['id_mappings'] = self._id_mappings[index_name]
            metadata['reverse_mappings'] = self._reverse_mappings[index_name]
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
                
        except Exception as e:
            self.logger.error("Error saving metadata", index_name=index_name, error=str(e))
    
    async def cleanup_old_indices(self, max_age_days: int = 30) -> None:
        """Clean up old or unused indices.
        
        Args:
            max_age_days: Maximum age of indices to keep
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            
            for index_name, metadata in self._metadata.items():
                created_at = metadata.get('created_at')
                if created_at and created_at < cutoff_date:
                    # Check if index is still being used
                    last_updated = metadata.get('last_updated', created_at)
                    if last_updated < cutoff_date:
                        self.logger.info("Cleaning up old index", index_name=index_name)
                        # Remove index files
                        index_file = self.index_path / f"{index_name}.faiss"
                        metadata_file = self.index_path / f"{index_name}_metadata.pkl"
                        
                        if index_file.exists():
                            index_file.unlink()
                        if metadata_file.exists():
                            metadata_file.unlink()
                        
                        # Remove from memory
                        del self._indices[index_name]
                        del self._metadata[index_name]
                        del self._id_mappings[index_name]
                        del self._reverse_mappings[index_name]
            
        except Exception as e:
            self.logger.error("Error cleaning up indices", error=str(e))
    
    def get_available_indices(self) -> List[str]:
        """Get list of available indices.
        
        Returns:
            List of index names
        """
        return list(self._indices.keys())


class MockFAISSIndex:
    """Mock FAISS index for testing and fallback."""
    
    def __init__(self):
        """Initialize mock index."""
        self.vectors: Dict[str, List[float]] = {}
        self.ntotal = 0
    
    def add_vector(self, embedding: List[float], content_id: str) -> None:
        """Add a vector to the mock index."""
        self.vectors[content_id] = embedding
        self.ntotal += 1
    
    def remove_vector(self, content_id: str) -> None:
        """Remove a vector from the mock index."""
        if content_id in self.vectors:
            del self.vectors[content_id]
            self.ntotal -= 1
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors in the mock index."""
        if not self.vectors:
            return []
        
        # Calculate cosine similarity with all vectors
        similarities = []
        
        for content_id, embedding in self.vectors.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            if similarity >= threshold:
                similarities.append((content_id, similarity))
        
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
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