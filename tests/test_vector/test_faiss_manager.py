"""Tests for FAISS manager."""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from simu_net.vector.faiss_manager import FAISSManager, MockFAISSIndex


class TestMockFAISSIndex:
    """Test MockFAISSIndex functionality."""
    
    def test_mock_index_initialization(self):
        """Test mock index initialization."""
        index = MockFAISSIndex()
        assert index.ntotal == 0
        assert len(index.vectors) == 0
    
    def test_add_vector(self):
        """Test adding vectors to mock index."""
        index = MockFAISSIndex()
        
        embedding = [0.1, 0.2, 0.3]
        content_id = "test-content-1"
        
        index.add_vector(embedding, content_id)
        
        assert index.ntotal == 1
        assert content_id in index.vectors
        assert index.vectors[content_id] == embedding
    
    def test_remove_vector(self):
        """Test removing vectors from mock index."""
        index = MockFAISSIndex()
        
        # Add vector
        embedding = [0.1, 0.2, 0.3]
        content_id = "test-content-1"
        index.add_vector(embedding, content_id)
        
        # Remove vector
        index.remove_vector(content_id)
        
        assert index.ntotal == 0
        assert content_id not in index.vectors
    
    def test_search_empty_index(self):
        """Test searching empty mock index."""
        index = MockFAISSIndex()
        
        query = [0.1, 0.2, 0.3]
        results = index.search(query, k=5)
        
        assert results == []
    
    def test_search_with_results(self):
        """Test searching mock index with results."""
        index = MockFAISSIndex()
        
        # Add some vectors
        vectors = [
            ([1.0, 0.0, 0.0], "content-1"),
            ([0.0, 1.0, 0.0], "content-2"),
            ([0.0, 0.0, 1.0], "content-3"),
            ([0.9, 0.1, 0.0], "content-4")  # Similar to content-1
        ]
        
        for embedding, content_id in vectors:
            index.add_vector(embedding, content_id)
        
        # Search for vector similar to content-1
        query = [1.0, 0.0, 0.0]
        results = index.search(query, k=3)
        
        assert len(results) <= 3
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)
        
        # Results should be sorted by similarity (descending)
        similarities = [result[1] for result in results]
        assert similarities == sorted(similarities, reverse=True)
        
        # Most similar should be content-1 (identical)
        assert results[0][0] == "content-1"
        assert results[0][1] == 1.0  # Perfect similarity
    
    def test_search_with_threshold(self):
        """Test searching with similarity threshold."""
        index = MockFAISSIndex()
        
        # Add vectors
        vectors = [
            ([1.0, 0.0, 0.0], "content-1"),
            ([0.5, 0.5, 0.0], "content-2"),
            ([0.0, 0.0, 1.0], "content-3")
        ]
        
        for embedding, content_id in vectors:
            index.add_vector(embedding, content_id)
        
        # Search with high threshold
        query = [1.0, 0.0, 0.0]
        results = index.search(query, k=5, threshold=0.8)
        
        # Only content-1 should meet the threshold
        assert len(results) == 1
        assert results[0][0] == "content-1"
    
    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation."""
        index = MockFAISSIndex()
        
        # Test identical vectors
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        similarity = index._cosine_similarity(a, b)
        assert similarity == 1.0
        
        # Test orthogonal vectors
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        similarity = index._cosine_similarity(a, b)
        assert similarity == 0.0
        
        # Test opposite vectors
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        similarity = index._cosine_similarity(a, b)
        assert similarity == -1.0
        
        # Test zero vectors
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        similarity = index._cosine_similarity(a, b)
        assert similarity == 0.0
        
        # Test different length vectors
        a = [1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        similarity = index._cosine_similarity(a, b)
        assert similarity == 0.0


class TestFAISSManager:
    """Test FAISSManager functionality."""
    
    @pytest.fixture
    def temp_index_path(self):
        """Create temporary directory for indices."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def faiss_manager(self, temp_index_path):
        """Create FAISS manager with temporary path."""
        return FAISSManager(
            index_path=str(temp_index_path),
            embedding_dim=3,  # Small dimension for testing
            index_type="Flat"
        )
    
    def test_faiss_manager_initialization(self, faiss_manager, temp_index_path):
        """Test FAISS manager initialization."""
        assert faiss_manager.index_path == temp_index_path
        assert faiss_manager.embedding_dim == 3
        assert faiss_manager.index_type == "Flat"
        assert faiss_manager.nlist == 100
        
        # Index path should be created
        assert temp_index_path.exists()
    
    async def test_initialize_mock_indices(self, faiss_manager):
        """Test initialization with mock indices."""
        # Mock FAISS as unavailable
        with patch('simu_net.vector.faiss_manager._FAISS_AVAILABLE', False):
            await faiss_manager.initialize()
        
        # Should have created mock indices
        expected_indices = ["content_all", "content_recent", "content_viral", "content_by_topic"]
        available_indices = faiss_manager.get_available_indices()
        
        for index_name in expected_indices:
            assert index_name in available_indices
            assert isinstance(faiss_manager._indices[index_name], MockFAISSIndex)
    
    async def test_add_embeddings_mock(self, faiss_manager):
        """Test adding embeddings with mock implementation."""
        # Initialize with mock
        with patch('simu_net.vector.faiss_manager._FAISS_AVAILABLE', False):
            await faiss_manager.initialize()
        
        # Add embeddings
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        content_ids = ["content-1", "content-2", "content-3"]
        
        success = await faiss_manager.add_embeddings(
            index_name="content_all",
            embeddings=embeddings,
            content_ids=content_ids
        )
        
        assert success is True
        
        # Check metadata
        stats = await faiss_manager.get_index_stats("content_all")
        assert stats["total_vectors"] == 3
        
        # Check mappings
        assert len(faiss_manager._id_mappings["content_all"]) == 3
        assert len(faiss_manager._reverse_mappings["content_all"]) == 3
    
    async def test_search_similar_mock(self, faiss_manager):
        """Test similarity search with mock implementation."""
        # Initialize with mock
        with patch('simu_net.vector.faiss_manager._FAISS_AVAILABLE', False):
            await faiss_manager.initialize()
        
        # Add embeddings
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],  # Similar to first
            [0.0, 1.0, 0.0]   # Different
        ]
        content_ids = ["content-1", "content-2", "content-3"]
        
        await faiss_manager.add_embeddings(
            index_name="content_all",
            embeddings=embeddings,
            content_ids=content_ids
        )
        
        # Search for similar content
        query_embedding = [1.0, 0.0, 0.0]
        results = await faiss_manager.search_similar(
            index_name="content_all",
            query_embedding=query_embedding,
            k=2,
            threshold=0.5
        )
        
        assert len(results) <= 2
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)
        
        # Should find content-1 and content-2 (both similar to query)
        content_ids_found = [result[0] for result in results]
        assert "content-1" in content_ids_found
        assert "content-2" in content_ids_found
    
    async def test_remove_embeddings_mock(self, faiss_manager):
        """Test removing embeddings with mock implementation."""
        # Initialize with mock
        with patch('simu_net.vector.faiss_manager._FAISS_AVAILABLE', False):
            await faiss_manager.initialize()
        
        # Add embeddings
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        content_ids = ["content-1", "content-2"]
        
        await faiss_manager.add_embeddings(
            index_name="content_all",
            embeddings=embeddings,
            content_ids=content_ids
        )
        
        # Remove one embedding
        success = await faiss_manager.remove_embeddings(
            index_name="content_all",
            content_ids=["content-1"]
        )
        
        assert success is True
        
        # Check that it was removed from mappings
        assert "content-1" not in faiss_manager._reverse_mappings["content_all"]
        assert "content-2" in faiss_manager._reverse_mappings["content_all"]
    
    async def test_get_index_stats(self, faiss_manager):
        """Test getting index statistics."""
        # Initialize with mock
        with patch('simu_net.vector.faiss_manager._FAISS_AVAILABLE', False):
            await faiss_manager.initialize()
        
        # Get stats for empty index
        stats = await faiss_manager.get_index_stats("content_all")
        
        assert "total_vectors" in stats
        assert "created_at" in stats
        assert "embedding_dim" in stats
        assert "index_type" in stats
        assert stats["total_vectors"] == 0
        assert stats["embedding_dim"] == 3
    
    async def test_nonexistent_index_operations(self, faiss_manager):
        """Test operations on non-existent indices."""
        await faiss_manager.initialize()
        
        # Try to add to non-existent index
        success = await faiss_manager.add_embeddings(
            index_name="nonexistent",
            embeddings=[[1.0, 0.0, 0.0]],
            content_ids=["content-1"]
        )
        assert success is False
        
        # Try to search non-existent index
        results = await faiss_manager.search_similar(
            index_name="nonexistent",
            query_embedding=[1.0, 0.0, 0.0],
            k=5
        )
        assert results == []
        
        # Try to get stats for non-existent index
        stats = await faiss_manager.get_index_stats("nonexistent")
        assert "error" in stats
    
    async def test_embeddings_content_ids_mismatch(self, faiss_manager):
        """Test handling of mismatched embeddings and content IDs."""
        # Initialize with mock
        with patch('simu_net.vector.faiss_manager._FAISS_AVAILABLE', False):
            await faiss_manager.initialize()
        
        # Try to add mismatched embeddings and content IDs
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        content_ids = ["content-1"]  # Only one ID for two embeddings
        
        success = await faiss_manager.add_embeddings(
            index_name="content_all",
            embeddings=embeddings,
            content_ids=content_ids
        )
        
        assert success is False
    
    async def test_cleanup_old_indices(self, faiss_manager, temp_index_path):
        """Test cleanup of old indices."""
        # Initialize with mock
        with patch('simu_net.vector.faiss_manager._FAISS_AVAILABLE', False):
            await faiss_manager.initialize()
        
        # Create some fake old index files
        old_index_file = temp_index_path / "old_index.faiss"
        old_metadata_file = temp_index_path / "old_index_metadata.pkl"
        
        old_index_file.touch()
        old_metadata_file.touch()
        
        # Run cleanup (should not affect anything since we don't have old indices in memory)
        await faiss_manager.cleanup_old_indices(max_age_days=0)
        
        # Files should still exist since they're not tracked in memory
        assert old_index_file.exists()
        assert old_metadata_file.exists()
    
    def test_get_available_indices(self, faiss_manager):
        """Test getting available indices."""
        # Initially empty
        indices = faiss_manager.get_available_indices()
        assert indices == []
        
        # Add a mock index
        faiss_manager._indices["test_index"] = MockFAISSIndex()
        
        indices = faiss_manager.get_available_indices()
        assert "test_index" in indices
    
    async def test_rebuild_index(self, faiss_manager):
        """Test index rebuilding."""
        # Initialize with mock
        with patch('simu_net.vector.faiss_manager._FAISS_AVAILABLE', False):
            await faiss_manager.initialize()
        
        # Rebuild should succeed (even though it's a placeholder implementation)
        success = await faiss_manager.rebuild_index("content_all")
        assert success is True
        
        # Try to rebuild non-existent index
        success = await faiss_manager.rebuild_index("nonexistent")
        assert success is False