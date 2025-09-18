"""
SimuNet: Multi-Agent Social Network Simulator for Engagement & Moderation

A research platform for studying social media dynamics through multi-agent simulation.
"""

__version__ = "0.1.0"
__author__ = "SimuNet Team"
__email__ = "team@simu-net.dev"

from .agents import SimuNetAgent, AgentRegistry, UserAgent, ContentAgent
from .events import EventManager, AgentEvent
from .config import Settings

# Optional vector database components
try:
    from .vector import FAISSManager, VectorStore, SimilaritySearchEngine
    _VECTOR_AVAILABLE = True
except ImportError:
    _VECTOR_AVAILABLE = False
    FAISSManager = None
    VectorStore = None
    SimilaritySearchEngine = None

__all__ = ["SimuNetAgent", "AgentRegistry", "UserAgent", "ContentAgent", "EventManager", "AgentEvent", "Settings"]

# Add vector components to __all__ if available
if _VECTOR_AVAILABLE:
    __all__.extend(["FAISSManager", "VectorStore", "SimilaritySearchEngine"])