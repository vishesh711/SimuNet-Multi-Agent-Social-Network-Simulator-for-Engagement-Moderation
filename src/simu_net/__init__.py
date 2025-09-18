"""
SimuNet: Multi-Agent Social Network Simulator for Engagement & Moderation

A research platform for studying social media dynamics through multi-agent simulation.
"""

__version__ = "0.1.0"
__author__ = "SimuNet Team"
__email__ = "team@simu-net.dev"

from .core import SimuNetAgent
from .config import Settings

__all__ = ["SimuNetAgent", "Settings"]