"""Agent registry for discovery and management."""

import asyncio
from typing import Any, Dict, List, Optional, Type

import structlog

from .base import SimuNetAgent


class AgentRegistry:
    """Registry for managing and discovering agents."""
    
    def __init__(self):
        """Initialize the agent registry."""
        self.logger = structlog.get_logger().bind(component="AgentRegistry")
        
        # Registry of active agents: agent_id -> agent instance
        self._agents: Dict[str, SimuNetAgent] = {}
        
        # Agent type registry: agent_type -> agent class
        self._agent_types: Dict[str, Type[SimuNetAgent]] = {}
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    def register_agent_type(
        self,
        agent_type: str,
        agent_class: Type[SimuNetAgent]
    ) -> None:
        """Register an agent type.
        
        Args:
            agent_type: Name of the agent type
            agent_class: Agent class to register
        """
        self._agent_types[agent_type] = agent_class
        self.logger.info("Agent type registered", agent_type=agent_type)
    
    async def create_agent(
        self,
        agent_type: str,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> SimuNetAgent:
        """Create and register a new agent.
        
        Args:
            agent_type: Type of agent to create
            agent_id: Optional agent ID (auto-generated if not provided)
            **kwargs: Additional arguments for agent initialization
            
        Returns:
            Created agent instance
            
        Raises:
            ValueError: If agent type is not registered
        """
        if agent_type not in self._agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = self._agent_types[agent_type]
        agent = agent_class(agent_id=agent_id, **kwargs)
        
        async with self._lock:
            self._agents[agent.agent_id] = agent
        
        self.logger.info(
            "Agent created",
            agent_id=agent.agent_id,
            agent_type=agent_type
        )
        
        return agent
    
    async def register_agent(self, agent: SimuNetAgent) -> None:
        """Register an existing agent.
        
        Args:
            agent: Agent instance to register
        """
        async with self._lock:
            self._agents[agent.agent_id] = agent
        
        self.logger.info(
            "Agent registered",
            agent_id=agent.agent_id,
            agent_type=agent.state.agent_type
        )
    
    async def unregister_agent(self, agent_id: str) -> Optional[SimuNetAgent]:
        """Unregister an agent.
        
        Args:
            agent_id: ID of agent to unregister
            
        Returns:
            Unregistered agent instance or None if not found
        """
        async with self._lock:
            agent = self._agents.pop(agent_id, None)
        
        if agent:
            self.logger.info("Agent unregistered", agent_id=agent_id)
        else:
            self.logger.warning("Agent not found for unregistration", agent_id=agent_id)
        
        return agent
    
    async def get_agent(self, agent_id: str) -> Optional[SimuNetAgent]:
        """Get an agent by ID.
        
        Args:
            agent_id: ID of agent to retrieve
            
        Returns:
            Agent instance or None if not found
        """
        async with self._lock:
            return self._agents.get(agent_id)
    
    async def get_agents_by_type(self, agent_type: str) -> List[SimuNetAgent]:
        """Get all agents of a specific type.
        
        Args:
            agent_type: Type of agents to retrieve
            
        Returns:
            List of agent instances
        """
        async with self._lock:
            return [
                agent for agent in self._agents.values()
                if agent.state.agent_type == agent_type
            ]
    
    async def get_all_agents(self) -> List[SimuNetAgent]:
        """Get all registered agents.
        
        Returns:
            List of all agent instances
        """
        async with self._lock:
            return list(self._agents.values())
    
    async def get_active_agents(self) -> List[SimuNetAgent]:
        """Get all active agents.
        
        Returns:
            List of active agent instances
        """
        async with self._lock:
            return [
                agent for agent in self._agents.values()
                if agent.is_active
            ]
    
    async def start_agent(self, agent_id: str) -> bool:
        """Start an agent.
        
        Args:
            agent_id: ID of agent to start
            
        Returns:
            True if agent was started, False if not found
        """
        agent = await self.get_agent(agent_id)
        if agent:
            await agent.start()
            return True
        return False
    
    async def stop_agent(self, agent_id: str) -> bool:
        """Stop an agent.
        
        Args:
            agent_id: ID of agent to stop
            
        Returns:
            True if agent was stopped, False if not found
        """
        agent = await self.get_agent(agent_id)
        if agent:
            await agent.stop()
            return True
        return False
    
    async def start_all_agents(self) -> None:
        """Start all registered agents."""
        agents = await self.get_all_agents()
        
        start_tasks = []
        for agent in agents:
            if not agent.is_running:
                start_tasks.append(agent.start())
        
        if start_tasks:
            await asyncio.gather(*start_tasks, return_exceptions=True)
            self.logger.info("All agents started", count=len(start_tasks))
    
    async def stop_all_agents(self) -> None:
        """Stop all registered agents."""
        agents = await self.get_all_agents()
        
        stop_tasks = []
        for agent in agents:
            if agent.is_running:
                stop_tasks.append(agent.stop())
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
            self.logger.info("All agents stopped", count=len(stop_tasks))
    
    async def cleanup_stopped_agents(self) -> int:
        """Remove stopped agents from registry.
        
        Returns:
            Number of agents removed
        """
        async with self._lock:
            stopped_agents = [
                agent_id for agent_id, agent in self._agents.items()
                if not agent.is_running
            ]
            
            for agent_id in stopped_agents:
                del self._agents[agent_id]
        
        if stopped_agents:
            self.logger.info("Cleaned up stopped agents", count=len(stopped_agents))
        
        return len(stopped_agents)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        total_agents = len(self._agents)
        active_agents = sum(1 for agent in self._agents.values() if agent.is_active)
        running_agents = sum(1 for agent in self._agents.values() if agent.is_running)
        
        agent_types = {}
        for agent in self._agents.values():
            agent_type = agent.state.agent_type
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "running_agents": running_agents,
            "agent_types": agent_types,
            "registered_types": list(self._agent_types.keys())
        }