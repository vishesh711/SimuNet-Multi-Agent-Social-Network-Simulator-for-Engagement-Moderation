"""Base agent class for SimuNet simulation."""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field

from ..events import EventManager, AgentEvent


class AgentState(BaseModel):
    """Agent state model."""
    
    agent_id: str
    agent_type: str
    created_at: datetime
    last_active: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SimuNetAgent(ABC):
    """Abstract base class for all SimuNet agents."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        event_manager: Optional[EventManager] = None,
        **kwargs
    ):
        """Initialize the agent.
        
        Args:
            agent_id: Unique identifier for the agent
            event_manager: Event manager for agent communication
            **kwargs: Additional agent-specific parameters
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.event_manager = event_manager
        self.logger = structlog.get_logger().bind(agent_id=self.agent_id)
        
        # Initialize state
        self.state = AgentState(
            agent_id=self.agent_id,
            agent_type=self.__class__.__name__,
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow()
        )
        
        # Agent lifecycle flags
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        self.logger.info("Agent initialized", agent_type=self.state.agent_type)
    
    @property
    def is_running(self) -> bool:
        """Check if agent is currently running."""
        return self._running
    
    @property
    def is_active(self) -> bool:
        """Check if agent is active."""
        return self.state.is_active
    
    async def start(self) -> None:
        """Start the agent."""
        if self._running:
            self.logger.warning("Agent already running")
            return
        
        self._running = True
        self.state.is_active = True
        self.state.last_active = datetime.utcnow()
        
        self.logger.info("Starting agent")
        
        # Subscribe to events if event manager is available
        if self.event_manager:
            await self._subscribe_to_events()
        
        # Start agent-specific initialization
        await self._on_start()
        
        # Start the main agent loop
        asyncio.create_task(self._agent_loop())
    
    async def stop(self) -> None:
        """Stop the agent."""
        if not self._running:
            self.logger.warning("Agent not running")
            return
        
        self.logger.info("Stopping agent")
        
        self._running = False
        self.state.is_active = False
        self._shutdown_event.set()
        
        # Agent-specific cleanup
        await self._on_stop()
        
        # Unsubscribe from events
        if self.event_manager:
            await self._unsubscribe_from_events()
        
        self.logger.info("Agent stopped")
    
    async def pause(self) -> None:
        """Pause the agent."""
        self.state.is_active = False
        self.logger.info("Agent paused")
    
    async def resume(self) -> None:
        """Resume the agent."""
        self.state.is_active = True
        self.state.last_active = datetime.utcnow()
        self.logger.info("Agent resumed")
    
    async def publish_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        target_agents: Optional[List[str]] = None
    ) -> None:
        """Publish an event to other agents.
        
        Args:
            event_type: Type of event to publish
            payload: Event payload data
            target_agents: List of target agent IDs (None for broadcast)
        """
        if not self.event_manager:
            self.logger.warning("No event manager available")
            return
        
        event = AgentEvent(
            event_type=event_type,
            source_agent=self.agent_id,
            target_agents=target_agents or [],
            payload=payload,
            timestamp=datetime.utcnow(),
            correlation_id=str(uuid.uuid4())
        )
        
        await self.event_manager.publish(event)
        self.logger.debug("Event published", event_type=event_type)
    
    async def _agent_loop(self) -> None:
        """Main agent processing loop."""
        while self._running:
            try:
                if self.state.is_active:
                    await self._process_tick()
                    self.state.last_active = datetime.utcnow()
                
                # Wait for next tick or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self._get_tick_interval()
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal tick timeout
                    
            except Exception as e:
                self.logger.error("Error in agent loop", error=str(e))
                await asyncio.sleep(1)  # Brief pause before retry
    
    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        event_types = self._get_subscribed_events()
        for event_type in event_types:
            await self.event_manager.subscribe(event_type, self._handle_event)
        
        self.logger.debug("Subscribed to events", event_types=event_types)
    
    async def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from events."""
        event_types = self._get_subscribed_events()
        for event_type in event_types:
            await self.event_manager.unsubscribe(event_type, self._handle_event)
        
        self.logger.debug("Unsubscribed from events", event_types=event_types)
    
    async def _handle_event(self, event: AgentEvent) -> None:
        """Handle incoming events.
        
        Args:
            event: The received event
        """
        # Skip events from self
        if event.source_agent == self.agent_id:
            return
        
        # Check if event is targeted to this agent
        if event.target_agents and self.agent_id not in event.target_agents:
            return
        
        self.logger.debug(
            "Handling event",
            event_type=event.event_type,
            source=event.source_agent
        )
        
        try:
            await self._on_event_received(event)
        except Exception as e:
            self.logger.error(
                "Error handling event",
                event_type=event.event_type,
                error=str(e)
            )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        return self.state.model_dump()
    
    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update agent metadata."""
        self.state.metadata.update(metadata)
        self.logger.debug("Metadata updated", metadata=metadata)
    
    # Abstract methods that subclasses must implement
    
    @abstractmethod
    async def _process_tick(self) -> None:
        """Process a single agent tick.
        
        This method is called periodically when the agent is active.
        Subclasses should implement their main logic here.
        """
        pass
    
    @abstractmethod
    def _get_tick_interval(self) -> float:
        """Get the tick interval in seconds.
        
        Returns:
            Tick interval in seconds
        """
        pass
    
    @abstractmethod
    def _get_subscribed_events(self) -> List[str]:
        """Get list of event types this agent subscribes to.
        
        Returns:
            List of event type strings
        """
        pass
    
    # Optional lifecycle hooks
    
    async def _on_start(self) -> None:
        """Called when agent starts. Override for custom initialization."""
        pass
    
    async def _on_stop(self) -> None:
        """Called when agent stops. Override for custom cleanup."""
        pass
    
    async def _on_event_received(self, event: AgentEvent) -> None:
        """Called when an event is received. Override for custom event handling.
        
        Args:
            event: The received event
        """
        pass