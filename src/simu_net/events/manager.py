"""Event manager for Redis-based agent communication."""

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional

import redis.asyncio as redis
import structlog
from pydantic import ValidationError

from .models import AgentEvent


class EventManager:
    """Manages event publishing and subscription using Redis pub/sub."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize the event manager.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.logger = structlog.get_logger().bind(component="EventManager")
        
        # Event handlers: event_type -> list of handler functions
        self._handlers: Dict[str, List[Callable[[AgentEvent], Any]]] = {}
        
        # Background task for message processing
        self._message_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the event manager."""
        if self._running:
            self.logger.warning("Event manager already running")
            return
        
        try:
            # Connect to Redis
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Create pub/sub client
            self.pubsub = self.redis_client.pubsub()
            
            self._running = True
            
            # Start message processing task
            self._message_task = asyncio.create_task(self._process_messages())
            
            self.logger.info("Event manager started", redis_url=self.redis_url)
            
        except Exception as e:
            self.logger.error("Failed to start event manager", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop the event manager."""
        if not self._running:
            self.logger.warning("Event manager not running")
            return
        
        self._running = False
        
        # Cancel message processing task
        if self._message_task:
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass
        
        # Close pub/sub connection
        if self.pubsub:
            await self.pubsub.close()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        self.logger.info("Event manager stopped")
    
    async def publish(self, event: AgentEvent) -> None:
        """Publish an event to Redis.
        
        Args:
            event: The event to publish
        """
        if not self._running or not self.redis_client:
            self.logger.warning("Event manager not running")
            return
        
        try:
            # Serialize event to JSON
            event_data = event.model_dump_json()
            
            # Publish to Redis channel
            channel = f"simu_net:events:{event.event_type}"
            await self.redis_client.publish(channel, event_data)
            
            self.logger.debug(
                "Event published",
                event_type=event.event_type,
                source=event.source_agent,
                channel=channel
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to publish event",
                event_type=event.event_type,
                error=str(e)
            )
    
    async def subscribe(
        self,
        event_type: str,
        handler: Callable[[AgentEvent], Any]
    ) -> None:
        """Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Function to handle received events
        """
        if not self._running or not self.pubsub:
            self.logger.warning("Event manager not running")
            return
        
        try:
            # Add handler to internal registry
            if event_type not in self._handlers:
                self._handlers[event_type] = []
                
                # Subscribe to Redis channel
                channel = f"simu_net:events:{event_type}"
                await self.pubsub.subscribe(channel)
                
                self.logger.debug("Subscribed to channel", channel=channel)
            
            self._handlers[event_type].append(handler)
            
            self.logger.debug(
                "Handler registered",
                event_type=event_type,
                handler_count=len(self._handlers[event_type])
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to subscribe to events",
                event_type=event_type,
                error=str(e)
            )
    
    async def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[AgentEvent], Any]
    ) -> None:
        """Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of events to unsubscribe from
            handler: Handler function to remove
        """
        if event_type not in self._handlers:
            return
        
        try:
            # Remove handler from registry
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
            
            # If no more handlers, unsubscribe from Redis channel
            if not self._handlers[event_type]:
                del self._handlers[event_type]
                
                if self.pubsub:
                    channel = f"simu_net:events:{event_type}"
                    await self.pubsub.unsubscribe(channel)
                    
                    self.logger.debug("Unsubscribed from channel", channel=channel)
            
            self.logger.debug(
                "Handler unregistered",
                event_type=event_type,
                remaining_handlers=len(self._handlers.get(event_type, []))
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to unsubscribe from events",
                event_type=event_type,
                error=str(e)
            )
    
    async def _process_messages(self) -> None:
        """Process incoming messages from Redis pub/sub."""
        if not self.pubsub:
            return
        
        self.logger.info("Started message processing")
        
        try:
            async for message in self.pubsub.listen():
                if not self._running:
                    break
                
                if message["type"] != "message":
                    continue
                
                try:
                    # Parse channel to get event type
                    channel = message["channel"].decode("utf-8")
                    if not channel.startswith("simu_net:events:"):
                        continue
                    
                    event_type = channel.replace("simu_net:events:", "")
                    
                    # Deserialize event
                    event_data = json.loads(message["data"])
                    event = AgentEvent(**event_data)
                    
                    # Call registered handlers
                    if event_type in self._handlers:
                        for handler in self._handlers[event_type]:
                            try:
                                if asyncio.iscoroutinefunction(handler):
                                    await handler(event)
                                else:
                                    handler(event)
                            except Exception as e:
                                self.logger.error(
                                    "Handler error",
                                    event_type=event_type,
                                    error=str(e)
                                )
                    
                    self.logger.debug(
                        "Event processed",
                        event_type=event_type,
                        handlers_called=len(self._handlers.get(event_type, []))
                    )
                    
                except (json.JSONDecodeError, ValidationError) as e:
                    self.logger.error("Failed to parse event", error=str(e))
                except Exception as e:
                    self.logger.error("Error processing message", error=str(e))
                    
        except asyncio.CancelledError:
            self.logger.info("Message processing cancelled")
        except Exception as e:
            self.logger.error("Error in message processing", error=str(e))
        finally:
            self.logger.info("Message processing stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event manager statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "running": self._running,
            "subscribed_events": list(self._handlers.keys()),
            "total_handlers": sum(len(handlers) for handlers in self._handlers.values()),
            "redis_connected": self.redis_client is not None
        }