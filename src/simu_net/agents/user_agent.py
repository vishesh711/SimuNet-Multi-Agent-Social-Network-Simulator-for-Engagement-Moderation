"""User agent implementation with configurable behaviors."""

import asyncio
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from .base import SimuNetAgent
from ..data.models import (
    UserAgent as UserAgentModel,
    PersonaType,
    ContentAgent as ContentAgentModel,
    InteractionEvent,
    NetworkConnection
)
from ..events import AgentEvent


class UserAgent(SimuNetAgent):
    """User agent with configurable persona-based behaviors."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        persona_type: PersonaType = PersonaType.CASUAL,
        behavior_params: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Initialize user agent.
        
        Args:
            agent_id: Unique agent identifier
            persona_type: User persona type
            behavior_params: Custom behavior parameters
            **kwargs: Additional agent parameters
        """
        super().__init__(agent_id=agent_id, **kwargs)
        
        self.persona_type = persona_type
        self.behavior_params = behavior_params or {}
        
        # Initialize user data model
        self.user_data = UserAgentModel(
            agent_id=self.agent_id,
            persona_type=persona_type,
            behavior_params=self.behavior_params,
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow()
        )
        
        # Set default behavior parameters based on persona
        self._initialize_persona_behaviors()
        
        # Internal state
        self._content_creation_cooldown = 0.0
        self._last_content_creation = datetime.utcnow()
        self._engagement_targets: List[str] = []
        self._network_connections: List[str] = []
        
        self.logger.info(
            "User agent initialized",
            persona_type=persona_type.value,
            behavior_params=self.behavior_params
        )
    
    def _initialize_persona_behaviors(self) -> None:
        """Initialize behavior parameters based on persona type."""
        # Default parameters for each persona type
        persona_defaults = {
            PersonaType.CASUAL: {
                "posting_frequency": 0.5,  # posts per day
                "engagement_likelihood": 0.1,  # probability to engage with content
                "sharing_likelihood": 0.05,  # probability to share content
                "commenting_likelihood": 0.03,  # probability to comment
                "misinformation_susceptibility": 0.2,  # susceptibility to misinformation
                "influence_weight": 0.1,  # how much influence affects behavior
                "network_growth_rate": 0.1,  # rate of making new connections
                "content_preference_strength": 0.3,  # how much content preferences matter
            },
            PersonaType.INFLUENCER: {
                "posting_frequency": 3.0,
                "engagement_likelihood": 0.3,
                "sharing_likelihood": 0.2,
                "commenting_likelihood": 0.15,
                "misinformation_susceptibility": 0.1,
                "influence_weight": 0.4,
                "network_growth_rate": 0.3,
                "content_preference_strength": 0.5,
            },
            PersonaType.BOT: {
                "posting_frequency": 10.0,
                "engagement_likelihood": 0.8,
                "sharing_likelihood": 0.6,
                "commenting_likelihood": 0.4,
                "misinformation_susceptibility": 0.05,
                "influence_weight": 0.1,
                "network_growth_rate": 0.5,
                "content_preference_strength": 0.2,
            },
            PersonaType.ACTIVIST: {
                "posting_frequency": 2.0,
                "engagement_likelihood": 0.4,
                "sharing_likelihood": 0.3,
                "commenting_likelihood": 0.25,
                "misinformation_susceptibility": 0.05,
                "influence_weight": 0.3,
                "network_growth_rate": 0.2,
                "content_preference_strength": 0.8,
            }
        }
        
        # Set defaults for this persona type
        defaults = persona_defaults.get(self.persona_type, persona_defaults[PersonaType.CASUAL])
        
        # Merge with custom parameters (custom parameters take precedence)
        for key, default_value in defaults.items():
            if key not in self.behavior_params:
                self.behavior_params[key] = default_value
        
        # Update user data model
        self.user_data.behavior_params = self.behavior_params
        self.user_data.posting_frequency = self.behavior_params["posting_frequency"]
        self.user_data.engagement_likelihood = self.behavior_params["engagement_likelihood"]
        self.user_data.misinformation_susceptibility = self.behavior_params["misinformation_susceptibility"]
    
    async def _process_tick(self) -> None:
        """Process a single agent tick."""
        try:
            # Update last active timestamp
            self.user_data.last_active = datetime.utcnow()
            
            # Decide what actions to take this tick
            actions = await self._decide_actions()
            
            # Execute actions
            for action in actions:
                await self._execute_action(action)
            
            # Update cooldowns
            self._update_cooldowns()
            
        except Exception as e:
            self.logger.error("Error in user agent tick", error=str(e))
    
    async def _decide_actions(self) -> List[Dict[str, Any]]:
        """Decide what actions to take this tick.
        
        Returns:
            List of action dictionaries
        """
        actions = []
        
        # Content creation decision
        if await self._should_create_content():
            actions.append({"type": "create_content"})
        
        # Engagement decisions (check available content)
        if await self._should_engage_with_content():
            actions.append({"type": "engage_content"})
        
        # Network growth decision
        if await self._should_grow_network():
            actions.append({"type": "grow_network"})
        
        return actions
    
    async def _should_create_content(self) -> bool:
        """Determine if agent should create content this tick.
        
        Returns:
            True if should create content
        """
        # Check cooldown
        if self._content_creation_cooldown > 0:
            return False
        
        # Calculate probability based on posting frequency
        # Convert daily frequency to per-tick probability
        tick_interval = self._get_tick_interval()
        daily_ticks = 86400 / tick_interval  # seconds in a day / tick interval
        tick_probability = self.behavior_params["posting_frequency"] / daily_ticks
        
        # Add some randomness
        return random.random() < tick_probability
    
    async def _should_engage_with_content(self) -> bool:
        """Determine if agent should engage with content this tick.
        
        Returns:
            True if should engage with content
        """
        # Base probability from behavior parameters
        base_probability = self.behavior_params["engagement_likelihood"]
        
        # Adjust based on available content (simulated for now)
        # In a real implementation, this would check for available content
        content_availability_factor = 0.5  # Assume moderate content availability
        
        adjusted_probability = base_probability * content_availability_factor
        return random.random() < adjusted_probability
    
    async def _should_grow_network(self) -> bool:
        """Determine if agent should try to grow network this tick.
        
        Returns:
            True if should try to grow network
        """
        # Less frequent than other actions
        base_probability = self.behavior_params["network_growth_rate"] / 100  # Much lower frequency
        return random.random() < base_probability
    
    async def _execute_action(self, action: Dict[str, Any]) -> None:
        """Execute a specific action.
        
        Args:
            action: Action dictionary with type and parameters
        """
        action_type = action["type"]
        
        try:
            if action_type == "create_content":
                await self._create_content()
            elif action_type == "engage_content":
                await self._engage_with_content()
            elif action_type == "grow_network":
                await self._grow_network()
            else:
                self.logger.warning("Unknown action type", action_type=action_type)
                
        except Exception as e:
            self.logger.error("Error executing action", action_type=action_type, error=str(e))
    
    async def _create_content(self) -> None:
        """Create new content."""
        # Generate content based on persona type
        content_text = self._generate_content_text()
        
        # Create content event
        content_data = {
            "content_id": str(uuid.uuid4()),
            "text_content": content_text,
            "created_by": self.agent_id,
            "created_at": datetime.utcnow().isoformat(),
            "persona_type": self.persona_type.value
        }
        
        # Publish content creation event
        await self.publish_event(
            event_type="content_created",
            payload=content_data
        )
        
        # Set cooldown based on persona
        self._set_content_creation_cooldown()
        
        self.logger.info("Content created", content_id=content_data["content_id"])
    
    def _generate_content_text(self) -> str:
        """Generate content text based on persona type.
        
        Returns:
            Generated content text
        """
        # Content templates by persona type
        content_templates = {
            PersonaType.CASUAL: [
                "Just had an amazing day! 😊",
                "Sharing some thoughts about life...",
                "Check out this interesting thing I found!",
                "Having a great time with friends today",
                "Beautiful weather outside! ☀️"
            ],
            PersonaType.INFLUENCER: [
                "Excited to share my latest project with you all! 🚀",
                "Here's my take on the trending topic everyone's talking about",
                "Thank you for all the support! You're amazing! 💖",
                "New collaboration announcement coming soon...",
                "Behind the scenes of my creative process ✨"
            ],
            PersonaType.BOT: [
                "Breaking: Latest news update on current events",
                "Automated content sharing: [LINK] Check this out!",
                "Daily reminder: Stay informed and engaged",
                "Trending now: #hashtag content update",
                "System generated post: Content distribution active"
            ],
            PersonaType.ACTIVIST: [
                "We need to raise awareness about this important issue! 📢",
                "Join us in making a difference - every voice matters!",
                "Educational thread about social justice and equality",
                "Call to action: Contact your representatives today!",
                "Sharing resources for community organizing and change"
            ]
        }
        
        templates = content_templates.get(self.persona_type, content_templates[PersonaType.CASUAL])
        return random.choice(templates)
    
    def _set_content_creation_cooldown(self) -> None:
        """Set cooldown period for content creation."""
        # Base cooldown in seconds, adjusted by persona
        base_cooldown = 3600  # 1 hour
        
        # Adjust based on posting frequency
        frequency_factor = 1.0 / max(self.behavior_params["posting_frequency"], 0.1)
        cooldown = base_cooldown * frequency_factor
        
        # Add some randomness (±20%)
        randomness = random.uniform(0.8, 1.2)
        self._content_creation_cooldown = cooldown * randomness
        
        self._last_content_creation = datetime.utcnow()
    
    async def _engage_with_content(self) -> None:
        """Engage with available content."""
        # In a real implementation, this would query for available content
        # For now, we'll simulate engagement with random content
        
        # Simulate finding content to engage with
        simulated_content_id = f"content-{random.randint(1000, 9999)}"
        
        # Decide type of engagement
        engagement_type = self._decide_engagement_type()
        
        # Create interaction event
        interaction_data = {
            "interaction_id": str(uuid.uuid4()),
            "user_id": self.agent_id,
            "content_id": simulated_content_id,
            "interaction_type": engagement_type,
            "timestamp": datetime.utcnow().isoformat(),
            "persona_type": self.persona_type.value
        }
        
        # Publish interaction event
        await self.publish_event(
            event_type="content_interaction",
            payload=interaction_data
        )
        
        self.logger.debug(
            "Content engagement",
            content_id=simulated_content_id,
            interaction_type=engagement_type
        )
    
    def _decide_engagement_type(self) -> str:
        """Decide what type of engagement to perform.
        
        Returns:
            Engagement type string
        """
        # Probabilities for different engagement types
        like_prob = 0.6
        share_prob = self.behavior_params["sharing_likelihood"] / self.behavior_params["engagement_likelihood"]
        comment_prob = self.behavior_params["commenting_likelihood"] / self.behavior_params["engagement_likelihood"]
        
        # Normalize probabilities
        total_prob = like_prob + share_prob + comment_prob
        like_prob /= total_prob
        share_prob /= total_prob
        comment_prob /= total_prob
        
        # Choose engagement type
        rand = random.random()
        if rand < like_prob:
            return "like"
        elif rand < like_prob + share_prob:
            return "share"
        else:
            return "comment"
    
    async def _grow_network(self) -> None:
        """Attempt to grow social network."""
        # Simulate finding a user to connect with
        target_user_id = f"user-{random.randint(1000, 9999)}"
        
        # Create connection event
        connection_data = {
            "connection_id": str(uuid.uuid4()),
            "user_a_id": self.agent_id,
            "user_b_id": target_user_id,
            "connection_type": "follow",
            "strength": random.uniform(0.1, 1.0),
            "created_at": datetime.utcnow().isoformat(),
            "initiator_persona": self.persona_type.value
        }
        
        # Publish network growth event
        await self.publish_event(
            event_type="network_connection_created",
            payload=connection_data
        )
        
        # Update internal network list
        if target_user_id not in self._network_connections:
            self._network_connections.append(target_user_id)
            self.user_data.network_connections = self._network_connections
            self.user_data.following_count = len(self._network_connections)
        
        self.logger.debug("Network connection created", target_user=target_user_id)
    
    def _update_cooldowns(self) -> None:
        """Update internal cooldowns."""
        tick_interval = self._get_tick_interval()
        
        # Update content creation cooldown
        if self._content_creation_cooldown > 0:
            self._content_creation_cooldown -= tick_interval
            if self._content_creation_cooldown < 0:
                self._content_creation_cooldown = 0
    
    async def _on_event_received(self, event: AgentEvent) -> None:
        """Handle received events.
        
        Args:
            event: The received event
        """
        try:
            if event.event_type == "content_created":
                await self._handle_content_created(event)
            elif event.event_type == "content_interaction":
                await self._handle_content_interaction(event)
            elif event.event_type == "network_connection_created":
                await self._handle_network_connection(event)
            
        except Exception as e:
            self.logger.error("Error handling event", event_type=event.event_type, error=str(e))
    
    async def _handle_content_created(self, event: AgentEvent) -> None:
        """Handle content creation events from other agents.
        
        Args:
            event: Content creation event
        """
        content_data = event.payload
        creator_id = content_data.get("created_by")
        
        # Don't react to own content
        if creator_id == self.agent_id:
            return
        
        # Decide whether to engage with this content
        # Consider factors like network connections, content type, etc.
        should_engage = await self._should_engage_with_specific_content(content_data)
        
        if should_engage:
            # Add to engagement targets for next tick
            content_id = content_data.get("content_id")
            if content_id and content_id not in self._engagement_targets:
                self._engagement_targets.append(content_id)
    
    async def _should_engage_with_specific_content(self, content_data: Dict[str, Any]) -> bool:
        """Determine if should engage with specific content.
        
        Args:
            content_data: Content data from event
            
        Returns:
            True if should engage with this content
        """
        creator_id = content_data.get("created_by")
        creator_persona = content_data.get("persona_type")
        
        # Base engagement probability
        base_prob = self.behavior_params["engagement_likelihood"]
        
        # Adjust based on network connection
        if creator_id in self._network_connections:
            base_prob *= 2.0  # More likely to engage with connected users
        
        # Adjust based on persona compatibility
        persona_compatibility = self._get_persona_compatibility(creator_persona)
        base_prob *= persona_compatibility
        
        return random.random() < min(base_prob, 1.0)
    
    def _get_persona_compatibility(self, other_persona: Optional[str]) -> float:
        """Get compatibility factor with another persona type.
        
        Args:
            other_persona: Other agent's persona type
            
        Returns:
            Compatibility factor (0.1 to 2.0)
        """
        if not other_persona:
            return 1.0
        
        # Compatibility matrix
        compatibility = {
            PersonaType.CASUAL: {
                PersonaType.CASUAL: 1.2,
                PersonaType.INFLUENCER: 1.5,
                PersonaType.BOT: 0.3,
                PersonaType.ACTIVIST: 0.8
            },
            PersonaType.INFLUENCER: {
                PersonaType.CASUAL: 1.0,
                PersonaType.INFLUENCER: 1.3,
                PersonaType.BOT: 0.5,
                PersonaType.ACTIVIST: 1.1
            },
            PersonaType.BOT: {
                PersonaType.CASUAL: 0.8,
                PersonaType.INFLUENCER: 1.0,
                PersonaType.BOT: 1.5,
                PersonaType.ACTIVIST: 0.7
            },
            PersonaType.ACTIVIST: {
                PersonaType.CASUAL: 0.9,
                PersonaType.INFLUENCER: 1.2,
                PersonaType.BOT: 0.2,
                PersonaType.ACTIVIST: 1.8
            }
        }
        
        try:
            other_persona_enum = PersonaType(other_persona)
            return compatibility.get(self.persona_type, {}).get(other_persona_enum, 1.0)
        except ValueError:
            return 1.0
    
    async def _handle_content_interaction(self, event: AgentEvent) -> None:
        """Handle content interaction events.
        
        Args:
            event: Content interaction event
        """
        interaction_data = event.payload
        user_id = interaction_data.get("user_id")
        content_id = interaction_data.get("content_id")
        interaction_type = interaction_data.get("interaction_type")
        
        # If someone interacted with our content, we might respond
        # This would require tracking our own content IDs
        # For now, just log the interaction
        self.logger.debug(
            "Observed content interaction",
            user_id=user_id,
            content_id=content_id,
            interaction_type=interaction_type
        )
    
    async def _handle_network_connection(self, event: AgentEvent) -> None:
        """Handle network connection events.
        
        Args:
            event: Network connection event
        """
        connection_data = event.payload
        user_a_id = connection_data.get("user_a_id")
        user_b_id = connection_data.get("user_b_id")
        
        # If someone connected to us, update our follower count
        if user_b_id == self.agent_id:
            self.user_data.follower_count += 1
            self.logger.debug("Gained new follower", follower_id=user_a_id)
        
        # If we connected to someone, it's already handled in _grow_network
    
    def _get_tick_interval(self) -> float:
        """Get agent tick interval in seconds."""
        # Base interval, can be adjusted based on persona
        base_interval = 5.0  # 5 seconds
        
        # Bots tick more frequently
        if self.persona_type == PersonaType.BOT:
            return base_interval * 0.5
        elif self.persona_type == PersonaType.INFLUENCER:
            return base_interval * 0.8
        else:
            return base_interval
    
    def _get_subscribed_events(self) -> List[str]:
        """Get list of event types this agent subscribes to."""
        return [
            "content_created",
            "content_interaction",
            "network_connection_created",
            "moderation_action"
        ]
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics and current state.
        
        Returns:
            Dictionary with user statistics
        """
        return {
            "agent_id": self.agent_id,
            "persona_type": self.persona_type.value,
            "behavior_params": self.behavior_params,
            "follower_count": self.user_data.follower_count,
            "following_count": self.user_data.following_count,
            "influence_score": self.user_data.influence_score,
            "credibility_score": self.user_data.credibility_score,
            "network_connections": len(self._network_connections),
            "content_creation_cooldown": self._content_creation_cooldown,
            "last_active": self.user_data.last_active.isoformat(),
            "is_active": self.is_active,
            "is_running": self.is_running
        }
    
    def update_behavior_params(self, new_params: Dict[str, float]) -> None:
        """Update behavior parameters.
        
        Args:
            new_params: New behavior parameters to merge
        """
        self.behavior_params.update(new_params)
        self.user_data.behavior_params = self.behavior_params
        
        # Update specific fields in user data
        if "posting_frequency" in new_params:
            self.user_data.posting_frequency = new_params["posting_frequency"]
        if "engagement_likelihood" in new_params:
            self.user_data.engagement_likelihood = new_params["engagement_likelihood"]
        if "misinformation_susceptibility" in new_params:
            self.user_data.misinformation_susceptibility = new_params["misinformation_susceptibility"]
        
        self.logger.info("Behavior parameters updated", new_params=new_params)