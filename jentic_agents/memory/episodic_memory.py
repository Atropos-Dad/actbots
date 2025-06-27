"""
Episodic memory implementation for storing personal experiences and events.

Episodic memory stores autobiographical memories, personal experiences, and
specific events that occurred at particular times and places. This includes:
- Personal experiences ("I went to Paris last summer")
- Specific events with context ("Meeting with John on Monday at 2pm")
- Temporal sequences and storylines
- Contextual information (time, place, emotions, people involved)
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from .base_memory import BaseMemory
from ..utils.logger import get_logger
from ..utils.block_timer import Timer

logger = get_logger(__name__)


class EpisodicMemory(BaseMemory):
    """
    Episodic memory for storing personal experiences and events.
    
    This memory system is designed to store:
    - Personal experiences and autobiographical memories
    - Specific events with temporal and spatial context
    - Emotional and sensory details
    - Sequential narratives and storylines
    - Contextual information (who, what, when, where, why, how)
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize episodic memory.
        
        Args:
            storage_path: Optional path to persist episodes to disk
        """
        logger.info("Initializing EpisodicMemory")
        
        self._episodes: Dict[str, Dict[str, Any]] = {}
        self._storage_path = storage_path
        self._episode_sequence = 0  # For ordering episodes
        
        # Load existing episodes if storage path exists
        if self._storage_path and os.path.exists(self._storage_path):
            with Timer("Load episodes from disk"):
                self._load_from_disk()
        
        logger.info("EpisodicMemory initialized successfully")
    
    def store(self, key: str, value: Any) -> None:
        """
        Store an episode under the given key.
        
        Args:
            key: Unique identifier for the episode
            value: Episode data (experience, event, etc.)
        """
        logger.debug(f"Storing episode: {key}")
        
        with Timer(f"Store episode: {key}"):
            # Create episode with rich metadata
            episode = {
                "value": value,
                "sequence": self._episode_sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": self._extract_context(value),
                "emotions": self._extract_emotions(value),
                "people": self._extract_people(value),
                "locations": self._extract_locations(value),
                "tags": self._extract_tags(value),
                "recalled_count": 0,
                "last_recalled": None,
                "vividness": self._assess_vividness(value),
                "importance": self._assess_importance(value)
            }
            
            self._episodes[key] = episode
            self._episode_sequence += 1
            
            # Persist to disk if storage path is configured
            if self._storage_path:
                self._save_to_disk()
        
        logger.debug(f"Stored episode: {key}")
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve an episode by key.
        
        Args:
            key: Unique identifier for the episode
            
        Returns:
            Stored episode, or None if key not found
        """
        logger.debug(f"Retrieving episode: {key}")
        
        with Timer(f"Retrieve episode: {key}"):
            if key not in self._episodes:
                logger.debug(f"Episode not found: {key}")
                return None
            
            # Update recall metadata (affects memory strength)
            episode = self._episodes[key]
            episode["recalled_count"] += 1
            episode["last_recalled"] = datetime.now(timezone.utc).isoformat()
            
            # Strengthen memory with each recall
            episode["vividness"] = min(1.0, episode["vividness"] + 0.1)
            
            # Persist updated metadata if storage is configured
            if self._storage_path:
                self._save_to_disk()
            
            return episode["value"]
    
    def delete(self, key: str) -> bool:
        """
        Delete an episode.
        
        Args:
            key: Unique identifier for the episode to delete
            
        Returns:
            True if episode was deleted, False if key not found
        """
        logger.debug(f"Deleting episode: {key}")
        
        with Timer(f"Delete episode: {key}"):
            if key not in self._episodes:
                logger.debug(f"Episode not found for deletion: {key}")
                return False
            
            del self._episodes[key]
            
            # Persist changes if storage is configured
            if self._storage_path:
                self._save_to_disk()
            
            logger.debug(f"Deleted episode: {key}")
            return True
    
    def clear(self) -> None:
        """Clear all episodes."""
        logger.info("Clearing all episodes")
        
        with Timer("Clear all episodes"):
            self._episodes.clear()
            self._episode_sequence = 0
            
            # Clear persisted storage if configured
            if self._storage_path and os.path.exists(self._storage_path):
                os.remove(self._storage_path)
        
        logger.info("Cleared all episodes")
    
    def keys(self) -> List[str]:
        """
        Get all episode keys.
        
        Returns:
            List of all keys in episodic memory
        """
        return list(self._episodes.keys())
    
    def get_episodes_by_timeframe(self, start_time: str, end_time: str) -> Dict[str, Any]:
        """
        Get episodes within a specific timeframe.
        
        Args:
            start_time: ISO format start time
            end_time: ISO format end time
            
        Returns:
            Dictionary of episodes within the timeframe
        """
        logger.debug(f"Retrieving episodes from {start_time} to {end_time}")
        
        result = {}
        for key, episode in self._episodes.items():
            episode_time = episode["timestamp"]
            if start_time <= episode_time <= end_time:
                result[key] = episode["value"]
        
        logger.debug(f"Found {len(result)} episodes in timeframe")
        return result
    
    def get_episodes_by_people(self, people: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Get episodes involving specific people.
        
        Args:
            people: Person name or list of names
            
        Returns:
            Dictionary of episodes involving the specified people
        """
        if isinstance(people, str):
            people = [people]
        
        logger.debug(f"Retrieving episodes involving: {people}")
        
        result = {}
        for key, episode in self._episodes.items():
            episode_people = episode.get("people", [])
            if any(person in episode_people for person in people):
                result[key] = episode["value"]
        
        logger.debug(f"Found {len(result)} episodes involving specified people")
        return result
    
    def get_episodes_by_location(self, location: str) -> Dict[str, Any]:
        """
        Get episodes that occurred at a specific location.
        
        Args:
            location: Location name
            
        Returns:
            Dictionary of episodes at the specified location
        """
        logger.debug(f"Retrieving episodes at location: {location}")
        
        result = {}
        for key, episode in self._episodes.items():
            episode_locations = episode.get("locations", [])
            if location.lower() in [loc.lower() for loc in episode_locations]:
                result[key] = episode["value"]
        
        logger.debug(f"Found {len(result)} episodes at location")
        return result
    
    def get_episodes_by_emotion(self, emotion: str) -> Dict[str, Any]:
        """
        Get episodes associated with a specific emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Dictionary of episodes with the specified emotion
        """
        logger.debug(f"Retrieving episodes with emotion: {emotion}")
        
        result = {}
        for key, episode in self._episodes.items():
            episode_emotions = episode.get("emotions", [])
            if emotion.lower() in [emo.lower() for emo in episode_emotions]:
                result[key] = episode["value"]
        
        logger.debug(f"Found {len(result)} episodes with emotion")
        return result
    
    def get_chronological_sequence(self, reverse: bool = False) -> List[Dict[str, Any]]:
        """
        Get episodes in chronological order.
        
        Args:
            reverse: If True, return most recent first
            
        Returns:
            List of episodes ordered by sequence/timestamp
        """
        episodes = [
            {"key": key, "episode": episode}
            for key, episode in self._episodes.items()
        ]
        
        # Sort by sequence number (preserves order of storage)
        episodes.sort(key=lambda x: x["episode"]["sequence"], reverse=reverse)
        
        return episodes
    
    def get_memory_strength(self, key: str) -> Optional[float]:
        """
        Get the memory strength for an episode based on recall frequency and vividness.
        
        Args:
            key: Episode identifier
            
        Returns:
            Memory strength score (0.0 to 1.0) or None if not found
        """
        if key not in self._episodes:
            return None
        
        episode = self._episodes[key]
        
        # Combine vividness, importance, and recall frequency
        recall_factor = min(1.0, episode["recalled_count"] * 0.1)
        strength = (
            episode["vividness"] * 0.4 +
            episode["importance"] * 0.4 +
            recall_factor * 0.2
        )
        
        return min(1.0, strength)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about episodic memory.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self._episodes:
            return {
                "total_episodes": 0,
                "average_vividness": 0.0,
                "average_importance": 0.0,
                "most_recalled": None,
                "strongest_memory": None
            }
        
        total_vividness = sum(ep["vividness"] for ep in self._episodes.values())
        total_importance = sum(ep["importance"] for ep in self._episodes.values())
        
        # Find most recalled and strongest memories
        most_recalled = max(
            self._episodes.items(),
            key=lambda x: x[1]["recalled_count"]
        )
        
        strongest = max(
            self._episodes.items(),
            key=lambda x: self.get_memory_strength(x[0])
        )
        
        return {
            "total_episodes": len(self._episodes),
            "average_vividness": total_vividness / len(self._episodes),
            "average_importance": total_importance / len(self._episodes),
            "most_recalled": {
                "key": most_recalled[0],
                "count": most_recalled[1]["recalled_count"]
            },
            "strongest_memory": {
                "key": strongest[0],
                "strength": self.get_memory_strength(strongest[0])
            }
        }
    
    def _extract_context(self, value: Any) -> Dict[str, Any]:
        """Extract contextual information from episode value."""
        context = {}
        
        if isinstance(value, dict):
            # Extract common context fields
            for field in ["when", "where", "what", "who", "why", "how"]:
                if field in value:
                    context[field] = value[field]
        elif isinstance(value, str):
            # Simple text analysis for context clues
            text_lower = value.lower()
            if any(word in text_lower for word in ["yesterday", "today", "tomorrow", "monday", "tuesday"]):
                context["temporal_reference"] = True
            if any(word in text_lower for word in ["at", "in", "near", "by"]):
                context["spatial_reference"] = True
        
        return context
    
    def _extract_emotions(self, value: Any) -> List[str]:
        """Extract emotional content from episode value."""
        emotions = []
        emotion_words = [
            "happy", "sad", "angry", "excited", "worried", "proud", "embarrassed",
            "surprised", "disappointed", "grateful", "frustrated", "hopeful",
            "anxious", "content", "overwhelmed", "relieved", "nostalgic"
        ]
        
        text = str(value).lower()
        for emotion in emotion_words:
            if emotion in text:
                emotions.append(emotion)
        
        return emotions
    
    def _extract_people(self, value: Any) -> List[str]:
        """Extract people mentioned in episode value."""
        people = []
        
        if isinstance(value, dict):
            # Check for common fields that might contain people
            for field in ["people", "who", "person"]:
                if field in value:
                    field_value = value[field]
                    if isinstance(field_value, list):
                        people.extend(field_value)
                    elif isinstance(field_value, str):
                        people.append(field_value)
        
        if isinstance(value, str):
            # Simple name detection (this could be enhanced with NLP)
            words = value.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 2:
                    people.append(word)
        
        return people
    
    def _extract_locations(self, value: Any) -> List[str]:
        """Extract locations mentioned in episode value."""
        locations = []
        
        if isinstance(value, dict) and "location" in value:
            locations = [value["location"]]
        elif isinstance(value, str):
            # Simple location detection
            location_indicators = ["at", "in", "near", "by", "to", "from"]
            words = value.split()
            for i, word in enumerate(words):
                if word.lower() in location_indicators and i + 1 < len(words):
                    next_word = words[i + 1]
                    if next_word[0].isupper():
                        locations.append(next_word)
        
        return locations
    
    def _extract_tags(self, value: Any) -> List[str]:
        """Extract tags or categories from episode value."""
        tags = []
        
        if isinstance(value, dict) and "tags" in value:
            tags = value["tags"] if isinstance(value["tags"], list) else [value["tags"]]
        
        # Add automatic tags based on content
        text = str(value).lower()
        if "work" in text or "office" in text or "meeting" in text:
            tags.append("work")
        if "family" in text or "mom" in text or "dad" in text:
            tags.append("family")
        if "travel" in text or "trip" in text or "vacation" in text:
            tags.append("travel")
        
        return tags
    
    def _assess_vividness(self, value: Any) -> float:
        """Assess the vividness of an episode (0.0 to 1.0)."""
        # Start with base vividness
        vividness = 0.5
        
        text = str(value).lower()
        
        # Increase vividness for sensory details
        sensory_words = ["saw", "heard", "felt", "smelled", "tasted", "bright", "loud", "soft"]
        sensory_count = sum(1 for word in sensory_words if word in text)
        vividness += min(0.3, sensory_count * 0.1)
        
        # Increase for emotional content
        if self._extract_emotions(value):
            vividness += 0.2
        
        return min(1.0, vividness)
    
    def _assess_importance(self, value: Any) -> float:
        """Assess the importance of an episode (0.0 to 1.0)."""
        # Start with base importance
        importance = 0.5
        
        text = str(value).lower()
        
        # Increase importance for significant events
        significant_words = ["first", "last", "important", "special", "memorable", "achievement"]
        if any(word in text for word in significant_words):
            importance += 0.3
        
        # Increase for emotional intensity
        emotion_count = len(self._extract_emotions(value))
        importance += min(0.2, emotion_count * 0.1)
        
        return min(1.0, importance)
    
    def _save_to_disk(self) -> None:
        """Save episodes to disk."""
        if not self._storage_path:
            return
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
            
            # Save episodes with sequence counter
            data = {
                "episodes": self._episodes,
                "episode_sequence": self._episode_sequence
            }
            
            with open(self._storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save episodes to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load episodes from disk."""
        if not self._storage_path or not os.path.exists(self._storage_path):
            return
        
        try:
            with open(self._storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._episodes = data.get("episodes", {})
            self._episode_sequence = data.get("episode_sequence", 0)
            
            logger.info(f"Loaded {len(self._episodes)} episodes from disk")
            
        except Exception as e:
            logger.error(f"Failed to load episodes from disk: {e}")
            self._episodes = {}
            self._episode_sequence = 0 