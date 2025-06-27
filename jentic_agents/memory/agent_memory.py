"""
Agent memory system with integrated logging and timing.

This provides a wrapper around Mem0 that adds our logging/timing infrastructure
while leveraging Mem0's proven performance improvements:
- +26% accuracy over OpenAI Memory
- 91% faster responses
- 90% fewer tokens
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Union
from mem0 import Memory
from ..utils.logger import get_logger
from ..utils.block_timer import Timer
from .base_memory import BaseMemory
from .memory_config import MemoryConfig, get_recommended_config

logger = get_logger(__name__)


class AgentMemory(BaseMemory):
    """
    High-performance agent memory system with logging and timing integration.
    
    Provides intelligent memory management for AI agents with multi-level storage,
    semantic search, and comprehensive observability infrastructure.
    
    Inherits from BaseMemory to provide a consistent interface with other memory types.
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        api_key: str = None,
        enable_telemetry: bool = False,
        use_recommended: bool = False
    ):
        """
        Initialize Mem0 memory with optional custom configuration.
        
        Args:
            config: Mem0 configuration (uses recommended config if None)
            api_key: API key for external services (optional if using local models)
            enable_telemetry: Whether to enable Mem0's telemetry
            use_recommended: Whether to auto-detect and use recommended config
        """
        logger.info("Initializing AgentMemory")
        
        # Use recommended configuration if requested
        if use_recommended and config is None:
            logger.info("Auto-detecting recommended configuration")
            config = get_recommended_config()
            
        # Default to Google + OpenAI configuration if none provided
        if config is None:
            logger.info("Using default Google + OpenAI configuration")
            # Validate API key for default cloud services
            if api_key is None:
                raise ValueError(
                    "API key is required for default configuration. "
                    "Provide api_key parameter, set use_recommended=True, "
                    "or provide a custom config with local models."
                )
            
            chroma_path = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
            config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": "mem0_memory",
                        "path": chroma_path,
                    }
                },
                "llm": {
                    "provider": "google",
                    "config": {
                        "model": "gemini-1.5-pro",
                        "temperature": 0.1,
                    }
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-3-small"
                    }
                }
            }
            logger.info(f"Using local ChromaDB at: {chroma_path}")
        
        # Check if config requires API keys
        requires_api_key = self._config_requires_api_key(config)
        if requires_api_key and api_key is None:
            logger.warning(
                "Configuration uses cloud services but no API key provided. "
                "This may cause initialization to fail."
            )
        
        # Initialize Mem0
        with Timer("Initialize Mem0"):
            if api_key:
                logger.info("Using Mem0 Platform with API key")
                # Disable telemetry if requested
                if not enable_telemetry:
                    os.environ["MEM0_TELEMETRY"] = "false"
                self.memory = Memory(api_key=api_key)
            else:
                logger.info("Using Mem0 local deployment")
                # Disable telemetry if requested
                if not enable_telemetry:
                    os.environ["MEM0_TELEMETRY"] = "false"
                self.memory = Memory.from_config(config)
        
        self.config = config
        self.api_key = api_key
        logger.info("AgentMemory initialized successfully")
    
    def _config_requires_api_key(self, config: Dict[str, Any]) -> bool:
        """Check if configuration requires external API keys."""
        cloud_providers = {"openai", "google", "huggingface", "anthropic"}
        
        llm_provider = config.get("llm", {}).get("provider", "")
        embedder_provider = config.get("embedder", {}).get("provider", "")
        
        return llm_provider in cloud_providers or embedder_provider in cloud_providers
    
    def add(
        self,
        messages: Union[str, List[Dict[str, str]]],
        user_id: str,
        agent_id: str = None,
        session_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> List[str]:
        """
        Add messages to memory.
        
        Args:
            messages: Message content or list of messages
            user_id: Unique user identifier
            agent_id: Optional agent identifier
            session_id: Optional session identifier
            metadata: Additional metadata
            
        Returns:
            List of memory IDs created
        """
        logger.info(f"Adding memories for user={user_id}, session={session_id}")
        
        # Prepare metadata
        mem_metadata = metadata or {}
        if agent_id:
            mem_metadata["agent_id"] = agent_id
        if session_id:
            mem_metadata["session_id"] = session_id
        
        with Timer(f"Add memories for user {user_id}"):
            if isinstance(messages, str):
                # Single message
                result = self.memory.add(messages, user_id=user_id, metadata=mem_metadata)
            else:
                # Multiple messages
                result = self.memory.add(messages, user_id=user_id, metadata=mem_metadata)
        
        memory_ids = [item['id'] for item in result.get('results', [])]
        logger.info(f"Created {len(memory_ids)} memory entries: {memory_ids}")
        return memory_ids
    
    def search(
        self,
        query: str,
        user_id: str,
        agent_id: str = None,
        session_id: str = None,
        limit: int = 5,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Search for relevant memories.
        
        Args:
            query: Search query
            user_id: User identifier
            agent_id: Optional agent identifier filter
            session_id: Optional session identifier filter
            limit: Maximum number of results
            filters: Additional filters
            
        Returns:
            Search results with metadata
        """
        logger.info(f"Searching memories for query='{query}', user={user_id}")
        
        # Use post-filtering approach to avoid ChromaDB filter compatibility issues
        # Mem0's filter handling has compatibility issues with ChromaDB validation
        logger.debug(f"Using post-filtering for agent_id={agent_id}, session_id={session_id}")
        
        with Timer(f"Search memories for user {user_id}"):
            # Search without filters first, then post-filter results
            # Get more results if we need to filter, to ensure we have enough after filtering
            search_limit = limit * 3 if (agent_id or session_id or filters) else limit
            results = self.memory.search(
                query=query,
                user_id=user_id,
                limit=search_limit
            )
        
        # Post-filter results by session/agent/custom filters if needed
        if (agent_id or session_id or filters) and results.get("results"):
            filtered_results = []
            for memory in results["results"]:
                metadata = memory.get("metadata", {})
                
                # Filter by agent_id
                if agent_id and metadata.get("agent_id") != agent_id:
                    continue
                    
                # Filter by session_id
                if session_id and metadata.get("session_id") != session_id:
                    continue
                
                # Filter by custom filters
                if filters:
                    skip_memory = False
                    for key, value in filters.items():
                        if metadata.get(key) != value:
                            skip_memory = True
                            break
                    if skip_memory:
                        continue
                
                filtered_results.append(memory)
                # Stop when we have enough results
                if len(filtered_results) >= limit:
                    break
                    
            results["results"] = filtered_results
        
        num_results = len(results.get("results", []))
        logger.info(f"Found {num_results} relevant memories")
        logger.debug(f"Search results: {json.dumps(results, indent=2, default=str)}")
        
        return results
    
    def get_all(
        self,
        user_id: str,
        agent_id: str = None,
        session_id: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all memories for a user/session.
        
        Args:
            user_id: User identifier
            agent_id: Optional agent identifier filter
            session_id: Optional session identifier filter
            limit: Maximum number of memories to return
            
        Returns:
            List of memory entries
        """
        logger.debug(f"Getting all memories for user={user_id}, session={session_id}")
        
        with Timer(f"Get all memories for user {user_id}"):
            result = self.memory.get_all(user_id=user_id, limit=limit)
            # Handle different response formats from Mem0
            if isinstance(result, dict) and "results" in result:
                all_memories = result["results"]
            elif isinstance(result, list):
                all_memories = result
            else:
                logger.warning(f"Unexpected get_all response format: {type(result)} - {result}")
                all_memories = []
        
        # Filter by agent_id or session_id if specified
        if agent_id or session_id:
            filtered_memories = []
            for memory in all_memories:
                metadata = memory.get("metadata", {})
                if agent_id and metadata.get("agent_id") != agent_id:
                    continue
                if session_id and metadata.get("session_id") != session_id:
                    continue
                filtered_memories.append(memory)
            all_memories = filtered_memories
        
        logger.debug(f"Retrieved {len(all_memories)} memories")
        return all_memories
    
    def update(
        self,
        memory_id: str,
        data: str,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of memory to update
            data: New memory content
            user_id: Optional user ID for validation
            
        Returns:
            Update result
        """
        logger.info(f"Updating memory {memory_id}")
        
        with Timer(f"Update memory {memory_id}"):
            result = self.memory.update(memory_id=memory_id, data=data)
        
        logger.debug(f"Update result: {result}")
        return result
    
    def delete_memory(
        self,
        memory_id: str,
        user_id: str = None
    ) -> bool:
        """
        Delete a specific memory by ID.
        
        Args:
            memory_id: ID of memory to delete
            user_id: Optional user ID for validation
            
        Returns:
            True if deleted successfully
        """
        logger.info(f"Deleting memory {memory_id}")
        
        with Timer(f"Delete memory {memory_id}"):
            try:
                self.memory.delete(memory_id=memory_id)
                logger.info(f"Successfully deleted memory {memory_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete memory {memory_id}: {e}")
                return False
    
    def delete_all(
        self,
        user_id: str,
        agent_id: str = None,
        session_id: str = None
    ) -> int:
        """
        Delete all memories for a user/session.
        
        Args:
            user_id: User identifier
            agent_id: Optional agent identifier filter
            session_id: Optional session identifier filter
            
        Returns:
            Number of memories deleted
        """
        logger.warning(f"Deleting all memories for user={user_id}, session={session_id}")
        
        with Timer(f"Delete all memories for user {user_id}"):
            # Get all memories first
            all_memories = self.get_all(user_id, agent_id, session_id)
            
            # Delete each memory
            deleted_count = 0
            for memory in all_memories:
                if self.delete_memory(memory["id"]):
                    deleted_count += 1
        
        logger.info(f"Deleted {deleted_count} memories")
        return deleted_count
    
    def get_memory_usage(self, user_id: str) -> Dict[str, Any]:
        """
        Get memory usage statistics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Usage statistics
        """
        logger.debug(f"Getting memory usage for user {user_id}")
        
        with Timer(f"Get memory usage for user {user_id}"):
            all_memories = self.get_all(user_id)
            
            # Calculate statistics
            total_memories = len(all_memories)
            
            # Group by session and agent
            sessions = set()
            agents = set()
            memory_types = {}
            
            for memory in all_memories:
                # Handle different memory formats safely
                if isinstance(memory, dict):
                    metadata = memory.get("metadata", {})
                    
                    if isinstance(metadata, dict):
                        if "session_id" in metadata:
                            sessions.add(metadata["session_id"])
                        if "agent_id" in metadata:
                            agents.add(metadata["agent_id"])
                        
                        # Count memory types (could be enhanced)
                        memory_type = metadata.get("type", "general")
                        memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
                    else:
                        # If metadata is not a dict, count as general
                        memory_types["general"] = memory_types.get("general", 0) + 1
                else:
                    # If memory is not a dict (e.g., string), count as general
                    logger.debug(f"Unexpected memory format: {type(memory)} - {memory}")
                    memory_types["general"] = memory_types.get("general", 0) + 1
        
        usage_stats = {
            "user_id": user_id,
            "total_memories": total_memories,
            "unique_sessions": len(sessions),
            "unique_agents": len(agents),
            "memory_types": memory_types,
            "sessions": list(sessions),
            "agents": list(agents)
        }
        
        logger.debug(f"Memory usage stats: {usage_stats}")
        return usage_stats
    
    def get_context_for_chat(
        self,
        query: str,
        user_id: str,
        agent_id: str = None,
        session_id: str = None,
        max_memories: int = 5
    ) -> str:
        """
        Get relevant memory context for a chat query.
        
        Args:
            query: Current user message/query
            user_id: User identifier
            agent_id: Optional agent identifier
            session_id: Optional session identifier
            max_memories: Maximum memories to include in context
            
        Returns:
            Formatted memory context string
        """
        logger.debug(f"Building chat context for query: '{query}'")
        
        with Timer("Build chat context"):
            # Search for relevant memories
            search_results = self.search(
                query=query,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                limit=max_memories
            )
            
            memories = search_results.get("results", [])
            
            if not memories:
                return "No relevant memories found."
            
            # Format memories into context
            context_lines = ["Relevant memories:"]
            for memory in memories:
                memory_text = memory.get("memory", "")
                score = memory.get("score", 0.0)
                context_lines.append(f"- {memory_text} (relevance: {score:.2f})")
            
            context = "\n".join(context_lines)
        
        logger.debug(f"Built context with {len(memories)} memories")
        return context
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the memory system.
        
        Returns:
            Health status information
        """
        logger.info("Performing AgentMemory health check")
        
        health = {
            "status": "healthy",
            "mem0_config": self.config,
            "using_api_key": bool(self.api_key),
            "timestamp": time.time(),
            "errors": []
        }
        
        try:
            with Timer("Health check - test operation"):
                # Try a simple operation
                test_user = "health_check_user"
                test_memory_id = self.add("Health check test", user_id=test_user)
                if test_memory_id:
                    self.delete_memory(test_memory_id[0])
                    logger.info("Health check passed")
                else:
                    health["status"] = "degraded"
                    health["errors"].append("Failed to create test memory")
                    
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health["status"] = "unhealthy"
            health["errors"].append(str(e))
        
        return health
    
    # BaseMemory interface implementation
    def store(self, key: str, value: Any) -> None:
        """
        Store a value under the given key (BaseMemory interface).
        
        This method maps to the add() method with a default user_id.
        For full functionality, use add() method directly.
        
        Args:
            key: Unique identifier for the stored value (used as user_id)
            value: Data to store (message content)
        """
        logger.debug(f"Storing value with key: {key} via BaseMemory interface")
        
        # Use key as user_id and convert value to string message
        message = str(value) if not isinstance(value, str) else value
        
        try:
            result = self.add(message, user_id=key)
            logger.debug(f"Stored value successfully, memory IDs: {result}")
        except Exception as e:
            logger.error(f"Failed to store value with key {key}: {e}")
            raise
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key (BaseMemory interface).
        
        This method gets all memories for the user_id (key) and returns
        the most recent memory content.
        
        Args:
            key: Unique identifier for the value (used as user_id)
            
        Returns:
            Most recent memory content, or None if no memories found
        """
        logger.debug(f"Retrieving value with key: {key} via BaseMemory interface")
        
        try:
            memories = self.get_all(user_id=key, limit=1)
            if memories:
                # Return the memory content from the most recent memory
                memory = memories[0]
                if isinstance(memory, dict):
                    return memory.get("memory", memory.get("text", str(memory)))
                else:
                    return str(memory)
            else:
                logger.debug(f"No memories found for key: {key}")
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve value with key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete stored values by key (BaseMemory interface).
        
        This deletes all memories for the user_id (key).
        
        Args:
            key: Unique identifier for the values to delete (used as user_id)
            
        Returns:
            True if any memories were deleted, False otherwise
        """
        logger.debug(f"Deleting values with key: {key} via BaseMemory interface")
        
        try:
            deleted_count = self.delete_all(user_id=key)
            return deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete values with key {key}: {e}")
            return False
    
    def clear(self) -> None:
        """
        Clear all stored values (BaseMemory interface).
        
        Note: This is a destructive operation that affects all users.
        Use with caution in production environments.
        """
        logger.warning("Clearing all memories via BaseMemory interface - this affects all users!")
        
        try:
            # This is a simplified implementation - in a real scenario you'd want
            # to get all user IDs first, but Mem0 doesn't provide that directly
            logger.warning("Cannot clear all memories - BaseMemory clear() is not fully supported")
            logger.warning("Use delete_all(user_id) for specific users instead")
        except Exception as e:
            logger.error(f"Failed to clear all memories: {e}")
    
    def keys(self) -> List[str]:
        """
        Get all stored keys (BaseMemory interface).
        
        Note: Mem0 doesn't provide a direct way to list all user IDs,
        so this method returns an empty list with a warning.
        
        Returns:
            Empty list (not supported by underlying Mem0 storage)
        """
        logger.warning("BaseMemory keys() method is not fully supported by Mem0")
        logger.warning("Use get_all(user_id) to get memories for specific users")
        return []


def create_agent_memory(
    config_type: str = "recommended",
    api_key: str = None,
    chroma_path: str = None,
    use_platform: bool = False,
    **kwargs
) -> AgentMemory:
    """
    Factory function to create an AgentMemory instance with predefined configurations.
    
    Args:
        config_type: Type of configuration ("local", "openai", "google", "anthropic", "huggingface", "recommended")
        api_key: API key for external services (auto-detected from env if not provided)
        chroma_path: Path for local ChromaDB storage
        use_platform: Whether to use Mem0 Platform (requires api_key)
        **kwargs: Additional configuration options
        
    Returns:
        Configured AgentMemory instance
    """
    if use_platform and not api_key:
        raise ValueError("API key required when use_platform=True")
    
    # Auto-detect API key from environment if not provided
    if not api_key:
        if config_type in ["openai", "google"]:
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif config_type == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        elif config_type == "huggingface":
            api_key = os.environ.get("HUGGINGFACE_API_KEY")
    
    # Get configuration based on type
    config = None
    if not use_platform:
        config_kwargs = {"chroma_path": chroma_path} if chroma_path else {}
        config_kwargs.update(kwargs)
        
        if config_type == "local":
            config = MemoryConfig.local_only(**config_kwargs)
        elif config_type == "openai":
            config = MemoryConfig.openai_config(**config_kwargs)
        elif config_type == "google":
            config = MemoryConfig.google_config(**config_kwargs)
        elif config_type == "anthropic":
            config = MemoryConfig.anthropic_config(**config_kwargs)
        elif config_type == "huggingface":
            config = MemoryConfig.huggingface_config(**config_kwargs)
        elif config_type == "recommended":
            config = MemoryConfig.get_recommended_config()
            # Apply custom chroma_path if provided
            if chroma_path:
                config["vector_store"]["config"]["path"] = chroma_path
        else:
            raise ValueError(f"Unknown config_type: {config_type}")
    
    return AgentMemory(
        config=config,
        api_key=api_key,
        enable_telemetry=kwargs.get("enable_telemetry", False),
        use_recommended=(config_type == "recommended")
    ) 