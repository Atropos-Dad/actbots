"""
Memory subsystem for jentic agents with multiple cognitive memory types.

This package provides specialized memory systems for different types of information:

Memory Types:
Main Agent Memory using Mem0 (https://docs.mem0.ai/open-source/overview):
- AgentMemory: AI-powered memory using Mem0 for intelligent conversation memory

Specialized Memory Types (based on https://blog.langchain.com/memory-for-agents/):
- SemanticMemory: Vector-based storage for general knowledge, facts, and concepts  
- EpisodicMemory: Personal experiences and events with temporal/contextual details
- ProceduralMemory: Skills, habits, and step-by-step procedures with mastery tracking
- BaseMemory: Abstract interface that all memory types implement

Configuration:
- MemoryConfig: Pre-built configurations for different deployment scenarios
- get_recommended_config: Auto-detects best available configuration
"""

# Base memory interface
from .base_memory import BaseMemory

# Main agent memory powered by Mem0
from .agent_memory import AgentMemory, create_agent_memory

# Specialized memory types for different cognitive functions
from .semantic_memory import SemanticMemory      # General knowledge & facts storage with vector search
from .episodic_memory import EpisodicMemory      # Personal experiences & events with temporal context  
from .procedural_memory import ProceduralMemory  # Skills, habits & step-by-step procedures

# Configuration utilities
from .memory_config import MemoryConfig, get_recommended_config

# Main exports for easy importing
__all__ = [
    # Base interface
    "BaseMemory",
    
    # Primary memory system
    "AgentMemory", 
    "create_agent_memory",
    
    # Specialized memory types
    "SemanticMemory",     # ChromaDB vector storage for facts/knowledge
    "EpisodicMemory",     # Personal experiences with time/context
    "ProceduralMemory",   # Skills and procedures with mastery tracking
    
    # Configuration
    "MemoryConfig",
    "get_recommended_config"
]
