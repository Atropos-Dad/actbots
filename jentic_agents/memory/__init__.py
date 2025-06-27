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