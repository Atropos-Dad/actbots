# Key Features to Implement from ActBots

## Priority 1: Core Infrastructure
1. **Modular Prompt Management**
   - Separate prompts from code
   - Easy prompt iteration and testing
   - Foundation for multiple reasoners

2. **Enhanced Configuration (TOML)**
   - Rich configuration options
   - Environment overrides
   - Per-component settings

3. **Advanced Logging**
   - Console + file handlers
   - Colored output with Rich
   - Per-module log levels
   - Log rotation

## Priority 2: Communication & HITL
4. **Communication Abstraction Layer**
   - Multi-channel support (CLI, Discord, UI)
   - Clean separation of concerns
   - Extensible architecture

5. **Human-in-the-Loop System**
   - Goal clarification
   - Tool selection help
   - Parameter assistance
   - Plan review
   - Step guidance
   - Decision points

## Priority 3: Advanced Reasoning
6. **Bullet List Reasoner**
   - Structured planning
   - Better visibility
   - Enhanced reflection

7. **Hybrid Reasoner**
   - Combines strategies
   - Adaptive reasoning
   - Context-aware

## Priority 4: Enhanced Features
8. **Discord Integration**
   - Remote access
   - Team collaboration
   - Async communication

9. **Web UI**
   - User-friendly interface
   - Visual progress
   - File handling

10. **Advanced Memory**
    - Vector embeddings
    - Semantic search
    - Persistence

## Implementation Order
1. Foundation upgrades (config, prompts, logging)
2. Communication framework
3. HITL system
4. New reasoners
5. Channel implementations
6. Memory enhancements
7. Testing & documentation

This approach ensures each feature builds on previous ones while maintaining backward compatibility.