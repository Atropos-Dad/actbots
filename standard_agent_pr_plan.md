# Standard Agent Upgrade Plan: Integrating ActBots Features

## Executive Summary

This document outlines a strategic plan to upgrade the `standard_agent` repository with advanced features from `actbots`. The goal is to transform `standard_agent` into a mini universal agent with enhanced capabilities while maintaining its simplicity and modularity.

## Repository Analysis

### Current State: standard_agent
- **Architecture**: Basic ReWOO reasoner with CLI interface
- **Build System**: requirements.txt + Makefile
- **Configuration**: config.json
- **Communication**: Simple CLI inbox/outbox pattern
- **Features**: Basic self-healing AI agent with Jentic integration

### Advanced Features in actbots
- **Multiple Reasoning Strategies**: Bullet list, freeform, and hybrid reasoners
- **Multi-Channel Communication**: CLI, Discord, and UI interfaces
- **Human-in-the-Loop (HITL)**: Comprehensive intervention system
- **Advanced Configuration**: TOML-based with rich options
- **Sophisticated Logging**: Multi-handler with rotation and per-module control
- **Modular Prompts**: Separated prompt management system
- **Enhanced Memory**: ChromaDB integration and embeddings support

## Chronological PR Plan

### Phase 1: Foundation (Already Completed)
**PR #1: Configuration and Build System Migration** ✅
- Migrate from requirements.txt to pyproject.toml (PDM)
- Update logging infrastructure
- Implement LiteLLM configuration for LLM agnosticism
- Status: Already submitted

### Phase 2: Core Infrastructure Upgrades

**PR #2: Modular Prompt Management System**
*Priority: High | Effort: Medium | Dependencies: PR #1*

**Description**: Extract all hardcoded prompts into a modular system
**Changes**:
- Create `jentic_agents/prompts/` directory
- Implement `PromptLoader` utility class
- Extract existing prompts from ReWOO reasoner
- Add prompt templates for:
  - agent_system_prompt.txt
  - reasoning_prompt.txt
  - reflection_prompt.txt
  - context_analysis.txt
- Update ReWOOReasoner to use PromptLoader

**Benefits**: 
- Easier prompt iteration and testing
- Better organization and maintenance
- Foundation for multiple reasoner strategies

---

**PR #3: Enhanced Configuration System**
*Priority: High | Effort: Low | Dependencies: PR #1*

**Description**: Expand configuration capabilities using TOML
**Changes**:
- Enhance config.toml structure with sections:
  - `[tool.actbots.logging]` - Advanced logging configuration
  - `[tool.actbots.memory]` - Memory backend settings
  - `[tool.actbots.reasoner.*]` - Per-reasoner configurations
  - `[tool.actbots.discord]` - Discord bot settings (for future)
- Implement `config.py` utility for centralized config access
- Add environment variable override support
- Document all configuration options

**Benefits**:
- Centralized configuration management
- Environment-specific settings
- Better defaults and customization

---

**PR #4: Advanced Logging Infrastructure**
*Priority: Medium | Effort: Medium | Dependencies: PR #3*

**Description**: Implement sophisticated logging with multiple handlers
**Changes**:
- Create `jentic_agents/utils/logger.py`
- Implement console and file handlers with:
  - Colored console output (using Rich)
  - File rotation support
  - Per-module log level control
  - Configurable formats
- Add structured logging support
- Create log directory management
- Update all modules to use new logger

**Benefits**:
- Better debugging and monitoring
- Production-ready logging
- Cleaner console output

### Phase 3: Communication Layer Enhancement

**PR #5: Communication Abstraction Layer**
*Priority: High | Effort: Large | Dependencies: PR #2, #3*

**Description**: Implement multi-channel communication framework
**Changes**:
- Create `jentic_agents/communication/` module structure:
  - `controllers/` - Channel controllers
  - `inbox/` - Input interfaces
  - `outbox/` - Output interfaces
  - `hitl/` - Human intervention interfaces
- Implement base abstractions:
  - `BaseController`
  - `BaseInbox`
  - `BaseOutbox`
  - `BaseInterventionHub`
- Refactor existing CLI interfaces to new structure
- Add controller registration system

**Benefits**:
- Foundation for multi-channel support
- Clean separation of concerns
- Easier to add new communication channels

---

**PR #6: Human-in-the-Loop (HITL) System**
*Priority: High | Effort: Large | Dependencies: PR #5*

**Description**: Add intervention capabilities for ambiguous situations
**Changes**:
- Implement intervention types:
  - Goal clarification
  - Tool selection assistance
  - Parameter correction
  - Plan review
  - Step guidance
  - Decision points
- Create CLI intervention hub
- Add intervention points to ReWOOReasoner
- Implement timeout and fallback mechanisms
- Add comprehensive documentation

**Benefits**:
- Increased reliability in complex scenarios
- User maintains control
- Better error recovery

### Phase 4: Advanced Reasoning Capabilities

**PR #7: Bullet List Reasoner**
*Priority: Medium | Effort: Large | Dependencies: PR #2, #6*

**Description**: Implement structured reasoning with bullet-point plans
**Changes**:
- Create `jentic_agents/reasoners/bullet_list_reasoner/` module
- Implement components:
  - `BulletPlanReasoner` - Main reasoner
  - `PlanParser` - Parse bullet lists
  - `StepExecutor` - Execute individual steps
  - `ReflectionEngine` - Enhanced reflection
  - `ParameterGenerator` - Smart parameter generation
  - `ToolSelector` - Improved tool selection
- Add state management (`ReasonerState`)
- Integrate with HITL system
- Add comprehensive tests

**Benefits**:
- More structured reasoning
- Better plan visibility
- Enhanced reflection capabilities

---

**PR #8: Hybrid Reasoner**
*Priority: Medium | Effort: Medium | Dependencies: PR #7*

**Description**: Combine multiple reasoning strategies
**Changes**:
- Create `jentic_agents/reasoners/hybrid_reasoner/`
- Implement strategy selection logic
- Add context analysis for strategy choice
- Create unified interface
- Add configuration for strategy preferences
- Implement fallback mechanisms

**Benefits**:
- Best of both reasoning approaches
- Adaptive to different task types
- More robust overall performance

### Phase 5: Enhanced Features

**PR #9: Discord Integration**
*Priority: Low | Effort: Large | Dependencies: PR #5, #6*

**Description**: Add Discord bot capabilities
**Changes**:
- Create Discord communication components:
  - `DiscordController`
  - `DiscordInbox`
  - `DiscordOutbox`
  - `DiscordInterventionHub`
- Implement Discord bot with:
  - Channel monitoring
  - Command handling
  - Embed formatting
  - User mentions for escalation
  - Auto-reactions
- Add Discord configuration section
- Create setup documentation

**Benefits**:
- Remote agent access
- Team collaboration
- Async communication

---

**PR #10: Simple UI Agent**
*Priority: Low | Effort: Large | Dependencies: PR #5*

**Description**: Add web-based UI for agent interaction
**Changes**:
- Create `SimpleUIAgent` class
- Implement local web server
- Add HTML/CSS/JS interface
- Create WebSocket for real-time updates
- Add file upload/download capabilities
- Implement session management

**Benefits**:
- User-friendly interface
- Better for non-technical users
- Visual progress tracking

---

**PR #11: Advanced Memory with Embeddings**
*Priority: Medium | Effort: Medium | Dependencies: PR #3*

**Description**: Enhance memory with vector search capabilities
**Changes**:
- Add ChromaDB integration
- Implement embedding generation
- Create semantic search capabilities
- Add memory persistence
- Implement memory management (cleanup, limits)
- Add configuration for embedding models

**Benefits**:
- Better context retention
- Semantic memory search
- Long-term memory capabilities

### Phase 6: Testing and Documentation

**PR #12: Comprehensive Test Suite**
*Priority: High | Effort: Large | Dependencies: All previous PRs*

**Description**: Expand test coverage and add integration tests
**Changes**:
- Add unit tests for all new components
- Create integration test scenarios
- Add mock implementations for testing
- Implement test fixtures
- Add performance benchmarks
- Create CI/CD test matrix

**Benefits**:
- Higher code quality
- Regression prevention
- Confidence in changes

---

**PR #13: Documentation Overhaul**
*Priority: Medium | Effort: Medium | Dependencies: All previous PRs*

**Description**: Create comprehensive documentation
**Changes**:
- Update README with new features
- Create user guides:
  - Quick start guide
  - Configuration guide
  - Extension guide
- Add API documentation
- Create example scripts
- Add troubleshooting guide
- Document best practices

**Benefits**:
- Easier adoption
- Better developer experience
- Reduced support burden

## Implementation Guidelines

### General Principles
1. **Backward Compatibility**: Ensure existing functionality remains intact
2. **Incremental Changes**: Each PR should be self-contained and testable
3. **Documentation First**: Update docs with each feature addition
4. **Test Coverage**: Maintain or improve test coverage with each PR
5. **Code Quality**: Follow existing code style and add type hints

### Testing Strategy
- Unit tests for each new component
- Integration tests for cross-component functionality
- Manual testing checklist for each PR
- Performance benchmarks for critical paths

### Migration Path
1. Each PR should include migration notes if breaking changes
2. Provide clear upgrade instructions
3. Consider feature flags for gradual rollout
4. Maintain compatibility layer where possible

## Timeline Estimate

- **Phase 1**: ✅ Completed
- **Phase 2**: 2-3 weeks (PRs #2-4)
- **Phase 3**: 3-4 weeks (PRs #5-6)
- **Phase 4**: 4-5 weeks (PRs #7-8)
- **Phase 5**: 4-6 weeks (PRs #9-11)
- **Phase 6**: 2-3 weeks (PRs #12-13)

**Total Estimated Time**: 15-21 weeks for full implementation

## Conclusion

This upgrade plan transforms `standard_agent` into a powerful yet maintainable universal agent framework. By incorporating the best features from `actbots` while maintaining simplicity, the upgraded system will support diverse use cases from simple CLI interactions to complex multi-channel deployments with human oversight.

The modular approach ensures that users can adopt features incrementally based on their needs, while the comprehensive testing and documentation ensure production readiness.