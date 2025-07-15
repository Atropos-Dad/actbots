# Standard Agent PR Checklist

## ✅ Completed
- [x] PR #1: Configuration and Build System Migration (pyproject.toml, logging, LiteLLM)

## 📋 Ready to Start (Phase 2: Core Infrastructure)
- [ ] PR #2: Modular Prompt Management System
- [ ] PR #3: Enhanced Configuration System
- [ ] PR #4: Advanced Logging Infrastructure

## 🔄 Phase 3: Communication Layer
- [ ] PR #5: Communication Abstraction Layer
- [ ] PR #6: Human-in-the-Loop (HITL) System

## 🧠 Phase 4: Advanced Reasoning
- [ ] PR #7: Bullet List Reasoner
- [ ] PR #8: Hybrid Reasoner

## ⚡ Phase 5: Enhanced Features
- [ ] PR #9: Discord Integration
- [ ] PR #10: Simple UI Agent
- [ ] PR #11: Advanced Memory with Embeddings

## 📚 Phase 6: Testing & Documentation
- [ ] PR #12: Comprehensive Test Suite
- [ ] PR #13: Documentation Overhaul

## Quick Start for Next PR
Start with **PR #2: Modular Prompt Management System**:
1. Create `jentic_agents/prompts/` directory
2. Implement `PromptLoader` class in `utils/prompt_loader.py`
3. Extract prompts from ReWOO reasoner
4. Create prompt template files
5. Update ReWOOReasoner to use PromptLoader
6. Add tests for PromptLoader
7. Update documentation

Each PR should include:
- Implementation code
- Unit tests
- Documentation updates
- Migration notes (if needed)
- Example usage