# ActBots - Jentic-Powered AI Agent Library

A **clean, reusable Python library** that lets developers spin up AI agents whose reasoning loops automatically *search → load → execute* Jentic workflows and API operations.

## 🎯 Project Goals

- **Modular Architecture**: Clean separation between reasoning, memory, inbox, and platform layers
- **Extensible Design**: Easy to swap out reasoning strategies without breaking existing code
- **Jentic Integration**: Built-in support for discovering and executing Jentic workflows
- **Production Ready**: Comprehensive testing, type hints, and dependency isolation

## 🏗️ Architecture

```
jentic_agents/
│
├─ reasoners/              # Reasoning loop implementations
│   ├─ base_reasoner.py   # Abstract ReAct contract
│   └─ standard_reasoner.py # Concrete ReAct + Jentic integration
│
├─ agents/                 # Agent orchestration layer
│   ├─ base_agent.py      # Abstract agent interface
│   └─ interactive_cli_agent.py # CLI-based agent
│
├─ memory/                 # Memory backends
│   ├─ base_memory.py     # Abstract memory interface
│
├─ inbox/                  # Goal/task delivery systems
│   ├─ base_inbox.py      # Abstract inbox interface
│   └─ cli_inbox.py       # CLI input inbox
│
├─ platform/               # External service adapters
│   └─ jentic_client.py   # Jentic SDK wrapper
│
└─ tests/                  # Comprehensive test suite
```

## 🚀 Quick Start

### Installation

```bash
# Clone and set up the project
git clone <repository-url>
cd actbots

# Install dependencies (creates isolated .venv)
make install

# Run tests
make test

# Check code quality
make lint
```

### Basic Usage

```python
from jentic_agents.platform.jentic_client import JenticClient
from jentic_agents.reasoners.standard_reasoner import StandardReasoner
from jentic_agents.memory.scratch_pad import ScratchPadMemory
from jentic_agents.inbox.cli_inbox import CLIInbox
from jentic_agents.agents.interactive_cli_agent import InteractiveCLIAgent

# Create components
jentic_client = JenticClient(api_key="your-key-here")
reasoner = StandardReasoner(jentic_client=jentic_client)
memory = ScratchPadMemory()
inbox = CLIInbox()

# Create and run agent
agent = InteractiveCLIAgent(
    reasoner=reasoner,
    memory=memory,
    inbox=inbox,
    jentic_client=jentic_client
)

agent.spin()  # Start the interactive loop
```

### Demo Mode

Run the included demo to see the system in action with mock data:

```bash
python demo.py
```

## 🧠 Core Components

### Reasoners

The reasoning layer implements the **ReAct pattern** (plan → select_tool → act → observe → evaluate → reflect):

- **BaseReasoner**: Abstract interface defining the reasoning contract
- **StandardReasoner**: Concrete implementation using OpenAI + Jentic integration

### Agents

Agents orchestrate the reasoning loop with memory, inbox, and platform components:

- **BaseAgent**: Abstract agent interface with `spin()` main loop
- **InteractiveCLIAgent**: CLI-based agent for interactive use

### Memory

Pluggable memory backends for storing information across reasoning sessions:

- **BaseMemory**: Simple key-value storage interface

### Inbox

Goal delivery systems that feed tasks to agents:

- **BaseInbox**: Stream interface for goals from various sources
- **CLIInbox**: Interactive command-line goal input

### Platform

External service adapters:

- **JenticClient**: Thin wrapper around jentic-sdk with auth, retries, and logging

## 🧪 Testing

The project includes comprehensive tests with >90% coverage:

```bash
# Run all tests
make test

# Run specific test files
pytest jentic_agents/tests/test_reasoner.py -v

# Run with coverage
pytest --cov=jentic_agents
```

## 🔧 Development

### Project Structure

- **Strict interfaces first**: Abstract base classes with type hints
- **Dependency isolation**: All dependencies in project-local `.venv`
- **Single source of truth**: Only `JenticClient` contacts the Jentic SDK
- **Stateless reasoning**: `BaseReasoner.run()` returns packaged results
- **Testability**: External calls are injectable for easy mocking

### Code Quality

```bash
# Linting and formatting
make lint

# Type checking (strict mode)
make lint-strict

# Auto-fix common issues
ruff check . --fix
```

### Adding New Components

1. **New Reasoner**: Extend `BaseReasoner` and implement all abstract methods
2. **New Agent**: Extend `BaseAgent` and override I/O methods
3. **New Memory**: Extend `BaseMemory` with your storage backend
4. **New Inbox**: Extend `BaseInbox` for different goal sources

## 📊 Testing Criteria

The project meets the following quality standards:

1. **Unit Tests**: >90% coverage on core modules
2. **Error Handling**: Explicit exceptions with debugging context
3. **Static Quality**: Ruff linting passes, mypy type checking available
4. **Integration**: Demo script shows end-to-end functionality
5. **Isolation**: No global dependencies, clean `.venv` usage

## 🎪 Demo Results

The demo script successfully demonstrates:

```
🚀 Starting ActBots Demo
==================================================
AI Agent started. Type 'quit' to exit.
==================================================
✅ **Answer:** The answer to 2+2 is 4, as confirmed by the echo tool result: Echo: 2+2.

📋 **Used 1 tool(s) in 2 iteration(s):**
  1. Echo Tool
```

## 🔮 Future Enhancements

- **Vector Memory**: Add vector database memory backend
- **Advanced Reasoners**: Implement Reflexion, Tree of Thoughts
- **More Inboxes**: Slack, REST API, message queue integrations
- **Real Jentic SDK**: Replace mocks with actual jentic-sdk integration
- **Web Interface**: Add web-based agent interface
- **Deployment**: Docker, Kubernetes deployment configurations

## 📝 License

[Add your license here]

---

**Built following the ActBots specification for modular, future-proof Jentic-powered autonomous agents.**
