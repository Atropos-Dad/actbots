# ActBots - Jentic-Powered AI Agent Library

A **clean, reusable Python library** that lets developers spin up AI agents whose reasoning loops automatically *search → load → execute* Jentic workflows and API operations.

## 🎯 Project Goals

- **Modular Architecture**: Clean separation between reasoning, memory, inbox, and platform layers
- **Extensible Design**: Easy to swap out reasoning strategies without breaking existing code
- **Jentic Integration**: Built-in support for discovering and executing Jentic workflows
- **Production Ready**: Comprehensive testing, type hints, and dependency isolation

## 🏗️ Architecture

```
├── README.md
├── docs
│   ├── dynamic_escalation_system.md
│   ├── escalation_system.md
│   └── human_in_the_loop.md
├── jentic_agents
│   ├── agents
│   │   ├── base_agent.py
│   │   ├── interactive_cli_agent.py
│   │   └── simple_ui_agent.py
│   ├── communication
│   │   ├── controllers
│   │   │   ├── base_controller.py
│   │   │   ├── cli_controller.py
│   │   │   └── discord_controller.py
│   │   ├── hitl
│   │   │   ├── base_intervention_hub.py
│   │   │   ├── cli_intervention_hub.py
│   │   │   └── discord_intervention_hub.py
│   │   ├── inbox
│   │   │   ├── base_inbox.py
│   │   │   ├── cli_inbox.py
│   │   │   └── discord_inbox.py
│   │   └── outbox
│   │       ├── base_outbox.py
│   │       ├── cli_outbox.py
│   │       └── discord_outbox.py
│   ├── inbox
│   ├── logs
│   │   └── actbots.log
│   ├── memory
│   │   ├── agent_memory.py
│   │   ├── base_memory.py
│   │   └── scratch_pad.py
│   ├── outbox
│   ├── platform
│   │   └── jentic_client.py
│   ├── prompts
│   │   ├── agent_system_prompt.txt
│   │   ├── bullet_plan.txt
│   │   ├── context_analysis.txt
│   │   ├── goal_evaluation.txt
│   │   ├── hybrid_classifier.txt
│   │   ├── keyword_extraction.txt
│   │   ├── param_correction_prompt.txt
│   │   ├── param_generation.txt
│   │   ├── reasoning_prompt.txt
│   │   ├── reflection_prompt.txt
│   │   └── select_tool.txt
│   ├── reasoners
│   │   ├── base_reasoner.py
│   │   ├── bullet_list_reasoner
│   │   │   ├── bullet_plan_reasoner.py
│   │   │   ├── parameter_generator.py
│   │   │   ├── plan_parser.py
│   │   │   ├── reasoner_state.py
│   │   │   ├── reflection_engine.py
│   │   │   ├── step_executor.py
│   │   │   └── tool_selector.py
│   │   ├── freeform_reasoner
│   │   │   └── freeform_reasoner.py
│   │   └── hybrid_reasoner
│   │       └── hybrid_reasoner.py
│   ├── tools
│   └── utils
│       ├── block_timer.py
│       ├── config.py
│       ├── llm.py
│       ├── logger.py
│       ├── parsing_helpers.py
│       ├── prompt_loader.py
│       └── shared_console.py
├── main.py
└── pyproject.toml
```

## 🚀 Quick Start

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) for dependency management
- Jentic platform access (API key required)
- API key for your chosen LLM provider (OpenAI, Gemini, or Anthropic)

### Installation

#### Using uv (Recommended)

**macOS/Linux:**
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd actbots

# Set up virtual environment and install dependencies
uv venv && source .venv/bin/activate && uv pip install -e .
```

**Windows:**
```powershell
# Install uv if you haven't already
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone the repository
git clone <repository-url>
cd actbots

# Set up virtual environment and install dependencies
uv venv
.venv\Scripts\activate
uv pip install -e .
```

### Configuration

1. **Create a `.env` file** in the project root:
```bash
# Required
JENTIC_API_KEY=your-jentic-api-key

# Choose one LLM provider
OPENAI_API_KEY=your-openai-api-key        # If using OpenAI
GEMINI_API_KEY=your-gemini-api-key        # If using Gemini  
ANTHROPIC_API_KEY=your-anthropic-api-key  # If using Anthropic

# Optional - for Discord mode
DISCORD_BOTTOKEN=your-discord-bot-token
```

2. **Configure your LLM provider** in `pyproject.toml`:
```toml
[tool.actbots.llm]
provider = "gemini"        # or "openai", "anthropic"
model = "gemini-2.5-flash" # or "gpt-4o", "claude-3-sonnet-20240229", etc.
```

3. **Configure Discord** (optional) in `pyproject.toml`:
```toml
[tool.actbots.discord]
enabled = true
target_user_id = 123456789         # Your Discord user ID for escalations
monitored_channels = [987654321]   # Channel IDs to monitor
default_channel_id = 987654321     # Default channel for responses
```

### Running the Application

#### CLI Mode (Default)
```bash
python main.py
# or explicitly
python main.py --mode cli
```

#### UI Mode
```bash
python main.py --mode ui
```

#### Discord Bot Mode
```bash
python main.py --mode discord
```

### Programmatic Usage
```python
from jentic_agents.agents.interactive_cli_agent import InteractiveCLIAgent
from jentic_agents.communication.controllers.cli_controller import CLIController
from jentic_agents.platform.jentic_client import JenticClient
from jentic_agents.memory.scratch_pad import ScratchPadMemory
from jentic_agents.utils.llm import LiteLLMChatLLM
from jentic_agents.reasoners.hybrid_reasoner.hybrid_reasoner import HybridReasoner

# Initialize components
jentic_client = JenticClient()
memory = ScratchPadMemory()
llm_wrapper = LiteLLMChatLLM(model="gemini-2.5-flash")
controller = CLIController()

# Create reasoner and agent
reasoner = HybridReasoner(
    jentic=jentic_client,
    memory=memory,
    llm=llm_wrapper,
    intervention_hub=controller.intervention_hub
)

agent = InteractiveCLIAgent(
    reasoner=reasoner,
    memory=memory,
    controller=controller,
    jentic_client=jentic_client
)

# Start the agent
agent.spin()
```

## 🔧 Development Setup

### Running Tests
```bash
# Using uv
uv run pytest jentic_agents/tests/

# Using pip
python -m pytest jentic_agents/tests/
```

### Code Quality
```bash
# Format code with ruff
uv run ruff format jentic_agents/

# Type checking
uv run mypy jentic_agents/

# Linting
uv run ruff check jentic_agents/
```

### Project Configuration

The project uses `pyproject.toml` for configuration. Key sections include:

- `[tool.actbots.llm]` - LLM provider and model settings
- `[tool.actbots.discord]` - Discord bot configuration
- `[tool.actbots.logging]` - Logging configuration
- `[tool.actbots.memory]` - Memory and embedding settings

## 📝 Usage Examples

### CLI Agent with Custom Configuration
```python
from jentic_agents.utils.config import get_config_value

# Get configured model from pyproject.toml
model_name = get_config_value("llm", "model", default="gpt-4o")
llm_wrapper = LiteLLMChatLLM(model=model_name)
```

### Discord Bot with Monitoring
```python
# Discord configuration from pyproject.toml
discord_user_id = get_config_value("discord", "target_user_id")
monitored_channels = get_config_value("discord", "monitored_channels", default=[])
default_channel_id = get_config_value("discord", "default_channel_id", default=None)
```

## 📚 Documentation

- [Dynamic Escalation System](docs/dynamic_escalation_system.md)
- [Human-in-the-Loop Guide](docs/human_in_the_loop.md)
- [Escalation System Overview](docs/escalation_system.md)
