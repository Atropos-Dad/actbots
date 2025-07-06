#!/usr/bin/env python3
"""
ActBots Live Demo with Jentic and OpenAI

This script demonstrates the ActBots agent working with live Jentic services
and a real OpenAI language model.

--------------------------------------------------------------------------
SETUP INSTRUCTIONS:

1. Create a `.env` file in this directory by copying `.env.template`.

2. Add your API keys to the `.env` file:
   - JENTIC_API_KEY: Your API key for the Jentic platform.
   - OPENAI_API_KEY: Your API key for OpenAI.

3. Make sure you have installed all dependencies:
   `make install`

4. Run the demo:
   `python main.py`
--------------------------------------------------------------------------
"""
import logging
import os
import sys

from dotenv import load_dotenv

# Add the package to the path
sys.path.insert(0, os.path.dirname(__file__))

from jentic_agents.agents.interactive_cli_agent import InteractiveCLIAgent
from jentic_agents.communication.inbox.cli_inbox import CLIInbox
from jentic_agents.communication.hitl.cli_intervention_hub import CLIInterventionHub
from jentic_agents.memory.scratch_pad import ScratchPadMemory
from jentic_agents.platform.jentic_client import JenticClient
from jentic_agents.reasoners.bullet_list_reasoner import BulletPlanReasoner
from jentic_agents.reasoners.standard_reasoner import StandardReasoner
# Local LiteLLM wrapper
from jentic_agents.utils.llm import LiteLLMChatLLM

# Prefix to detect Gemini provider
_GEMINI_PREFIX = "gemini/"

def main():
    """Run the live demo."""
    # Load environment variables from .env file
    load_dotenv()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


    print("🚀 Starting ActBots Live Demo")
    print("=" * 50)
    print("This agent uses live Jentic and OpenAI services.")
    print("Type your goal below, or 'quit' to exit.")
    print("-" * 50)

    # ------------------------------------------------------------------
    # Decide which LLM provider/model to use based on one env var.
    # ------------------------------------------------------------------
    model_name = os.getenv("LLM_MODEL", "gpt-4o")

    if not os.getenv("JENTIC_API_KEY"):
        print("❌ ERROR: Missing JENTIC_API_KEY in your .env file.")
        sys.exit(1)

    using_gemini = model_name.startswith(_GEMINI_PREFIX)

    # print(f"Using Gemini: {using_gemini}")
    if using_gemini and not os.getenv("GEMINI_API_KEY"):
        print("❌ ERROR: LLM_MODEL is Gemini but GEMINI_API_KEY is not set in .env.")
        sys.exit(1)

    if not using_gemini and not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: LLM_MODEL is OpenAI but OPENAI_API_KEY is not set in .env.")
        sys.exit(1)

    try:
        # 1. Initialize the JenticClient
        # This will use the live Jentic services.
        jentic_client = JenticClient()

        # 2. Initialize the LLM wrapper and Reasoner
        # Build the LLM wrapper for the selected model
        llm_wrapper = LiteLLMChatLLM(model=model_name)
        memory = ScratchPadMemory()

        # Initialize the CLI intervention hub for human-in-the-loop
        escalation_hub = CLIInterventionHub()

        reasoner = BulletPlanReasoner(
            jentic=jentic_client,
            memory=memory,
            llm=llm_wrapper,
            escalation=escalation_hub,
        )

        # 3. Initialize Memory and Inbox
        inbox = CLIInbox(prompt="Enter your goal: ")

        # 4. Create and run the Agent
        agent = InteractiveCLIAgent(
            reasoner=reasoner,
            memory=memory,
            inbox=inbox,
            jentic_client=jentic_client,
        )

        agent.spin()

    except ImportError as e:
        print(f"❌ ERROR: A required package is not installed. {e}")
        print("Please make sure you have run 'make install'.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error("An unexpected error occurred during the demo.", exc_info=True)
        sys.exit(1)

    print("-" * 50)
    print("👋 Demo finished. Goodbye!")


if __name__ == "__main__":
    main() 