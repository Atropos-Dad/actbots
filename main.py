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
   - CLI mode: `python main.py` or `python main.py --mode cli`
   - UI mode: `python main.py --mode ui`
--------------------------------------------------------------------------
"""
import argparse
import logging
import os
import sys

from dotenv import load_dotenv

# Add the package to the path
sys.path.insert(0, os.path.dirname(__file__))

from jentic_agents.agents.interactive_cli_agent import InteractiveCLIAgent
from jentic_agents.communication.inbox.cli_inbox import CLIInbox
from jentic_agents.communication.hitl.cli_intervention_hub import CLIInterventionHub
from jentic_agents.memory.agent_memory import create_agent_memory
from jentic_agents.reasoners.freeform_reasoner import FreeformReasoner
from jentic_agents.platform.jentic_client import JenticClient

def main():
    """Main entry point for the agent CLI."""
    # This is a basic example of how to run the agent.
    # It creates an agent, gives it a goal, and runs it.
    
    # Create a memory instance for the agent
    memory = create_agent_memory()

    inbox = CLIInbox()
    intervention_hub = CLIInterventionHub()
    jentic_client = JenticClient()

    reasoner = FreeformReasoner(
        jentic=jentic_client,
        memory=memory,
        intervention_hub=intervention_hub,
        model="gpt-4o",
    )

    agent = InteractiveCLIAgent(
        reasoner=reasoner,
        memory=memory,
        inbox=inbox,
        jentic_client=jentic_client,
    )

    # Run the agent with a goal
    goal = "Send a message to the 'general' channel on Discord saying 'Hello, world!' using the 'create_message' tool"
    result = agent.process_goal(goal)
    print("Final answer:", result.final_answer)

if __name__ == "__main__":
    main() 