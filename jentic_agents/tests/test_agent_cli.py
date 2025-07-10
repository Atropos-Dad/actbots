"""
Unit tests for InteractiveCLIAgent.
"""

import pytest
from unittest.mock import MagicMock

from jentic_agents.agents.interactive_cli_agent import InteractiveCLIAgent
from jentic_agents.reasoners.base_reasoner import ReasoningResult
from jentic_agents.communication.controllers.cli_controller import CLIController

@pytest.fixture
def fully_integrated_agent(mocker):
    """
    Sets up a complete InteractiveCLIAgent with a mocked controller
    and its sub-components (inbox, outbox, hub).
    """
    mock_controller = MagicMock(spec=CLIController)
    mock_controller.inbox = MagicMock()
    mock_controller.outbox = MagicMock()
    mock_controller.intervention_hub = MagicMock()
    
    mock_reasoner = MagicMock()
    mock_jentic_client = MagicMock()
    mock_memory = MagicMock()

    agent = InteractiveCLIAgent(
        controller=mock_controller,
        reasoner=mock_reasoner,
        jentic_client=mock_jentic_client,
        memory=mock_memory,
    )
    
    # Mock process_goal to isolate testing of goal handling from reasoning
    agent.process_goal = MagicMock()
    
    return agent, mock_controller

def test_handle_goal_success(fully_integrated_agent):
    """Test that successful results from a goal are sent to the outbox."""
    agent, mock_controller = fully_integrated_agent
    
    result = ReasoningResult(final_answer="Success!", success=True, iterations=1, tool_calls=[])
    agent.process_goal.return_value = result

    agent._handle_goal("test goal")
    
    agent.process_goal.assert_called_once_with("test goal")
    mock_controller.outbox.display_reasoning_result.assert_called_once_with(result)

def test_handle_goal_failure(fully_integrated_agent):
    """Test that failure results from a goal are sent to the outbox."""
    agent, mock_controller = fully_integrated_agent

    result = ReasoningResult(final_answer="Failure.", success=False, iterations=1, tool_calls=[], error_message="An error occurred")
    agent.process_goal.return_value = result

    agent._handle_goal("failing goal")

    agent.process_goal.assert_called_once_with("failing goal")
    mock_controller.outbox.display_reasoning_result.assert_called_once_with(result)


def test_handle_goal_exception(fully_integrated_agent):
    """Test that exceptions during goal processing are handled."""
    agent, mock_controller = fully_integrated_agent

    agent.process_goal.side_effect = Exception("A big error")

    agent._handle_goal("failing goal")

    mock_controller.outbox.display_goal_error.assert_called_once_with(
        "failing goal", "Error processing goal: A big error"
    )

def test_spin_single_goal(fully_integrated_agent):
    """Test the agent's main loop for a single goal."""
    agent, mock_controller = fully_integrated_agent
    agent.process_goal.reset_mock()

    # Simulate the inbox providing a single goal and then stopping
    mock_controller.inbox.get_next_goal.side_effect = ["test goal", None]
    
    # Mock the reasoner's output
    reasoning_result = ReasoningResult(final_answer="Done", success=True, iterations=1, tool_calls=[])
    agent.process_goal.return_value = reasoning_result
    
    agent.spin()
    
    agent.process_goal.assert_called_once_with("test goal")
    mock_controller.outbox.display_reasoning_result.assert_called_once_with(reasoning_result)

def test_spin_with_reasoning_error(fully_integrated_agent):
    """Test that errors from the reasoner are caught and displayed."""
    agent, mock_controller = fully_integrated_agent
    agent.process_goal.reset_mock()

    mock_controller.inbox.get_next_goal.side_effect = ["failing goal", None]
    agent.process_goal.side_effect = Exception("A big error")
    
    agent.spin()
    
    mock_controller.outbox.display_goal_error.assert_called_once_with(
        "failing goal", "Error processing goal: A big error"
    )

def test_spin_keyboard_interrupt(fully_integrated_agent):
    """Test that a KeyboardInterrupt is handled gracefully."""
    agent, mock_controller = fully_integrated_agent

    mock_controller.inbox.get_next_goal.side_effect = KeyboardInterrupt
    
    agent.spin()
    
    # The agent should not raise an exception, and should close gracefully.
    # The 'finally' block in spin() calls self.close(), which calls controller.close()
    mock_controller.close.assert_called_once()
