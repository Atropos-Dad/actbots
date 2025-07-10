"""
Integration tests for the InteractiveCLIAgent.
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

    This fixture mocks the dependencies of the agent to test its internal
    logic without making external calls.
    """
    # Mock the controller and its components
    mock_controller = MagicMock(spec=CLIController)
    mock_controller.inbox = MagicMock()
    mock_controller.outbox = MagicMock()
    mock_controller.intervention_hub = MagicMock()
    
    # Mock the agent's direct dependencies
    mock_reasoner = MagicMock()
    mock_jentic_client = MagicMock()
    mock_memory = MagicMock()

    # Assemble the agent with mocked components
    agent = InteractiveCLIAgent(
        controller=mock_controller,
        reasoner=mock_reasoner,
        jentic_client=mock_jentic_client,
        memory=mock_memory,
    )
    
    # Isolate the agent's goal processing logic by mocking process_goal
    agent.process_goal = MagicMock()
    
    return agent, mock_controller

# --- Integration-like Tests ---

def test_agent_handles_single_goal_correctly(fully_integrated_agent):
    """
    Tests a full agent loop for a single goal, from receiving input
    to displaying the final output.
    """
    agent, mock_controller = fully_integrated_agent
    
    # --- Arrange ---
    # 1. Simulate the inbox providing a single goal and then stopping
    mock_controller.inbox.get_next_goal.side_effect = ["test goal", None]
    
    # 2. Define the expected result from the mocked processing logic
    reasoning_result = ReasoningResult(
        final_answer="The goal was processed successfully.", 
        success=True, 
        iterations=1, 
        tool_calls=[]
    )
    agent.process_goal.return_value = reasoning_result

    # --- Act ---
    # Run the agent's main loop
    agent.spin()

    # --- Assert ---
    # 1. The agent's processing logic was called with the correct goal
    agent.process_goal.assert_called_once_with("test goal")

    # 2. The outbox was called to display the final result
    mock_controller.outbox.display_reasoning_result.assert_called_once_with(reasoning_result)

    # 3. The inbox was acknowledged (if it has the method)
    mock_controller.inbox.acknowledge_goal.assert_called_once_with("test goal")

def test_agent_handles_processing_exception_gracefully(fully_integrated_agent):
    """
    Tests that the agent correctly handles an exception raised during
    goal processing and reports the error.
    """
    agent, mock_controller = fully_integrated_agent

    # --- Arrange ---
    # 1. Simulate the inbox providing a failing goal
    mock_controller.inbox.get_next_goal.side_effect = ["failing goal", None]

    # 2. Mock the processing logic to raise an error
    agent.process_goal.side_effect = Exception("A critical error occurred")

    # --- Act ---
    # Run the agent's main loop
    agent.spin()

    # --- Assert ---
    # 1. The agent's processing logic was called
    agent.process_goal.assert_called_once_with("failing goal")

    # 2. The outbox was called to display the error, not the result
    mock_controller.outbox.display_goal_error.assert_called_once_with(
        "failing goal", "Error processing goal: A critical error occurred"
    )
    mock_controller.outbox.display_reasoning_result.assert_not_called()

    # 3. The inbox goal was rejected (if it has the method)
    mock_controller.inbox.reject_goal.assert_called_once() 