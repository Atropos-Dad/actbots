"""
Unit tests for inbox implementations.
"""

import pytest
from io import StringIO
from unittest.mock import patch, MagicMock
from jentic_agents.communication.inbox.cli_inbox import CLIInbox

class TestCLIInbox:
    """Test cases for CLIInbox"""

    def test_get_next_goal_from_stream(self):
        """Test getting goals from a stream"""
        input_stream = StringIO("goal 1\ngoal 2\n")
        inbox = CLIInbox(input_stream=input_stream)

        assert inbox.get_next_goal() == "goal 1"
        assert inbox.get_next_goal() == "goal 2"
        assert inbox.get_next_goal() is None  # EOF

    @pytest.mark.parametrize("exit_cmd", ["quit", "exit", "q"])
    def test_exit_commands(self, exit_cmd):
        """Test that various exit commands stop the inbox."""
        input_stream = StringIO(f"goal 1\n{exit_cmd}\ngoal 2\n")
        inbox = CLIInbox(input_stream=input_stream)

        assert inbox.get_next_goal() == "goal 1"
        assert inbox.get_next_goal() is None  # Exit command encountered
        assert inbox.get_next_goal() is None  # Inbox should remain closed

    def test_empty_lines_are_ignored(self):
        """Test that empty lines are ignored and the next goal is read."""
        input_stream = StringIO("goal 1\n\ngoal 2\n")
        inbox = CLIInbox(input_stream=input_stream)

        assert inbox.get_next_goal() == "goal 1"
        assert inbox.get_next_goal() == "goal 2" # Empty line is skipped
        assert inbox.get_next_goal() is None

    def test_whitespace_is_stripped(self):
        """Test that leading/trailing whitespace is stripped from goals."""
        input_stream = StringIO("  goal with spaces  \n")
        inbox = CLIInbox(input_stream=input_stream)
        assert inbox.get_next_goal() == "goal with spaces"

    def test_has_goals_after_exit(self):
        """Test has_goals returns False after an exit command."""
        input_stream = StringIO("goal 1\nquit\n")
        inbox = CLIInbox(input_stream=input_stream)

        assert inbox.has_goals() is True
        inbox.get_next_goal()  # Reads "goal 1"
        assert inbox.has_goals() is True
        inbox.get_next_goal()  # Reads "quit", which closes the inbox
        assert inbox.has_goals() is False

    def test_close_stops_goals(self):
        """Test that closing the inbox stops goal retrieval."""
        input_stream = StringIO("goal 1\ngoal 2\n")
        inbox = CLIInbox(input_stream=input_stream)

        assert inbox.has_goals() is True
        inbox.close()
        assert inbox.has_goals() is False
        assert inbox.get_next_goal() is None

    def test_keyboard_interrupt_handling(self):
        """Test that KeyboardInterrupt is handled gracefully."""
        mock_stream = MagicMock()
        mock_stream.readline.side_effect = KeyboardInterrupt
        inbox = CLIInbox(input_stream=mock_stream)

        with patch("jentic_agents.communication.inbox.cli_inbox.console") as mock_console:
            assert inbox.get_next_goal() is None
            assert inbox.has_goals() is False
            mock_console.print.assert_called_with("\n[yellow]Interrupted by user. Goodbye![/yellow]")
