"""
Discord agent that handles Discord-specific interaction patterns.
"""
import logging
from typing import Any, Dict

from .base_agent import BaseAgent
from ..reasoners.base_reasoner import ReasoningResult
from ..inbox.discord_inbox import DiscordInbox

logger = logging.getLogger(__name__)


class DiscordAgent(BaseAgent):
    """
    Discord-specific agent that handles Discord interaction patterns.
    
    Supports sending immediate loading responses and updating them with final answers.
    Designed for single-interaction processing in a serverless environment.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the Discord agent."""
        super().__init__(*args, **kwargs)
        self._running = False
    
    def spin(self) -> Dict[str, Any]:
        """
        Process a single Discord interaction.
        
        Unlike the CLI agent's continuous loop, this processes one interaction
        and returns a response suitable for Lambda.
        
        Returns:
            Dict containing the Lambda response
        """
        logger.info("Starting DiscordAgent interaction processing")
        
        if not isinstance(self.inbox, DiscordInbox):
            error_msg = "DiscordAgent requires DiscordInbox"
            logger.error(error_msg)
            return self._create_error_response(error_msg)
        
        self._running = True
        
        try:
            goal = self.inbox.get_next_goal()
            
            if goal is None:
                logger.warning("No goal received from Discord interaction")
                return self._create_error_response("No goal received")
            
            # Send immediate loading response
            self.inbox.send_loading_response("🤔 Let me think about that...")
            
            # Process the goal
            result = self.process_goal(goal)
            
            # Format and send the final response
            response_message = self._format_response(result)
            
            # Update the loading message with the final response
            success = self.inbox.update_response(response_message)
            
            if success:
                # Acknowledge successful processing
                self.inbox.acknowledge_goal(goal)
                logger.info(f"Successfully processed Discord goal: {goal}")
            else:
                # If update failed, try sending as new message
                self.inbox.send_response(response_message)
                logger.warning("Response update failed, sent as new message")
            
            return self._create_success_response()
                
        except Exception as e:
            error_msg = f"Error processing Discord goal: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Try to send error response to Discord
            if hasattr(self, 'inbox') and isinstance(self.inbox, DiscordInbox):
                self.inbox.send_error_response(f"❌ Sorry, I encountered an error: {str(e)}")
            
            return self._create_error_response(error_msg)
        
        finally:
            self._running = False
            if hasattr(self, 'inbox'):
                self.inbox.close()
            logger.info("DiscordAgent interaction processing completed")
    
    def handle_input(self, input_data: Any) -> str:
        """
        Handle input from Discord interaction.
        
        Args:
            input_data: Discord interaction data
            
        Returns:
            Processed goal string
        """
        if isinstance(input_data, dict):
            # Extract goal from Discord interaction
            command_data = input_data.get('data', {})
            options = command_data.get('options', [])
            
            for option in options:
                if option.get('name') in ['question', 'query', 'goal', 'message', 'task']:
                    return str(option.get('value', '')).strip()
            
            # Handle commands without parameters
            command_name = command_data.get('name', 'unknown command')
            if command_name == 'help':
                return "Show me how to use this AI agent and what commands are available"
            elif command_name == 'status':
                return "Check the current status and capabilities of this AI agent"
            else:
                return f"Help with {command_name}"
        
        return str(input_data).strip()
    
    def handle_output(self, result: ReasoningResult) -> None:
        """
        Handle output for Discord.
        
        This method is not used in the normal flow since Discord responses
        are handled directly through the DiscordInbox methods.
        
        Args:
            result: Reasoning result to present
        """
        # For Discord agent, output is handled through inbox methods
        # This method is kept for interface compatibility
        logger.debug("handle_output called (not used in Discord flow)")
    
    def should_continue(self) -> bool:
        """
        Determine if the agent should continue processing.
        
        For Discord agent, this is always False after processing one interaction.
        
        Returns:
            Always False for single-interaction processing
        """
        return False
    
    def stop(self) -> None:
        """Stop the agent gracefully."""
        self._running = False
    
    def _format_response(self, result: ReasoningResult) -> str:
        """
        Format the reasoning result for Discord display.
        
        Args:
            result: The reasoning result to format
            
        Returns:
            Formatted message string
        """
        if result.success:
            message = f"✅ **Answer:** {result.final_answer}"
            
            if result.tool_calls:
                message += f"\n\n📋 **Used {len(result.tool_calls)} tool(s) in {result.iterations} iteration(s):**"
                for i, call in enumerate(result.tool_calls, 1):
                    tool_name = call.get('tool_name', call.get('tool_id', 'Unknown'))
                    message += f"\n  {i}. {tool_name}"
        else:
            message = f"❌ **Failed:** {result.final_answer}"
            if result.error_message:
                message += f"\n   Error: {result.error_message}"
        
        # Ensure message fits Discord's limit
        return message[:2000]
    
    def _create_success_response(self) -> Dict[str, Any]:
        """
        Create a successful Lambda response.
        
        Returns:
            Lambda response dict
        """
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': '{"status": "success"}'
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create an error Lambda response.
        
        Args:
            error_message: Error message to include
            
        Returns:
            Lambda response dict
        """
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': f'{{"status": "error", "message": "{error_message}"}}'
        } 