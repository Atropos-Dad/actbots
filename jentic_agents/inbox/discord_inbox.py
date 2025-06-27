"""
Discord-based inbox that receives goals from Discord interactions and sends responses back.
"""
import json
import logging
import requests
from typing import Optional, Dict, Any
from .base_inbox import BaseInbox

logger = logging.getLogger(__name__)


class DiscordInbox(BaseInbox):
    """
    Inbox that receives goals from Discord messages and can send responses back.
    
    Handles Discord interaction events and provides a way to respond to users.
    Unlike CLI inbox, this is designed for single-interaction processing.
    """
    
    def __init__(self, discord_interaction: Optional[Dict[str, Any]] = None, app_id: Optional[str] = None):
        """
        Initialize Discord inbox.
        
        Args:
            discord_interaction: The Discord interaction data from the webhook
            app_id: Discord application ID for sending responses
        """
        self.discord_interaction = discord_interaction
        self.app_id = app_id
        self._closed = False
        self._current_goal: Optional[str] = None
        self._interaction_processed = False
        
        # Extract interaction details if provided
        self.interaction_id = None
        self.interaction_token = None
        self.channel_id = None
        self.user_id = None
        
        if discord_interaction:
            self.interaction_id = discord_interaction.get('id')
            self.interaction_token = discord_interaction.get('token')
            self.channel_id = discord_interaction.get('channel_id')
            self.user_id = discord_interaction.get('member', {}).get('user', {}).get('id')
    
    def set_interaction(self, discord_interaction: Dict[str, Any], app_id: str) -> None:
        """
        Set the Discord interaction data for processing.
        
        Args:
            discord_interaction: The Discord interaction data from the webhook
            app_id: Discord application ID for sending responses
        """
        self.discord_interaction = discord_interaction
        self.app_id = app_id
        self.interaction_id = discord_interaction.get('id')
        self.interaction_token = discord_interaction.get('token')
        self.channel_id = discord_interaction.get('channel_id')
        self.user_id = discord_interaction.get('member', {}).get('user', {}).get('id')
        self._interaction_processed = False
        self._closed = False
    
    def get_next_goal(self) -> Optional[str]:
        """
        Get the next goal from Discord interaction.
        
        For Discord inbox, there's typically only one goal per interaction.
        
        Returns:
            The goal string from Discord command, or None if no goal available
        """
        if self._closed or self._interaction_processed or not self.discord_interaction:
            return None
        
        try:
            # Extract the goal from the Discord interaction
            interaction_type = self.discord_interaction.get('type')
            
            if interaction_type == 2:  # APPLICATION_COMMAND
                # Get the command and its options
                command_data = self.discord_interaction.get('data', {})
                command_name = command_data.get('name', '')
                
                # Extract goal from command options
                options = command_data.get('options', [])
                goal = None
                
                # Look for a parameter that contains the user's request
                for option in options:
                    if option.get('name') in ['question', 'query', 'goal', 'message', 'task']:
                        goal = option.get('value', '')
                        break
                
                # If no specific goal parameter, handle based on command type
                if not goal:
                    if command_name == 'help':
                        goal = "Show me how to use this AI agent and what commands are available"
                    elif command_name == 'status':
                        goal = "Check the current status and capabilities of this AI agent"
                    else:
                        goal = f"Help with {command_name}"
                
                self._current_goal = goal
                self._interaction_processed = True
                
                logger.info(f"Received Discord goal: {goal}")
                return goal
            
            else:
                logger.warning(f"Unsupported Discord interaction type: {interaction_type}")
                self._closed = True
                return None
                
        except Exception as e:
            logger.error(f"Error processing Discord interaction: {e}")
            self._closed = True
            return None
    
    def acknowledge_goal(self, goal: str) -> None:
        """
        Acknowledge that a goal has been processed successfully.
        
        For Discord, this means the response was sent successfully.
        
        Args:
            goal: The goal that was successfully processed
        """
        if goal == self._current_goal:
            self._current_goal = None
            logger.info(f"Discord goal acknowledged: {goal}")
    
    def reject_goal(self, goal: str, reason: str) -> None:
        """
        Reject a goal that couldn't be processed.
        
        For Discord, this sends an error message back to the user.
        
        Args:
            goal: The goal that failed to process
            reason: Reason for rejection
        """
        if goal == self._current_goal:
            self._current_goal = None
            
        # Send error message to Discord
        self.send_error_response(f"❌ Sorry, I couldn't process your request: {reason}")
        logger.error(f"Discord goal rejected: {goal} - {reason}")
    
    def send_response(self, message: str) -> bool:
        """
        Send a response message to Discord.
        
        Args:
            message: The message to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.interaction_id or not self.interaction_token:
            logger.error("No Discord interaction data available for response")
            return False
        
        try:
            # Use Discord's interaction callback endpoint
            url = f"https://discord.com/api/interactions/{self.interaction_id}/{self.interaction_token}/callback"
            
            callback_data = {
                "type": 4,  # CHANNEL_MESSAGE_WITH_SOURCE
                "data": {
                    "content": message[:2000]  # Discord message limit
                }
            }
            
            response = requests.post(url, json=callback_data, timeout=10)
            
            if response.status_code == 200:
                logger.info("Discord response sent successfully")
                return True
            else:
                logger.error(f"Discord API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Discord response: {e}")
            return False
    
    def send_loading_response(self, message: str = "🤔 Let me think about that...") -> bool:
        """
        Send an immediate loading response to Discord while processing.
        
        Args:
            message: Loading message to display
            
        Returns:
            True if successful, False otherwise
        """
        return self.send_response(message)
    
    def update_response(self, message: str) -> bool:
        """
        Update the initial response with the final answer.
        
        Args:
            message: Final message to display
            
        Returns:
            True if successful, False otherwise
        """
        if not self.interaction_token or not self.app_id:
            logger.error("No Discord interaction data available for response update")
            return False
        
        try:
            # Use Discord's webhook edit endpoint
            url = f"https://discord.com/api/webhooks/{self.app_id}/{self.interaction_token}/messages/@original"
            
            data = {
                "content": message[:2000]  # Discord message limit
            }
            
            response = requests.patch(url, json=data, timeout=10)
            
            if response.status_code == 200:
                logger.info("Discord response updated successfully")
                return True
            else:
                logger.error(f"Discord API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating Discord response: {e}")
            return False
    
    def send_error_response(self, error_message: str) -> bool:
        """
        Send an error response to Discord.
        
        Args:
            error_message: Error message to send
            
        Returns:
            True if successful, False otherwise
        """
        return self.send_response(error_message)
    
    def has_goals(self) -> bool:
        """
        Check if there are pending goals.
        
        For Discord inbox, this is true only if we have an unprocessed interaction.
        
        Returns:
            True if goals are available, False otherwise
        """
        return not self._closed and not self._interaction_processed and self.discord_interaction is not None
    
    def close(self) -> None:
        """
        Clean up inbox resources.
        """
        self._closed = True
        self.discord_interaction = None
        self.interaction_id = None
        self.interaction_token = None 