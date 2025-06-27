"""
AWS Lambda function for Discord bot integration.

This Lambda function serves as a Discord interaction endpoint that:
1. Validates Discord requests using signature verification
2. Handles PING interactions for Discord verification
3. Processes application commands through the Discord agent
4. Returns appropriate responses for Lambda/API Gateway
"""

import json
import logging
import os
from typing import Dict, Any

from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError

# Import your agent components
from jentic_agents.platform.jentic_client import JenticClient
from jentic_agents.reasoners.standard_reasoner import StandardReasoner
from jentic_agents.memory.scratch_pad import ScratchPadMemory
from jentic_agents.inbox.discord_inbox import DiscordInbox
from jentic_agents.agents.discord_agent import DiscordAgent

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Discord configuration
DISCORD_PUBLIC_KEY = os.environ.get('DISCORD_PUBLIC_KEY')
DISCORD_APP_ID = os.environ.get('DISCORD_APP_ID')

# Jentic configuration
JENTIC_API_KEY = os.environ.get('JENTIC_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4-turbo')


def verify_discord_signature(event: Dict[str, Any]) -> bool:
    """
    Verify the Discord request signature.
    
    Args:
        event: Lambda event containing headers and body
        
    Returns:
        True if signature is valid, False otherwise
    """
    if not DISCORD_PUBLIC_KEY:
        logger.error("DISCORD_PUBLIC_KEY not configured")
        return False
    
    try:
        headers = event.get('headers', {})
        # Handle case variations in header names
        signature = headers.get('x-signature-ed25519') or headers.get('X-Signature-Ed25519')
        timestamp = headers.get('x-signature-timestamp') or headers.get('X-Signature-Timestamp')
        
        if not signature or not timestamp:
            logger.error("Missing Discord signature headers")
            return False
        
        body = event.get('body', '')
        
        # Create the verify key
        verify_key = VerifyKey(bytes.fromhex(DISCORD_PUBLIC_KEY))
        
        # Create the message to verify
        message = timestamp + body
        
        # Verify the signature
        verify_key.verify(message.encode(), signature=bytes.fromhex(signature))
        return True
        
    except BadSignatureError:
        logger.error("Invalid Discord signature")
        return False
    except Exception as e:
        logger.error(f"Error verifying Discord signature: {e}")
        return False


def handle_ping_interaction() -> Dict[str, Any]:
    """
    Handle Discord PING interaction for verification.
    
    Returns:
        Lambda response with PONG
    """
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({'type': 1})
    }


def handle_application_command(interaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle Discord application command through the agent system.
    
    Args:
        interaction: Discord interaction data
        
    Returns:
        Lambda response
    """
    try:
        logger.info(f"Processing command: {interaction.get('data', {}).get('name', 'unknown')}")
        
        # Initialize Jentic client
        jentic_client = JenticClient()
        
        # Initialize reasoner with OpenAI
        reasoner = StandardReasoner(
            jentic_client=jentic_client,
            model=OPENAI_MODEL
        )
        
        # Initialize memory
        memory = ScratchPadMemory()
        
        # Initialize Discord inbox with the interaction
        inbox = DiscordInbox(
            discord_interaction=interaction,
            app_id=DISCORD_APP_ID
        )
        
        # Create and run the Discord agent
        agent = DiscordAgent(
            reasoner=reasoner,
            memory=memory,
            inbox=inbox,
            jentic_client=jentic_client
        )
        
        # Process the interaction
        result = agent.spin()
        
        logger.info("Command processed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error processing command: {e}", exc_info=True)
        
        # Try to send error response to Discord if possible
        if 'id' in interaction and 'token' in interaction:
            try:
                import requests
                url = f"https://discord.com/api/interactions/{interaction['id']}/{interaction['token']}/callback"
                error_response = {
                    "type": 4,
                    "data": {
                        "content": f"❌ Sorry, I encountered an error: {str(e)}"
                    }
                }
                requests.post(url, json=error_response, timeout=10)
            except Exception as send_error:
                logger.error(f"Failed to send error response to Discord: {send_error}")
        
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for Discord interactions.
    
    Args:
        event: Lambda event from API Gateway
        context: Lambda context
        
    Returns:
        Lambda response
    """
    try:
        logger.info(f"Received Discord interaction: {event.get('httpMethod', 'unknown method')}")
        
        # Verify Discord signature
        if not verify_discord_signature(event):
            logger.error("Invalid Discord signature")
            return {
                'statusCode': 401,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Invalid request signature'})
            }
        
        # Parse the interaction body
        try:
            body = json.loads(event.get('body', '{}'))
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in request body: {e}")
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Invalid JSON'})
            }
        
        # Handle different interaction types
        interaction_type = body.get('type')
        
        if interaction_type == 1:  # PING
            logger.info("Handling PING interaction")
            return handle_ping_interaction()
        
        elif interaction_type == 2:  # APPLICATION_COMMAND
            logger.info("Handling APPLICATION_COMMAND interaction")
            return handle_application_command(body)
        
        else:
            logger.warning(f"Unhandled interaction type: {interaction_type}")
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Unhandled interaction type'})
            }
    
    except Exception as e:
        logger.error(f"Unexpected error in lambda_handler: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Internal server error'})
        } 