"""
Script to register Discord slash commands for the ActBots AI agent.

This script registers commands with Discord's API that users can use to interact
with the AI agent through slash commands.
"""

import os
import requests
import json
from typing import List, Dict, Any

# Discord configuration
DISCORD_APP_ID = os.environ.get('DISCORD_APP_ID')
DISCORD_BOT_TOKEN = os.environ.get('DISCORD_BOT_TOKEN')
DISCORD_GUILD_ID = os.environ.get('DISCORD_GUILD_ID')  # Optional: for guild-specific commands


def create_commands() -> List[Dict[str, Any]]:
    """
    Define the Discord slash commands for the AI agent.
    
    Returns:
        List of command definitions
    """
    commands = [
        {
            "name": "ask",
            "description": "Ask the AI agent a question or give it a task",
            "options": [
                {
                    "name": "question",
                    "description": "Your question or task for the AI agent",
                    "type": 3,  # STRING type
                    "required": True
                }
            ]
        },
        {
            "name": "help",
            "description": "Get help on how to use the AI agent",
            "options": []
        },
        {
            "name": "search",
            "description": "Search for available tools and workflows",
            "options": [
                {
                    "name": "query",
                    "description": "What capability are you looking for?",
                    "type": 3,  # STRING type
                    "required": True
                }
            ]
        },
        {
            "name": "status",
            "description": "Check the status of the AI agent",
            "options": []
        }
    ]
    
    return commands


def register_global_commands(commands: List[Dict[str, Any]]) -> bool:
    """
    Register commands globally (available in all servers).
    
    Note: Global commands can take up to 1 hour to propagate.
    
    Args:
        commands: List of command definitions
        
    Returns:
        True if successful, False otherwise
    """
    if not DISCORD_APP_ID or not DISCORD_BOT_TOKEN:
        print("❌ ERROR: DISCORD_APP_ID and DISCORD_BOT_TOKEN must be set")
        return False
    
    url = f'https://discord.com/api/v10/applications/{DISCORD_APP_ID}/commands'
    
    headers = {
        'Authorization': f'Bot {DISCORD_BOT_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.put(url, headers=headers, json=commands)
        
        if response.status_code == 200:
            print("✅ Global commands registered successfully!")
            print(f"Registered {len(commands)} commands:")
            for cmd in commands:
                print(f"  - /{cmd['name']}: {cmd['description']}")
            return True
        else:
            print(f"❌ Failed to register global commands: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error registering global commands: {e}")
        return False


def register_guild_commands(commands: List[Dict[str, Any]], guild_id: str) -> bool:
    """
    Register commands for a specific guild (server).
    
    Guild commands update instantly and are good for testing.
    
    Args:
        commands: List of command definitions
        guild_id: Discord guild (server) ID
        
    Returns:
        True if successful, False otherwise
    """
    if not DISCORD_APP_ID or not DISCORD_BOT_TOKEN:
        print("❌ ERROR: DISCORD_APP_ID and DISCORD_BOT_TOKEN must be set")
        return False
    
    url = f'https://discord.com/api/v10/applications/{DISCORD_APP_ID}/guilds/{guild_id}/commands'
    
    headers = {
        'Authorization': f'Bot {DISCORD_BOT_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.put(url, headers=headers, json=commands)
        
        if response.status_code == 200:
            print(f"✅ Guild commands registered successfully for guild {guild_id}!")
            print(f"Registered {len(commands)} commands:")
            for cmd in commands:
                print(f"  - /{cmd['name']}: {cmd['description']}")
            return True
        else:
            print(f"❌ Failed to register guild commands: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error registering guild commands: {e}")
        return False


def list_existing_commands(guild_id: str = None) -> None:
    """
    List existing registered commands.
    
    Args:
        guild_id: Optional guild ID to list guild-specific commands
    """
    if not DISCORD_APP_ID or not DISCORD_BOT_TOKEN:
        print("❌ ERROR: DISCORD_APP_ID and DISCORD_BOT_TOKEN must be set")
        return
    
    if guild_id:
        url = f'https://discord.com/api/v10/applications/{DISCORD_APP_ID}/guilds/{guild_id}/commands'
        scope = f"guild {guild_id}"
    else:
        url = f'https://discord.com/api/v10/applications/{DISCORD_APP_ID}/commands'
        scope = "global"
    
    headers = {
        'Authorization': f'Bot {DISCORD_BOT_TOKEN}',
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            commands = response.json()
            print(f"\n📋 Existing {scope} commands:")
            if commands:
                for cmd in commands:
                    print(f"  - /{cmd['name']}: {cmd['description']}")
            else:
                print("  No commands registered")
        else:
            print(f"❌ Failed to list {scope} commands: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error listing {scope} commands: {e}")


def main():
    """Main function to register Discord commands."""
    print("🤖 Discord Bot Command Registration")
    print("=" * 50)
    
    # Check required environment variables
    if not DISCORD_APP_ID:
        print("❌ ERROR: DISCORD_APP_ID environment variable not set")
        print("Get this from your Discord Developer Portal > General Information")
        return
    
    if not DISCORD_BOT_TOKEN:
        print("❌ ERROR: DISCORD_BOT_TOKEN environment variable not set")
        print("Get this from your Discord Developer Portal > Bot")
        return
    
    # Create command definitions
    commands = create_commands()
    
    print(f"📝 Created {len(commands)} command definitions")
    
    # List existing commands first
    if DISCORD_GUILD_ID:
        print(f"\n🔍 Checking existing guild commands for {DISCORD_GUILD_ID}...")
        list_existing_commands(DISCORD_GUILD_ID)
    
    print(f"\n🔍 Checking existing global commands...")
    list_existing_commands()
    
    # Register commands
    if DISCORD_GUILD_ID:
        print(f"\n🚀 Registering guild commands for {DISCORD_GUILD_ID}...")
        guild_success = register_guild_commands(commands, DISCORD_GUILD_ID)
        
        if guild_success:
            print("\n✅ Guild commands registered! You can test them immediately.")
            print(f"💡 TIP: Add your bot to the server using this URL:")
            print(f"https://discord.com/oauth2/authorize?client_id={DISCORD_APP_ID}&scope=applications.commands")
    else:
        print("\n⚠️  DISCORD_GUILD_ID not set. Registering global commands only.")
        print("💡 TIP: Set DISCORD_GUILD_ID for instant command updates during development.")
    
    print(f"\n🌍 Registering global commands...")
    global_success = register_global_commands(commands)
    
    if global_success:
        print("\n✅ Global commands registered! They may take up to 1 hour to appear.")
    
    print("\n" + "=" * 50)
    print("🎉 Command registration completed!")
    
    if DISCORD_GUILD_ID:
        print(f"🔗 Invite bot to server: https://discord.com/oauth2/authorize?client_id={DISCORD_APP_ID}&scope=applications.commands")
    
    print("\n📚 Next steps:")
    print("1. Deploy your Lambda function")
    print("2. Set the Interactions Endpoint URL in Discord Developer Portal")
    print("3. Test the commands in your Discord server")


if __name__ == "__main__":
    main() 