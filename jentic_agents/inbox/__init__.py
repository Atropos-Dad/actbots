"""Inbox implementations for delivering goals to agents."""

from .base_inbox import BaseInbox
from .cli_inbox import CLIInbox
from .discord_inbox import DiscordInbox

__all__ = [
    "BaseInbox",
    "CLIInbox", 
    "DiscordInbox",
]
