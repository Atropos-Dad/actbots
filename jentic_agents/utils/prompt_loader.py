"""
Utility for loading prompts from the filesystem.
"""
import json
from pathlib import Path
from typing import Any, Dict
from functools import lru_cache

from .logger import get_logger

logger = get_logger(__name__)

# Global cache for prompt contents to avoid repeated file I/O
_prompt_cache: Dict[str, Any] = {}


def load_prompt(prompt_name: str) -> Any:
    """Load a prompt from the prompts directory. Return JSON if file is JSON, else string."""
    # Check cache first
    if prompt_name in _prompt_cache:
        logger.debug(f"Loading prompt '{prompt_name}' from cache")
        return _prompt_cache[prompt_name]
    
    # Prompts are stored relative to the 'jentic_agents' package root.
    # This makes the loader independent of where it's called from.
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.txt"
    try:
        logger.debug(f"Loading prompt '{prompt_name}' from file: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("{"):
                try:
                    parsed_content = json.loads(content)
                    _prompt_cache[prompt_name] = parsed_content
                    return parsed_content
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from prompt file: {prompt_path}")
                    logger.error(f"--- FAULTY PROMPT CONTENT ---\n{content}\n")
                    raise e  # Re-raise the original error after logging
            
            _prompt_cache[prompt_name] = content
            return content
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_path}")
        raise RuntimeError(f"Prompt file not found: {prompt_path}")


def clear_prompt_cache() -> None:
    """Clear the prompt cache. Useful for testing or memory management."""
    global _prompt_cache
    _prompt_cache.clear()
    logger.debug("Prompt cache cleared")


def get_cache_size() -> int:
    """Get the current number of cached prompts."""
    return len(_prompt_cache)


def get_cached_prompt_names() -> list[str]:
    """Get a list of currently cached prompt names."""
    return list(_prompt_cache.keys())


# Alternative implementation using functools.lru_cache for automatic size management
@lru_cache(maxsize=128)
def load_prompt_with_lru(prompt_name: str) -> Any:
    """
    Load a prompt using LRU cache with automatic size management.
    This is an alternative to the manual cache above.
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("{"):
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from prompt file: {prompt_path}")
                    logger.error(f"--- FAULTY PROMPT CONTENT ---\n{content}\n")
                    raise e
            return content
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_path}")
        raise RuntimeError(f"Prompt file not found: {prompt_path}") 
    