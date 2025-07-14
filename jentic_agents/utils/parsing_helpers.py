"""
Parsing and cleaning utilities for LLM outputs and data structures.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union


def extract_fenced_code(text: str, language: str = "") -> str:
    """
    Extract code from markdown-style fenced code blocks.

    Args:
        text: Text that may contain fenced code blocks
        language: Expected language identifier (optional)

    Returns:
        Code content without fences, or original text if no fences found
    """
    # First try to find language-specific fenced blocks
    if language:
        pattern = rf"```{re.escape(language)}\s*\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fall back to generic fenced blocks
    pattern = r"```(?:\w+)?\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try single backticks as fallback
    pattern = r"`([^`]+)`"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()

    return text.strip()


def strip_backtick_fences(text: str) -> str:
    """Remove markdown code fences from text."""
    if not text:
        return text
    
    # Remove fenced code blocks
    text = re.sub(r'```[\w]*\n', '', text)
    text = re.sub(r'\n```', '', text)
    text = re.sub(r'```', '', text)
    
    return text.strip()


def safe_json_loads(text: str) -> Any:
    """
    Safely parse JSON from text, handling common LLM formatting issues.

    Args:
        text: Text that should contain JSON

    Returns:
        Parsed JSON object

    Raises:
        ValueError: If JSON cannot be parsed
    """
    if not text:
        raise ValueError("Empty text provided")

    # Clean up common LLM formatting issues
    text = strip_backtick_fences(text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to fix common issues
        
        # Remove trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Fix single quotes to double quotes (but be careful with contractions)
        text = re.sub(r"(?<!\\)'([^']*?)(?<!\\)'", r'"\1"', text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse JSON: {e}")


def cleanse(obj: Any) -> Any:
    """
    Convert an object to a JSON-serializable format.
    
    This function handles common non-serializable types and attempts to
    preserve as much information as possible while ensuring JSON compatibility.
    
    Args:
        obj: Object to cleanse
        
    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None
    
    # Handle primitive types
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Handle collections
    if isinstance(obj, (list, tuple)):
        return [cleanse(item) for item in obj]
    
    if isinstance(obj, dict):
        return {str(k): cleanse(v) for k, v in obj.items()}
    
    # Handle complex objects that need special serialization
    if hasattr(obj, '__dict__'):
        # For objects with attributes, convert to dict representation
        try:
            if hasattr(obj, 'model_dump'):
                # Pydantic models
                return cleanse(obj.model_dump())
            elif hasattr(obj, 'dict'):
                # Pydantic v1 models
                return cleanse(obj.dict())
            elif hasattr(obj, '__dict__'):
                # Generic objects - convert attributes to dict
                return cleanse(obj.__dict__)
        except Exception:
            # If serialization methods fail, fall back to string representation
            pass
    
    # For anything else, convert to string
    try:
        return str(obj)
    except Exception:
        return f"<{type(obj).__name__} object>"


def make_json_serializable(obj: Any) -> Any:
    """
    Convert complex objects to JSON-serializable formats.
    
    This is specifically designed to handle objects like OperationResult
    that may contain nested non-serializable data.
    
    Args:
        obj: Object to make JSON-serializable
        
    Returns:
        JSON-serializable version of the object
    """
    try:
        # Test if it's already serializable
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # Object is not serializable, convert it
        return cleanse(obj)


def unwrap_singleton_json(obj: Any) -> Any:
    """
    If obj is a single-item list/dict containing JSON data, unwrap it.
    
    Args:
        obj: Object to potentially unwrap
        
    Returns:
        Unwrapped object or original object if unwrapping not applicable
    """
    if isinstance(obj, list) and len(obj) == 1:
        return obj[0]
    
    if isinstance(obj, dict) and len(obj) == 1:
        key, value = next(iter(obj.items()))
        # Only unwrap if the key suggests it's a wrapper
        if key.lower() in ('result', 'data', 'response', 'output'):
            return value
    
    return obj



