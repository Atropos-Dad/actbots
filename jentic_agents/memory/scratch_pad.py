"""
Simple dictionary-based memory implementation for development and testing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional
from .base_memory import BaseMemory
from ..utils.parsing_helpers import cleanse
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class MemoryItem:
    key: str
    value: Any
    description: str
    type: str | None = None
    schema: Dict[str, Any] | None = None

    def as_prompt_line(self) -> str:
        part_type = f" ({self.type})" if self.type else ""
        return f"• {self.key}{part_type} – {self.description}"


class ScratchPadMemory(BaseMemory):
    """
    Simple in-memory storage using a dictionary.

    This is suitable for development, testing, and single-session use cases.
    Data is lost when the process terminates.

    Supports both simple key-value storage and enhanced memory items with
    descriptions, types, and placeholder resolution.
    """
    
    _MEMORY_RE = re.compile(r"\$\{memory\.([\w._\[\]]+)\}")

    def __init__(self):
        """Initialize empty scratch pad memory."""
        self._storage: dict[str, Any] = {}  # Simple key-value storage
        self._store: Dict[str, MemoryItem] = {}  # Enhanced memory items
        # Cache for resolved placeholders (dotted path → string representation)
        self._placeholder_cache: Dict[str, str] = {}

    # ---------- BaseMemory interface ----------

    def store(self, key: str, value: Any) -> None:
        """
        Store a value under the given key.

        Args:
            key: Unique identifier for the stored value
            value: Data to store
        """
        self._storage[key] = value
        self._placeholder_cache.clear()  # invalidate cache

    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.

        Args:
            key: Unique identifier for the value

        Returns:
            Stored value, or None if key not found
        """
        if key in self._store:
            return self._store[key].value
        return self._storage.get(key)

    def delete(self, key: str) -> bool:
        """
        Delete a stored value.

        Args:
            key: Unique identifier for the value to delete

        Returns:
            True if value was deleted, False if key not found
        """
        deleted_simple = key in self._storage
        deleted_enhanced = key in self._store

        if deleted_simple:
            del self._storage[key]
        if deleted_enhanced:
            del self._store[key]

        if deleted_simple or deleted_enhanced:
            self._placeholder_cache.clear()

        return deleted_simple or deleted_enhanced

    def clear(self) -> None:
        """
        Clear all stored values.
        """
        self._storage.clear()
        self._store.clear()
        self._placeholder_cache.clear()

    def keys(self) -> list[str]:
        """
        Get all stored keys.

        Returns:
            List of all keys in memory
        """
        # Combine keys from both storage systems
        all_keys = set(self._storage.keys()) | set(self._store.keys())
        return list(all_keys)

    def __len__(self) -> int:
        """Return number of stored items."""
        all_keys = set(self._storage.keys()) | set(self._store.keys())
        return len(all_keys)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in storage."""
        return key in self._storage or key in self._store

    # ---------- Enhanced memory API ----------

    def set(
        self,
        key: str,
        value: Any,
        description: str,
        type_: str | None = None,
        schema: Dict[str, Any] | None = None,
    ) -> None:
        """
        Store a value with metadata as a MemoryItem.

        Args:
            key: Unique identifier for the stored value
            value: Data to store
            description: Human-readable description of the data
            type_: Optional type information
            schema: Optional schema information
        """
        self._store[key] = MemoryItem(key, value, description, type_, schema)

    def get(self, key: str) -> Any:
        """
        Get a value from enhanced storage.

        Args:
            key: Unique identifier for the value

        Returns:
            Stored value from MemoryItem

        Raises:
            KeyError: If key not found in enhanced storage
        """
        return self._store[key].value

    def has(self, key: str) -> bool:
        """
        Check if key exists in enhanced storage.

        Args:
            key: Unique identifier to check

        Returns:
            True if key exists in enhanced storage
        """
        return key in self._store

    def enumerate_for_prompt(self) -> str:
        """
        Generate a formatted string listing all memory items for LLM prompts.

        Each line now includes a short (<=200-character) JSON/text preview of the
        stored value so the model can actually use the data, not just the key.
        """
        if not self._store:
            return "(memory empty)"
        lines = ["Available memory:"]
        for item in self._store.values():
            # Build a compact preview of the value
            val = item.value
            try:
                preview = json.dumps(val, ensure_ascii=False)
            except (TypeError, ValueError):
                preview = str(val)
            if len(preview) > 200:
                preview = preview[:200] + "…"
            part_type = f" ({item.type})" if item.type else ""
            lines.append(f"• {item.key}{part_type} – {preview}  // {item.description}")
        return "\n".join(lines)

    def can_resolve(self, dotted_path: str) -> bool:
        """
        Check if a dotted path can be resolved without raising an error.
        Args:
            dotted_path: Path like "key.subkey.index"
        Returns:
            True if the path can be resolved, False otherwise.
        """
        try:
            # We don't need the result, just to see if _lookup raises an exception.
            self._lookup(dotted_path)
            return True
        except (KeyError, IndexError):
            return False

    def validate_placeholders(self, args: Dict[str, Any], required_fields: list) -> tuple[Optional[str], Optional[str]]:
        """
        Validates placeholders in a dictionary of arguments.
        Returns a tuple of (error_message, correction_prompt).
        """
        error_summary = []
        is_unrecoverable = False

        for param, value in args.items():
            if isinstance(value, str):
                for match in self._MEMORY_RE.finditer(value):
                    path = match.group(1)
                    if not self.can_resolve(path):
                        error_summary.append(f"param '{param}' references invalid memory path '${{memory.{path}}}'")
                        if param in required_fields:
                            is_unrecoverable = True
        
        if not error_summary:
            return None, None

        full_error = f"Placeholder validation failed: {', '.join(error_summary)}."
        if is_unrecoverable:
            raise RuntimeError(f"A required parameter is missing from memory. {full_error}. Available keys: [{', '.join(self.keys())}].")

        # Logic for generating a correction prompt
        from ..utils.prompt_loader import load_prompt
        logger.warning(f"Optional parameter placeholder failed. {full_error}. Attempting local correction.")
        correction_template = load_prompt("param_correction_prompt")
        failed_params_str = json.dumps(args, indent=2)
        
        if isinstance(correction_template, dict):
            correction_template.get('context', {})['failed_parameters'] = failed_params_str
            correction_template.get('context', {})['available_memory_keys'] = ", ".join(self.keys())
            correction_prompt = json.dumps(correction_template, ensure_ascii=False)
        else:
            correction_prompt = correction_template.format(failed_params=failed_params_str, available_memory_keys=", ".join(self.keys()))
        
        return full_error, correction_prompt

    # ---------- Placeholder resolution ----------

    def resolve_placeholders(self, obj: Any) -> Any:
        """
        Recursively substitute ${memory.foo.bar} placeholders inside *obj*.

        Args:
            obj: Object that may contain memory placeholders

        Returns:
            Object with placeholders resolved to actual memory values
        """
        if isinstance(obj, str):
            return self._MEMORY_RE.sub(lambda m: self._lookup(m.group(1)), obj)
        if isinstance(obj, list):
            return [self.resolve_placeholders(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self.resolve_placeholders(v) for k, v in obj.items()}
        return cleanse(obj)

    # ---------- Private helpers ----------

    def _lookup(self, dotted_path: str) -> str:
        """
        Look up a value using a path that can include dots and array indices.
        Args:
            dotted_path: Path like "key.subkey[0].field"
        Returns:
            String representation of the looked up value
        """
        # Shortcut – cached value
        if dotted_path in self._placeholder_cache:
            return self._placeholder_cache[dotted_path]

        # Use a regex to split the path into keys and indices
        path_parts = re.split(r'\.|\[|\]', dotted_path)
        path_parts = [p for p in path_parts if p]  # Remove empty strings

        key, *path_parts = path_parts

        # Get the top-level item from memory
        if key in self._store:
            item = self._store[key].value
        elif key in self._storage:
            item = self._storage[key]
        else:
            raise KeyError(f"Memory key '{key}' not found for path '{dotted_path}'")

        # Traverse the remaining path
        current = item
        for part in path_parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    index = int(part)
                    if 0 <= index < len(current):
                        current = current[index]
                    else:
                        current = None  # Index out of bounds
                except (ValueError, TypeError):
                    current = None  # Part is not a valid index
            else:
                current = None  # Cannot traverse further

            if current is None:
                raise KeyError(f"Path '{dotted_path}' could not be resolved at part '{part}'")

        # Safely stringify the final result
        try:
            resolved_str = json.dumps(current)
        except (TypeError, ValueError):
            resolved_str = str(current)

        # Cache and return
        self._placeholder_cache[dotted_path] = resolved_str
        return resolved_str
