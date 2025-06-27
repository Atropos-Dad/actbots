"""
Procedural memory implementation for storing skills, habits, and procedures.

Procedural memory stores learned skills, automatic behaviors, and step-by-step
procedures that can be executed without conscious thought. This includes:
- Motor skills and muscle memory
- Cognitive procedures and problem-solving strategies
- Habits and automatic behaviors
- Workflows and step-by-step processes
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from .base_memory import BaseMemory
from ..utils.logger import get_logger
from ..utils.block_timer import Timer

logger = get_logger(__name__)


class ProceduralMemory(BaseMemory):
    """
    Procedural memory for storing skills, habits, and procedures.
    
    This memory system is designed to store:
    - Skills and abilities
    - Habits and automatic behaviors
    - Step-by-step procedures and workflows
    - Problem-solving strategies
    - Motor patterns and sequences
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize procedural memory.
        
        Args:
            storage_path: Optional path to persist procedures to disk
        """
        logger.info("Initializing ProceduralMemory")
        
        self._procedures: Dict[str, Dict[str, Any]] = {}
        self._storage_path = storage_path
        
        # Load existing procedures if storage path exists
        if self._storage_path and os.path.exists(self._storage_path):
            with Timer("Load procedures from disk"):
                self._load_from_disk()
        
        logger.info("ProceduralMemory initialized successfully")
    
    def store(self, key: str, value: Any) -> None:
        """
        Store a procedure under the given key.
        
        Args:
            key: Unique identifier for the procedure
            value: Procedure data (steps, skills, habits, etc.)
        """
        logger.debug(f"Storing procedure: {key}")
        
        with Timer(f"Store procedure: {key}"):
            # Create procedure with execution metadata
            procedure = {
                "value": value,
                "type": self._classify_procedure_type(value),
                "steps": self._extract_steps(value),
                "difficulty": self._assess_difficulty(value),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "execution_count": 0,
                "last_executed": None,
                "success_rate": 1.0,  # Start optimistic
                "average_execution_time": None,
                "mastery_level": self._assess_initial_mastery(value),
                "prerequisites": self._extract_prerequisites(value),
                "tags": self._extract_tags(value)
            }
            
            self._procedures[key] = procedure
            
            # Persist to disk if storage path is configured
            if self._storage_path:
                self._save_to_disk()
        
        logger.debug(f"Stored procedure: {key}")
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a procedure by key.
        
        Args:
            key: Unique identifier for the procedure
            
        Returns:
            Stored procedure, or None if key not found
        """
        logger.debug(f"Retrieving procedure: {key}")
        
        with Timer(f"Retrieve procedure: {key}"):
            if key not in self._procedures:
                logger.debug(f"Procedure not found: {key}")
                return None
            
            # Update execution metadata
            procedure = self._procedures[key]
            procedure["execution_count"] += 1
            procedure["last_executed"] = datetime.now(timezone.utc).isoformat()
            
            # Improve mastery with practice (muscle memory effect)
            procedure["mastery_level"] = min(1.0, procedure["mastery_level"] + 0.05)
            
            # Persist updated metadata if storage is configured
            if self._storage_path:
                self._save_to_disk()
            
            return procedure["value"]
    
    def delete(self, key: str) -> bool:
        """
        Delete a procedure.
        
        Args:
            key: Unique identifier for the procedure to delete
            
        Returns:
            True if procedure was deleted, False if key not found
        """
        logger.debug(f"Deleting procedure: {key}")
        
        with Timer(f"Delete procedure: {key}"):
            if key not in self._procedures:
                logger.debug(f"Procedure not found for deletion: {key}")
                return False
            
            del self._procedures[key]
            
            # Persist changes if storage is configured
            if self._storage_path:
                self._save_to_disk()
            
            logger.debug(f"Deleted procedure: {key}")
            return True
    
    def clear(self) -> None:
        """Clear all procedures."""
        logger.info("Clearing all procedures")
        
        with Timer("Clear all procedures"):
            self._procedures.clear()
            
            # Clear persisted storage if configured
            if self._storage_path and os.path.exists(self._storage_path):
                os.remove(self._storage_path)
        
        logger.info("Cleared all procedures")
    
    def keys(self) -> List[str]:
        """
        Get all procedure keys.
        
        Returns:
            List of all keys in procedural memory
        """
        return list(self._procedures.keys())
    
    def get_procedures_by_type(self, procedure_type: str) -> Dict[str, Any]:
        """
        Get all procedures of a specific type.
        
        Args:
            procedure_type: Type of procedure (skill, habit, workflow, etc.)
            
        Returns:
            Dictionary of procedures of the specified type
        """
        logger.debug(f"Retrieving procedures by type: {procedure_type}")
        
        result = {}
        for key, procedure in self._procedures.items():
            if procedure["type"] == procedure_type:
                result[key] = procedure["value"]
        
        logger.debug(f"Found {len(result)} procedures of type: {procedure_type}")
        return result
    
    def get_procedures_by_mastery(self, min_mastery: float = 0.0, max_mastery: float = 1.0) -> Dict[str, Any]:
        """
        Get procedures within a mastery level range.
        
        Args:
            min_mastery: Minimum mastery level (0.0 to 1.0)
            max_mastery: Maximum mastery level (0.0 to 1.0)
            
        Returns:
            Dictionary of procedures within the mastery range
        """
        result = {}
        for key, procedure in self._procedures.items():
            mastery = procedure["mastery_level"]
            if min_mastery <= mastery <= max_mastery:
                result[key] = procedure["value"]
        
        return result
    
    def get_procedures_by_difficulty(self, difficulty: str) -> Dict[str, Any]:
        """
        Get procedures by difficulty level.
        
        Args:
            difficulty: Difficulty level (easy, medium, hard)
            
        Returns:
            Dictionary of procedures with the specified difficulty
        """
        result = {}
        for key, procedure in self._procedures.items():
            if procedure["difficulty"] == difficulty:
                result[key] = procedure["value"]
        
        return result
    
    def execute_procedure(self, key: str, execution_time: float = None, success: bool = True) -> Optional[Any]:
        """
        Execute a procedure and update performance metrics.
        
        Args:
            key: Procedure identifier
            execution_time: Time taken to execute (in seconds)
            success: Whether execution was successful
            
        Returns:
            Procedure value or None if not found
        """
        logger.debug(f"Executing procedure: {key}")
        
        if key not in self._procedures:
            return None
        
        procedure = self._procedures[key]
        
        # Update execution statistics
        procedure["execution_count"] += 1
        procedure["last_executed"] = datetime.now(timezone.utc).isoformat()
        
        # Update success rate (moving average)
        current_rate = procedure["success_rate"]
        execution_count = procedure["execution_count"]
        new_rate = ((current_rate * (execution_count - 1)) + (1.0 if success else 0.0)) / execution_count
        procedure["success_rate"] = new_rate
        
        # Update average execution time if provided
        if execution_time is not None:
            if procedure["average_execution_time"] is None:
                procedure["average_execution_time"] = execution_time
            else:
                # Moving average
                current_avg = procedure["average_execution_time"]
                procedure["average_execution_time"] = (
                    (current_avg * (execution_count - 1) + execution_time) / execution_count
                )
        
        # Update mastery based on success and practice
        if success:
            procedure["mastery_level"] = min(1.0, procedure["mastery_level"] + 0.02)
        else:
            procedure["mastery_level"] = max(0.0, procedure["mastery_level"] - 0.01)
        
        # Persist changes
        if self._storage_path:
            self._save_to_disk()
        
        return procedure["value"]
    
    def get_procedure_stats(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get performance statistics for a procedure.
        
        Args:
            key: Procedure identifier
            
        Returns:
            Statistics dictionary or None if not found
        """
        if key not in self._procedures:
            return None
        
        procedure = self._procedures[key]
        return {
            "type": procedure["type"],
            "difficulty": procedure["difficulty"],
            "execution_count": procedure["execution_count"],
            "success_rate": procedure["success_rate"],
            "mastery_level": procedure["mastery_level"],
            "average_execution_time": procedure["average_execution_time"],
            "last_executed": procedure["last_executed"],
            "created_at": procedure["created_at"]
        }
    
    def get_mastery_overview(self) -> Dict[str, Any]:
        """
        Get overview of mastery levels across all procedures.
        
        Returns:
            Dictionary with mastery statistics
        """
        if not self._procedures:
            return {
                "total_procedures": 0,
                "average_mastery": 0.0,
                "mastery_distribution": {},
                "most_practiced": None,
                "highest_mastery": None
            }
        
        mastery_levels = [p["mastery_level"] for p in self._procedures.values()]
        execution_counts = [(k, p["execution_count"]) for k, p in self._procedures.items()]
        
        # Mastery distribution
        distribution = {"beginner": 0, "intermediate": 0, "advanced": 0, "expert": 0}
        for mastery in mastery_levels:
            if mastery < 0.25:
                distribution["beginner"] += 1
            elif mastery < 0.5:
                distribution["intermediate"] += 1
            elif mastery < 0.75:
                distribution["advanced"] += 1
            else:
                distribution["expert"] += 1
        
        # Most practiced procedure
        most_practiced = max(execution_counts, key=lambda x: x[1]) if execution_counts else None
        
        # Highest mastery procedure
        highest_mastery = max(
            self._procedures.items(),
            key=lambda x: x[1]["mastery_level"]
        ) if self._procedures else None
        
        return {
            "total_procedures": len(self._procedures),
            "average_mastery": sum(mastery_levels) / len(mastery_levels),
            "mastery_distribution": distribution,
            "most_practiced": {
                "key": most_practiced[0],
                "count": most_practiced[1]
            } if most_practiced else None,
            "highest_mastery": {
                "key": highest_mastery[0],
                "level": highest_mastery[1]["mastery_level"]
            } if highest_mastery else None
        }
    
    def _classify_procedure_type(self, value: Any) -> str:
        """Classify the type of procedure."""
        if isinstance(value, dict):
            if "steps" in value or "procedure" in value:
                return "workflow"
            elif "trigger" in value or "condition" in value:
                return "habit"
            elif "skill" in value or "ability" in value:
                return "skill"
            else:
                return "procedure"
        elif isinstance(value, str):
            text_lower = value.lower()
            if "step" in text_lower or "first" in text_lower or "then" in text_lower:
                return "workflow"
            elif "when" in text_lower or "if" in text_lower:
                return "habit"
            elif "how to" in text_lower:
                return "skill"
            else:
                return "procedure"
        else:
            return "procedure"
    
    def _extract_steps(self, value: Any) -> List[str]:
        """Extract procedural steps from the value."""
        steps = []
        
        if isinstance(value, dict):
            if "steps" in value:
                steps = value["steps"] if isinstance(value["steps"], list) else [value["steps"]]
            elif "procedure" in value:
                proc = value["procedure"]
                if isinstance(proc, list):
                    steps = proc
                elif isinstance(proc, str):
                    steps = [proc]
        elif isinstance(value, str):
            # Simple step extraction
            text = value.lower()
            if "step" in text:
                # Split by step indicators
                parts = text.split("step")
                steps = [part.strip() for part in parts[1:] if part.strip()]
            elif any(word in text for word in ["first", "then", "next", "finally"]):
                steps = [value]  # Treat as single step with sequence indicators
        
        return steps
    
    def _extract_prerequisites(self, value: Any) -> List[str]:
        """Extract prerequisites or dependencies."""
        prereqs = []
        
        if isinstance(value, dict) and "prerequisites" in value:
            prereqs = value["prerequisites"]
            if not isinstance(prereqs, list):
                prereqs = [prereqs]
        
        return prereqs
    
    def _extract_tags(self, value: Any) -> List[str]:
        """Extract tags or categories."""
        tags = []
        
        if isinstance(value, dict) and "tags" in value:
            tags = value["tags"] if isinstance(value["tags"], list) else [value["tags"]]
        
        # Add automatic tags based on content
        text = str(value).lower()
        if "physical" in text or "motor" in text:
            tags.append("physical")
        if "cognitive" in text or "mental" in text:
            tags.append("cognitive")
        if "social" in text or "communication" in text:
            tags.append("social")
        
        return tags
    
    def _assess_difficulty(self, value: Any) -> str:
        """Assess the difficulty level of a procedure."""
        if isinstance(value, dict) and "difficulty" in value:
            return value["difficulty"]
        
        # Simple heuristic based on content
        text = str(value).lower()
        step_count = len(self._extract_steps(value))
        
        if step_count > 10 or any(word in text for word in ["complex", "advanced", "expert"]):
            return "hard"
        elif step_count > 5 or any(word in text for word in ["intermediate", "moderate"]):
            return "medium"
        else:
            return "easy"
    
    def _assess_initial_mastery(self, value: Any) -> float:
        """Assess initial mastery level."""
        if isinstance(value, dict) and "mastery" in value:
            return float(value["mastery"])
        
        # Start with low mastery for new procedures
        text = str(value).lower()
        if any(word in text for word in ["simple", "basic", "easy"]):
            return 0.3
        elif any(word in text for word in ["familiar", "known"]):
            return 0.5
        else:
            return 0.1  # Start low for most procedures
    
    def _save_to_disk(self) -> None:
        """Save procedures to disk."""
        if not self._storage_path:
            return
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
            
            with open(self._storage_path, 'w', encoding='utf-8') as f:
                json.dump(self._procedures, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save procedures to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load procedures from disk."""
        if not self._storage_path or not os.path.exists(self._storage_path):
            return
        
        try:
            with open(self._storage_path, 'r', encoding='utf-8') as f:
                self._procedures = json.load(f)
                
            logger.info(f"Loaded {len(self._procedures)} procedures from disk")
            
        except Exception as e:
            logger.error(f"Failed to load procedures from disk: {e}")
            self._procedures = {} 