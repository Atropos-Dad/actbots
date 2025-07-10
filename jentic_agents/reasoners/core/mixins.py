"""
Reusable behavior mixins for reasoners.

These mixins provide specific capabilities that can be mixed into any reasoner:
- EscalationMixin: Human intervention capabilities
- ToolExecutionMixin: Advanced tool execution features  
- MemoryIntegrationMixin: Enhanced memory operations
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from ...utils.logger import get_logger

logger = get_logger(__name__)


class EscalationMixin:
    """
    Mixin providing human escalation capabilities.
    
    Adds the ability to:
    - Process LLM responses for escalation requests
    - Request human help directly
    - Add human guidance to prompts
    - Store human responses in memory
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize escalation capabilities."""
        super().__init__(*args, **kwargs)
        self._last_escalation_question: Optional[str] = None
    
    def process_llm_response_for_escalation(
        self, response: str, context: str = ""
    ) -> str:
        """
        Check if LLM response contains escalation request and handle it.
        
        Looks for XML escalation patterns in the LLM response and processes them.
        
        Args:
            response: LLM response to check for escalation requests
            context: Additional context for the human
            
        Returns:
            Processed response (either original or human response if escalation occurred)
        """
        response = response.strip()
        
        # Check for XML escalation pattern
        escalation_pattern = (
            r'<escalate_to_human\s+reason="([^"]+)"\s+question="([^"]+)"\s*/>'
        )
        match = re.search(escalation_pattern, response)
        
        if match:
            reason = match.group(1).strip()
            question = match.group(2).strip()
            logger.info(f"🤖➡️👤 LLM requested escalation: {reason}")
            
            # Store the question for later reference
            self._last_escalation_question = question
            
            if self.intervention_hub.is_available():
                try:
                    human_response = self.intervention_hub.ask_human(question, context)
                    if human_response.strip():
                        logger.info(f"👤➡️🤖 Human provided response: {human_response}")
                        
                        # Store human guidance in memory for future reference
                        self._store_human_guidance(human_response, question)
                        
                        return human_response
                    else:
                        logger.warning("👤 No response from human, continuing with original")
                except Exception as e:
                    logger.warning(f"Escalation failed: {e}")
            else:
                logger.warning("⚠️ Escalation requested but not available")
            
            # Remove the escalation tag from the response
            return re.sub(escalation_pattern, "", response).strip()
        
        return response
    
    def request_human_help(self, question: str, context: str = "") -> str:
        """
        Direct method for requesting human help from anywhere in the code.
        
        Args:
            question: Question to ask the human
            context: Additional context for the human
            
        Returns:
            Human response or empty string if not available
        """
        logger.info(f"🤖➡️👤 Direct escalation request: {question}")
        
        if self.intervention_hub.is_available():
            try:
                response = self.intervention_hub.ask_human(question, context)
                logger.info("👤➡️🤖 Human response received")
                
                # Store the guidance in memory
                self._store_human_guidance(response, question)
                
                return response
            except Exception as e:
                logger.warning(f"Direct escalation failed: {e}")
        else:
            logger.warning("⚠️ Direct escalation requested but not available")
        
        return ""
    
    def add_human_guidance_to_prompt(self, base_prompt: str) -> str:
        """
        Add recent human guidance from memory to prompts.
        
        Args:
            base_prompt: The original prompt
            
        Returns:
            Prompt with human guidance appended if available
        """
        try:
            # Get latest human guidance from memory
            latest_guidance = self.memory.retrieve("human_guidance_latest")
            if latest_guidance and latest_guidance.strip():
                guidance_section = f"\n\nRECENT HUMAN GUIDANCE: {latest_guidance}\n"
                return base_prompt + guidance_section
        except (KeyError, AttributeError):
            # No human guidance in memory yet
            pass
        return base_prompt
    
    def _store_human_guidance(self, response: str, question: str) -> None:
        """Store human guidance in memory for future reference."""
        try:
            # Create unique key for this guidance
            guidance_key = f"human_guidance_{len(getattr(self.memory, 'keys', lambda: [])())}"
            
            # Store with description if enhanced memory available
            if hasattr(self.memory, 'set'):
                self.memory.set(
                    key=guidance_key,
                    value=response,
                    description=f"Human guidance for: {question}",
                )
                # Also store the latest guidance under a well-known key
                self.memory.set(
                    key="human_guidance_latest",
                    value=response,
                    description=f"Latest human guidance: {question}",
                )
            else:
                # Fallback to basic memory interface
                self.memory.store(guidance_key, response)
                self.memory.store("human_guidance_latest", response)
                
            logger.info(f"Stored human guidance in memory: {guidance_key}")
            
        except Exception as e:
            logger.warning(f"Failed to store human guidance: {e}")


class ToolExecutionMixin:
    """
    Mixin providing advanced tool execution capabilities.
    
    Adds the ability to:
    - Resolve tool IDs from memory
    - Validate tool parameters comprehensively
    - Format tool results consistently
    - Handle tool loading and caching
    """
    
    def resolve_tool_id_from_memory(self, tool_id: str) -> str:
        """
        If tool_id is a memory key, resolve it to the actual tool UUID.
        
        Args:
            tool_id: Tool ID or memory key containing tool ID
            
        Returns:
            Actual tool ID for execution
        """
        try:
            if hasattr(self.memory, 'keys') and tool_id in self.memory.keys():
                stored = self.memory.retrieve(tool_id)
                if isinstance(stored, dict) and "id" in stored:
                    resolved_id = stored["id"]
                    logger.info(f"Resolved memory key '{tool_id}' to tool_id: {resolved_id}")
                    return resolved_id
                else:
                    logger.warning(
                        f"Memory key '{tool_id}' did not resolve to a valid tool_id. Using as-is."
                    )
        except Exception as e:
            logger.debug(f"Memory resolution failed for {tool_id}: {e}")
        
        return tool_id
    
    def validate_tool_parameters(
        self, params: Dict[str, Any], tool_info: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Validate tool parameters against tool schema.
        
        Args:
            params: Parameters to validate
            tool_info: Tool information including schema
            
        Returns:
            Tuple of (validated_params, error_message)
        """
        try:
            # Check required fields
            required_fields = tool_info.get("required", [])
            missing_fields = [field for field in required_fields if field not in params]
            
            if missing_fields:
                error_msg = f"Missing required fields: {missing_fields}"
                logger.warning(error_msg)
                return params, error_msg
            
            # Validate placeholders if memory supports it
            if hasattr(self.memory, 'validate_placeholders'):
                error, correction_prompt = self.memory.validate_placeholders(
                    params, required_fields
                )
                if error:
                    logger.warning(f"Parameter validation failed: {error}")
                    return params, error
            
            logger.debug("Parameter validation successful")
            return params, None
            
        except Exception as e:
            error_msg = f"Parameter validation error: {str(e)}"
            logger.error(error_msg)
            return params, error_msg
    
    def format_tool_result(self, result: Any) -> str:
        """
        Format tool execution result for display or further processing.
        
        Args:
            result: Raw tool execution result
            
        Returns:
            Formatted result string
        """
        try:
            if isinstance(result, dict):
                if "status" in result and result["status"] == "success":
                    inner_result = result.get("result", "Success")
                    return f"Tool executed successfully. Result: {inner_result}"
                elif "error" in result:
                    return f"Tool execution failed. Error: {result['error']}"
                else:
                    return f"Tool result: {result}"
            
            elif hasattr(result, "success"):
                if result.success:
                    output = getattr(result, "output", "Success")
                    return f"Tool executed successfully. Output: {output}"
                else:
                    error = getattr(result, "error", "Unknown error")
                    return f"Tool execution failed. Error: {error}"
            
            else:
                return f"Tool result: {str(result)}"
                
        except Exception as e:
            logger.warning(f"Failed to format tool result: {e}")
            return f"Tool execution completed. Raw result: {str(result)[:200]}"
    
    def load_tool_with_caching(self, tool_id: str) -> Dict[str, Any]:
        """
        Load tool information with basic caching to avoid repeated loads.
        
        Args:
            tool_id: ID of the tool to load
            
        Returns:
            Tool information dictionary
        """
        # Simple caching using memory if available
        cache_key = f"_tool_cache_{tool_id}"
        
        try:
            if hasattr(self.memory, 'retrieve'):
                cached_info = self.memory.retrieve(cache_key)
                if cached_info:
                    logger.debug(f"Using cached tool info for {tool_id}")
                    return cached_info
        except (KeyError, AttributeError):
            pass
        
        # Load fresh tool information
        logger.debug(f"Loading fresh tool info for {tool_id}")
        tool_info = self.jentic_client.load(tool_id)
        
        # Cache the result if memory supports it
        try:
            if hasattr(self.memory, 'store'):
                self.memory.store(cache_key, tool_info)
                logger.debug(f"Cached tool info for {tool_id}")
        except Exception as e:
            logger.debug(f"Failed to cache tool info: {e}")
        
        return tool_info


class MemoryIntegrationMixin:
    """
    Mixin providing enhanced memory integration capabilities.
    
    Adds the ability to:
    - Store data with rich descriptions
    - Enumerate memory for prompts  
    - Resolve complex memory references
    - Manage memory lifecycle
    """
    
    def store_with_description(
        self, key: str, value: Any, description: str, type_hint: Optional[str] = None
    ) -> None:
        """
        Store data in memory with rich metadata.
        
        Args:
            key: Memory key
            value: Value to store
            description: Human-readable description
            type_hint: Optional type information
        """
        try:
            if hasattr(self.memory, 'set'):
                # Use enhanced memory interface
                self.memory.set(key, value, description, type_hint)
            else:
                # Fallback to basic interface
                self.memory.store(key, value)
                
            logger.debug(f"Stored in memory: {key} = {description}")
            
        except Exception as e:
            logger.warning(f"Failed to store in memory: {e}")
    
    def enumerate_memory_for_prompt(self) -> str:
        """
        Get formatted memory enumeration suitable for LLM prompts.
        
        Returns:
            Formatted memory summary
        """
        try:
            if hasattr(self.memory, 'enumerate_for_prompt'):
                return self.memory.enumerate_for_prompt()
            else:
                # Fallback implementation
                if hasattr(self.memory, 'keys'):
                    keys = self.memory.keys()
                    if not keys:
                        return "(memory empty)"
                    
                    lines = ["Available memory:"]
                    for key in keys[:10]:  # Limit for prompt size
                        try:
                            value = self.memory.retrieve(key)
                            preview = str(value)[:100]
                            if len(str(value)) > 100:
                                preview += "..."
                            lines.append(f"• {key}: {preview}")
                        except Exception:
                            lines.append(f"• {key}: <unavailable>")
                    
                    return "\n".join(lines)
                else:
                    return "Memory status unknown."
                    
        except Exception as e:
            logger.debug(f"Failed to enumerate memory: {e}")
            return "Memory enumeration failed."
    
    def resolve_memory_references(self, text: str) -> str:
        """
        Resolve memory references in text using placeholder syntax.
        
        Args:
            text: Text that may contain memory references
            
        Returns:
            Text with memory references resolved
        """
        try:
            if hasattr(self.memory, 'resolve_placeholders'):
                return self.memory.resolve_placeholders(text)
            else:
                # Basic fallback - look for ${memory.key} patterns
                import re
                pattern = r'\$\{memory\.([^}]+)\}'
                
                def replace_ref(match):
                    key = match.group(1)
                    try:
                        value = self.memory.retrieve(key)
                        return str(value)
                    except Exception:
                        return f"${{{key}_NOT_FOUND}}"
                
                return re.sub(pattern, replace_ref, text)
                
        except Exception as e:
            logger.warning(f"Memory reference resolution failed: {e}")
            return text
    
    def clear_temporary_memory(self, prefix: str = "_temp_") -> int:
        """
        Clear temporary memory entries with a given prefix.
        
        Args:
            prefix: Prefix of keys to clear
            
        Returns:
            Number of keys cleared
        """
        cleared = 0
        try:
            if hasattr(self.memory, 'keys') and hasattr(self.memory, 'delete'):
                keys_to_clear = [
                    key for key in self.memory.keys() 
                    if key.startswith(prefix)
                ]
                
                for key in keys_to_clear:
                    if self.memory.delete(key):
                        cleared += 1
                        
                logger.debug(f"Cleared {cleared} temporary memory entries")
                
        except Exception as e:
            logger.warning(f"Failed to clear temporary memory: {e}")
        
        return cleared