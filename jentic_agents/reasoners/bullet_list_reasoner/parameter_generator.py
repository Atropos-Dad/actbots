"""Parameter generation and validation logic for BulletPlanReasoner."""

import json
from typing import Any, Dict, List, Optional, Tuple

from ...utils.logger import get_logger
from ...utils.prompt_loader import load_prompt
from ...utils.parsing_helpers import safe_json_loads
from .reasoner_state import ReasonerState

logger = get_logger(__name__)


class ParameterGenerator:
    """Handles LLM-based parameter generation and validation for tools."""

    def __init__(self, memory, llm, max_retries: int = 3):
        self.memory = memory
        self.llm = llm
        self.max_retries = max_retries

    def generate_and_validate_parameters(
        self, tool_id: str, tool_info: Dict[str, Any], state: ReasonerState
    ) -> Dict[str, Any]:
        """Generate and validate parameters for a tool using LLM."""
        logger.info(f"Generating parameters for tool: {tool_id}")
        
        required_fields = tool_info.get("required", [])
        initial_prompt = self._prepare_param_generation_prompt(tool_id, tool_info, state)
        
        last_error = None
        current_prompt = initial_prompt

        for attempt in range(self.max_retries):
            logger.info(f"Parameter generation attempt {attempt + 1}/{self.max_retries}")
            
            # Get LLM response
            args_json = self._safe_llm_call([{"role": "user", "content": current_prompt}])
            logger.info(f"LLM parameter response:\n{args_json}")

            # Validate the response
            args, error, correction_prompt = self._validate_llm_params(args_json, required_fields)
            
            if not error:
                logger.info("Parameter validation successful.")
                return args  # Success
            
            # Handle validation error
            last_error = error
            if correction_prompt:
                if "ERROR:" in correction_prompt:
                    current_prompt = f"{correction_prompt} Original goal was: {state.goal}"
                else:
                    current_prompt += correction_prompt

        raise RuntimeError(
            f"Parameter generation failed after {self.max_retries} attempts. Last error: {last_error}"
        )

    def _prepare_param_generation_prompt(
        self, tool_id: str, tool_info: Dict[str, Any], state: ReasonerState
    ) -> str:
        """Prepare the parameter generation prompt for LLM."""
        required_fields = tool_info.get("required", [])
        memory_enum = self.memory.enumerate_for_prompt()
        available_memory_keys = list(self.memory.keys())
        allowed_memory_keys_str = ", ".join(available_memory_keys) if available_memory_keys else "(none)"

        def _escape_braces(text: str) -> str:
            return text.replace("{", "{{").replace("}", "}}")

        tool_schema_str = str(tool_info)
        param_generation_template = load_prompt("param_generation")

        if isinstance(param_generation_template, dict):
            # Complex prompt building for JSON templates
            param_generation_template["inputs"].update({
                "tool_id": tool_id,
                "selected_operation": _escape_braces(tool_schema_str),
                "memory": _escape_braces(memory_enum),
                "goal": state.goal,
                "allowed_memory_keys": allowed_memory_keys_str,
            })
            if "instruction" in param_generation_template:
                param_generation_template["instruction"] = param_generation_template["instruction"].replace(
                    "{allowed_memory_keys}", allowed_memory_keys_str
                )
            if "rules" in param_generation_template:
                param_generation_template["rules"] = [
                    rule.replace("{allowed_memory_keys}", allowed_memory_keys_str) 
                    if isinstance(rule, str) else rule
                    for rule in param_generation_template["rules"]
                ]
            prompt = json.dumps(param_generation_template, ensure_ascii=False)
        else:
            # Simple string formatting
            prompt = param_generation_template.format(
                tool_id=tool_id,
                selected_operation=_escape_braces(tool_schema_str),
                memory=_escape_braces(memory_enum),
                goal=state.goal,
                allowed_memory_keys=allowed_memory_keys_str,
            )
        
        logger.info(f"Available memory keys for parameter filling: {available_memory_keys}")
        return self._add_human_guidance_to_prompt(prompt)

    def _validate_llm_params(
        self, args_json: str, required_fields: List[str]
    ) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
        """
        Parse and validate LLM-generated parameters.
        Returns (parsed_args, error_message, correction_prompt).
        """
        # 1. Parse JSON
        try:
            args = safe_json_loads(args_json)
        except ValueError as e:
            logger.error(f"Failed to parse JSON args: {e}")
            correction_prompt = self._build_correction_prompt(
                error_type='json_error',
                error_details=str(e),
                failed_parameters=args_json,
                required_fields=required_fields
            )
            return None, f"Invalid JSON: {e}", correction_prompt

        # 2. Check for missing required fields
        missing_fields = [field for field in required_fields if field not in args]
        if missing_fields:
            error = f"Missing required fields: {missing_fields}"
            logger.warning(f"{error}. Re-prompting LLM.")
            correction_prompt = self._build_correction_prompt(
                error_type='missing_fields',
                error_details=f"Missing fields: {missing_fields}",
                failed_parameters=json.dumps(args, indent=2),
                required_fields=required_fields
            )
            return args, error, correction_prompt

        # 3. Validate placeholders via memory class
        error, correction_prompt = self.memory.validate_placeholders(args, required_fields)
        return args, error, correction_prompt

    def resolve_memory_placeholders(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve memory placeholders in parameters."""
        try:
            resolved = self.memory.resolve_placeholders(params)
            logger.debug(f"Resolved placeholders: {resolved}")
            return resolved
        except Exception as e:
            logger.warning(f"Failed to resolve memory placeholders: {e}")
            return params

    def _resolve_tool_id_from_memory(self, tool_id: str) -> str:
        """Resolve placeholders in a stored tool ID reference."""
        resolved = self.resolve_memory_placeholders({"id": tool_id})
        return resolved.get("id", tool_id)

    def _safe_llm_call(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Delegate to shared cached safe_llm_call utility for consistency."""
        from ...utils.llm import safe_llm_call as _safe_call
        return _safe_call(self.llm, messages, **kwargs)

    def _build_correction_prompt(self, error_type: str, error_details: str, failed_parameters: str, required_fields: List[str]) -> str:
        """Build a correction prompt for parameter validation errors."""
        correction_template = load_prompt("param_correction_prompt")
        
        if isinstance(correction_template, dict):
            import copy
            correction_template = copy.deepcopy(correction_template)
            correction_template.get('context', {}).update({
                'error_type': error_type,
                'error_details': error_details,
                'failed_parameters': failed_parameters,
                'required_fields': ', '.join(required_fields),
                'available_memory_keys': ', '.join(self.memory.keys())
            })
            return json.dumps(correction_template, ensure_ascii=False)
        else:
            return correction_template.format(
                error_type=error_type,
                error_details=error_details,
                failed_parameters=failed_parameters,
                required_fields=', '.join(required_fields),
                available_memory_keys=', '.join(self.memory.keys())
            )

    def _add_human_guidance_to_prompt(self, base_prompt: str) -> str:
        """Add recent human guidance from memory to prompts."""
        try:
            latest_guidance = self.memory.retrieve("human_guidance_latest")
            if latest_guidance and latest_guidance.strip():
                guidance_section = f"\n\nRECENT HUMAN GUIDANCE: {latest_guidance}\n"
                return base_prompt + guidance_section
        except KeyError:
            pass
        return base_prompt 