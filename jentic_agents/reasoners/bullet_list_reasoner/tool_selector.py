"""Tool selection logic for BulletPlanReasoner."""

import json
import re
from typing import Any, Dict, List, Optional

import logging
from ...utils.logger import get_logger
from ...utils.async_helpers import safe_llm_call as _global_safe_llm_call
from ...utils.prompt_loader import load_prompt
from .reasoner_state import Step, ReasonerState

logger = get_logger(__name__)


class ToolSelector:
    """Handles tool search and LLM-based selection for BulletPlanReasoner."""

    def __init__(self, jentic_client, memory, llm, search_top_k: int = 10):
        self.jentic_client = jentic_client
        self.memory = memory
        self.llm = llm
        self.search_top_k = search_top_k

    def select_tool(self, step: Step, state: ReasonerState) -> str:
        """Select the best tool for a given plan step."""
        logger.info(f"Selecting tool for step: {step.text}")

        # Fast path: Check for 'Execute <memory_key>' pattern
        tool_id = self._check_execute_pattern(step)
        if tool_id:
            return tool_id

        # Search for candidate tools
        search_query = self._build_search_query(step, state)
        search_hits = self._search_tools(search_query)

        if not search_hits:
            raise RuntimeError(f"No tools found for query: '{search_query}'")

        # Prioritize tools mentioned in step text
        prioritized_hits = self._prioritize_by_provider(search_hits, step)

        # Use LLM to select best tool
        selected_tool_id = self._select_with_llm(step, prioritized_hits, state)
        
        if not selected_tool_id:
            raise RuntimeError("LLM tool selection failed: No valid tool selected")

        return selected_tool_id

    def _check_execute_pattern(self, step: Step) -> Optional[str]:
        """Check if step follows 'Execute <memory_key>' pattern and return cached tool_id."""
        # Pattern 1: Simple "execute memory_key" format
        exec_match = re.match(r"execute\s+([\w\-_]+)", step.text.strip(), re.IGNORECASE)
        if exec_match:
            mem_key = exec_match.group(1)
            if mem_key in self.memory.keys():
                stored = self.memory.retrieve(mem_key)
                if isinstance(stored, dict) and "id" in stored:
                    logger.info(f"Reusing tool_id from memory key '{mem_key}': {stored['id']}")
                    return stored["id"]
        
        # Pattern 2: "Execute the 'memory_key'" format  
        exec_match_quoted = re.search(r"execute\s+(?:the\s+)?['\"]([^'\"]+)['\"]", step.text.strip(), re.IGNORECASE)
        if exec_match_quoted:
            mem_key = exec_match_quoted.group(1)
            if mem_key in self.memory.keys():
                stored = self.memory.retrieve(mem_key)
                if isinstance(stored, dict) and "id" in stored:
                    logger.info(f"Reusing tool_id from memory key '{mem_key}' (quoted format): {stored['id']}")
                    return stored["id"]
        
        return None

    def _build_search_query(self, step: Step, state: ReasonerState) -> str:
        """Build search query for tool discovery using keyword extraction."""
        try:
            kw_template = load_prompt("keyword_extraction")

            # Combine step text with goal and history for better context
            history_str = "\n".join(state.history)
            contextual_text = (
                f"Goal: {state.goal}\n\n"
                f"History of previous steps:\n{history_str}\n\n"
                f"Current step to find a tool for:\n{step.text}"
            )

            if isinstance(kw_template, dict):
                kw_template["inputs"]["context_text"] = contextual_text
                prompt = json.dumps(kw_template, ensure_ascii=False)
            else:
                prompt = kw_template.format(context_text=contextual_text)

            # Add human guidance context if available
            context_aware_prompt = self._add_human_guidance_to_prompt(prompt)
            reply = self._safe_llm_call([{"role": "user", "content": context_aware_prompt}]).strip()
            
            if reply:
                logger.info("LLM keyword-extraction produced query: %s", reply)
                return reply
            else:
                raise RuntimeError("LLM keyword-extraction returned empty query.")
                
        except Exception as e:
            logger.error(f"Keyword-extraction prompt failed: {e}")
            raise RuntimeError(f"Keyword-extraction prompt failed: {e}")

    def _search_tools(self, query: str) -> List[Dict[str, Any]]:
        """Search for tools using the query."""
        logger.info(f"Search query: {query}")
        search_hits = self.jentic_client.search(query, top_k=self.search_top_k)
        logger.info(f"Found {len(search_hits)} tool candidates")
        return search_hits

    def _prioritize_by_provider(self, hits: List[Dict[str, Any]], step: Step) -> List[Dict[str, Any]]:
        """Sort candidates so provider mentioned in step text comes first."""
        step_text_lower = step.text.lower()

        def provider_mentioned(hit):
            api_name = (
                hit.get("api_name", "").lower()
                if isinstance(hit, dict)
                else getattr(hit, "api_name", "").lower()
            )
            if not api_name:
                return False
            domain_part = api_name.split(".")[0]
            return (api_name in step_text_lower) or (domain_part in step_text_lower)

        return sorted(hits, key=lambda h: not provider_mentioned(h))

    def _select_with_llm(self, step: Step, hits: List[Dict[str, Any]], state: ReasonerState) -> Optional[str]:
        """Use LLM to select the best tool from candidates."""
        if not hits:
            return None

        # Build numbered candidate list
        numbered_lines = self._build_candidate_list(hits)
        candidate_block = "\n".join(numbered_lines)
        if logger.isEnabledFor(__import__("logging").DEBUG):
            logger.debug(f"Tools returned from keyword search:\n{candidate_block}")

        # Fill the prompt
        prompt = self._build_selection_prompt(step, state, candidate_block, hits)
        raw_reply = self._safe_llm_call([{"role": "user", "content": prompt}]).strip()
        logger.debug("LLM tool-selection reply: %s", raw_reply)

        # Parse LLM response for tool selection
        return self._parse_selection_response(raw_reply, hits)

    def _build_candidate_list(self, hits: List[Dict[str, Any]]) -> List[str]:
        """Build numbered list of tool candidates."""
        numbered_lines = []
        for idx, hit in enumerate(hits, 1):
            name = hit.get("name") if isinstance(hit, dict) else getattr(hit, "name", None)
            if not name or not str(name).strip():
                name = (
                    hit.get("id", "Unknown")
                    if isinstance(hit, dict)
                    else getattr(hit, "id", "Unknown")
                )
            api_name = (
                hit.get("api_name")
                if isinstance(hit, dict)
                else getattr(hit, "api_name", None)
            )
            desc = (
                hit.get("description", "")
                if isinstance(hit, dict)
                else getattr(hit, "description", "")
            )
            display = f"{name} ({api_name})" if api_name else name
            numbered_lines.append(f"{idx}. {display} — {desc}")
        return numbered_lines

    def _build_selection_prompt(self, step: Step, state: ReasonerState, candidate_block: str, hits: List[Dict[str, Any]] = None) -> str:
        """Build the tool selection prompt."""
        # Generate context analysis with hits for dynamic API domain detection
        context_analysis = self._analyze_context(step, state, hits)
        
        # Generate workflow state
        workflow_state = self._build_workflow_state(step, state)
        
        prompt_tpl = load_prompt("select_tool")
        if isinstance(prompt_tpl, dict):
            prompt_tpl["inputs"].update({
                "goal": state.goal,
                "plan_step": step.text,
                "memory_keys": ", ".join(self.memory.keys()),
                "tool_candidates": candidate_block,
                "context_analysis": context_analysis,
                "workflow_state": workflow_state,
            })
            prompt = json.dumps(prompt_tpl, ensure_ascii=False)
        else:
            prompt = prompt_tpl.format(
                goal=state.goal,
                plan_step=step.text,
                memory_keys=", ".join(self.memory.keys()),
                tool_candidates=candidate_block,
                context_analysis=context_analysis,
                workflow_state=workflow_state,
            )
        return prompt

    def _analyze_context(self, step: Step, state: ReasonerState, hits: List[Dict[str, Any]] = None) -> str:
        """Analyze the context to provide structured information for tool selection."""
        # Extract action type from step text
        action_type = "unknown"
        step_lower = step.text.lower()
        
        if any(word in step_lower for word in ["send", "post", "create", "add"]):
            action_type = "create/send"
        elif any(word in step_lower for word in ["get", "retrieve", "fetch", "find", "search"]):
            action_type = "read/search"
        elif any(word in step_lower for word in ["update", "modify", "edit", "change"]):
            action_type = "update"
        elif any(word in step_lower for word in ["delete", "remove", "ban"]):
            action_type = "delete"
        
        # Extract API domain dynamically from available tools
        api_domain = "unknown"
        confidence = "low"
        
        if hits:
            # Check which API domains are mentioned in step text
            for hit in hits:
                api_name = (
                    hit.get("api_name", "").lower()
                    if isinstance(hit, dict)
                    else getattr(hit, "api_name", "").lower()
                )
                if api_name:
                    domain_part = api_name.split(".")[0]
                    if (api_name in step_lower) or (domain_part in step_lower):
                        api_domain = domain_part
                        confidence = "high"
                        break
            
            # If no direct mention, try to infer from most common domain in hits
            if api_domain == "unknown" and hits:
                domain_counts = {}
                for hit in hits:
                    api_name = (
                        hit.get("api_name", "")
                        if isinstance(hit, dict)
                        else getattr(hit, "api_name", "")
                    )
                    if api_name:
                        domain = api_name.split(".")[0].lower()
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
                
                if domain_counts:
                    api_domain = max(domain_counts, key=domain_counts.get)
                    confidence = "medium"
        
        # Determine workflow complexity
        workflow_complexity = "single-step" if len(state.plan) <= 2 else "multi-step"
        
        # Available data from memory
        available_data = list(self.memory.keys()) if self.memory.keys() else ["none"]
        
        return f"action_type: {action_type}, api_domain: {api_domain} ({confidence} confidence), workflow_complexity: {workflow_complexity}, available_data: {', '.join(available_data)}"

    def _build_workflow_state(self, step: Step, state: ReasonerState) -> str:
        """Build workflow state information for tool selection."""
        # Previous steps
        completed_steps = [s.text for s in state.plan if s.status.name == "DONE"]
        current_step_index = state.plan.index(step) if step in state.plan else 0
        
        # Memory contents summary
        memory_summary = f"{len(self.memory.keys())} items stored" if self.memory.keys() else "empty"
        
        # Step dependencies (basic analysis)
        dependencies = "none"
        if step.store_key:
            dependencies = f"stores result as '{step.store_key}'"
        
        return f"previous_steps: {len(completed_steps)} completed, memory: {memory_summary}, step_position: {current_step_index + 1}/{len(state.plan)}, dependencies: {dependencies}"

    def _parse_selection_response(self, response: str, hits: List[Dict[str, Any]]) -> Optional[str]:
        """Parse LLM response to extract selected tool ID."""
        # 1. Try numeric selection
        match = re.search(r"(\d+)", response)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(hits):
                chosen = hits[idx]
                tool_name = chosen.get("name") if isinstance(chosen, dict) else getattr(chosen, "name", None)
                tool_desc = chosen.get("description", "") if isinstance(chosen, dict) else getattr(chosen, "description", "")
                logger.info(f"LLM chose tool: {tool_name} — {tool_desc}")
                return (
                    chosen["id"]
                    if isinstance(chosen, dict)
                    else getattr(chosen, "id", "unknown")
                )

        # 2. Try ID or name substring match
        lower_response = response.lower()
        for hit in hits:
            hit_id = hit["id"] if isinstance(hit, dict) else getattr(hit, "id", "")
            hit_name = hit.get("name", "") if isinstance(hit, dict) else getattr(hit, "name", "")
            
            if hit_id and hit_id.lower() in lower_response:
                tool_name = hit.get("name") if isinstance(hit, dict) else getattr(hit, "name", None)
                tool_desc = hit.get("description", "") if isinstance(hit, dict) else getattr(hit, "description", "")
                logger.info(f"LLM chose tool: {tool_name} — {tool_desc}")
                return hit_id
                
            if hit_name and hit_name.lower() in lower_response:
                tool_name = hit.get("name") if isinstance(hit, dict) else getattr(hit, "name", None)
                tool_desc = hit.get("description", "") if isinstance(hit, dict) else getattr(hit, "description", "")
                logger.info(f"LLM chose tool: {tool_name} — {tool_desc}")
                return hit["id"] if isinstance(hit, dict) else getattr(hit, "id", "unknown")

        return None

    def _safe_llm_call(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Thin wrapper around the shared *safe_llm_call* utility."""
        return _global_safe_llm_call(self.llm, messages, **kwargs)

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