"""Refactored BulletPlanReasoner - Component-based orchestrator."""

from typing import Any, Dict, List, Optional
import json

from ..base_reasoner import BaseReasoner, ReasoningResult, StepType
from ...utils.config import get_bullet_plan_config_value
from ...utils.logger import get_logger
from ...utils.prompt_loader import load_prompt
from ...utils.parsing_helpers import extract_fenced_code, make_json_serializable

from .reasoner_state import ReasonerState, Step, StepStatus
from .plan_parser import BulletPlanParser
from .tool_selector import ToolSelector
from .parameter_generator import ParameterGenerator
from .step_executor import StepExecutor
from .reflection_engine import ReflectionEngine

logger = get_logger(__name__)


class BulletPlanReasoner(BaseReasoner):
    """Refactored BulletPlanReasoner using component-based architecture."""

    def __init__(
        self,
        jentic,
        memory,
        llm=None,
        model=None,
        max_iters=20,
        search_top_k=10,
        intervention_hub=None,
        **kwargs
    ):
        """Initialize the BulletPlanReasoner with component dependency injection."""
        super().__init__(
            jentic_client=jentic,
            memory=memory,
            llm=llm,
            model=model,
            intervention_hub=intervention_hub,
            max_iterations=max_iters,
            **kwargs
        )
        
        # Load configuration
        self.max_iters = max_iters or get_bullet_plan_config_value("max_iterations", 20)
        self.search_top_k = search_top_k or get_bullet_plan_config_value("search_top_k", 10)
        
        # Initialize components
        self.plan_parser = BulletPlanParser()
        self.tool_selector = ToolSelector(
            jentic_client=jentic,
            memory=memory,
            llm=self.llm,
            search_top_k=self.search_top_k
        )
        self.parameter_generator = ParameterGenerator(
            memory=memory,
            llm=self.llm,
            max_retries=get_bullet_plan_config_value("parameter_generation_retries", 3)
        )
        self.step_executor = StepExecutor(
            jentic_client=jentic,
            memory=memory,
            llm=self.llm,
            intervention_hub=intervention_hub
        )
        self.reflection_engine = ReflectionEngine(
            memory=memory,
            llm=self.llm,
            intervention_hub=intervention_hub
        )
        
        # Initialize instance logger for error handling in methods
        self.logger = logger
        
        logger.info(f"BulletPlanReasoner initialized with component architecture")

    def run(self, goal: str, max_iterations: int = None) -> ReasoningResult:
        """Execute the reasoning loop using component orchestration."""
        max_iterations = max_iterations or self.max_iters
        
        logger.info(f"Starting reasoning for goal: {goal} | Max iterations: {max_iterations}")
        
        # Initialize state  
        state = ReasonerState(goal=goal)
        
        # Reset base class state for fresh run
        self.reset_state()
        
        iteration = 0
        while iteration < max_iterations:
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            state = state.next_iteration()

            # Check termination conditions
            if state.goal_completed:
                logger.info("Goal marked as completed! Breaking from loop")
                break

            if state.failed:
                logger.error("Plan has failed and could not be recovered. Terminating loop.")
                break

            # Generate plan if needed
            if not state.plan:
                if iteration > 0:
                    # Check if goal is actually completed
                    if self._evaluate_goal_completion(state):
                        state = state.mark_completed()
                        break
                    else:
                        logger.info("Goal not yet complete. Generating a new plan.")
                
                state = self._generate_plan(state)
                if not state.plan:
                    logger.error("Planning resulted in an empty plan. Cannot proceed.")
                    break

            # Execute next step
            current_step = state.current_step
            if not current_step or current_step.status not in (StepStatus.PENDING, StepStatus.RUNNING):
                logger.warning(f"Skipping step with unexpected status: {current_step}")
                state = state.advance_plan()
                continue

            logger.info(f"Executing step: {current_step.text}")

            try:
                # Classify and execute step
                step_type = self._classify_step(current_step)
                result = self._execute_step(current_step, step_type, state)
                
                # Process result and update state
                state = self._process_step_result(current_step, result, state)
                
            except Exception as e:
                logger.error(f"Step execution failed: {e}")
                state = self._handle_step_failure(current_step, str(e), state)

            iteration += 1

        # Generate final result
        if state.goal_completed:
            final_answer = "Goal completed successfully!"
        else:
            final_answer = "Unable to complete goal."
            
        success = state.goal_completed and not state.failed
        error_message = None if success else "Max iterations reached or failure during steps"
        
        logger.info(f"Reasoning loop complete. Success: {success}")
        if state.goal_completed:
            executed_tools = self.jentic_client.get_executed_tools()
            if executed_tools:
                logger.info(f"Tools executed: {[f'{tool['name']} ({tool['id']})' for tool in executed_tools]}")
        
        return self.create_reasoning_result(
            final_answer=final_answer,
            success=success,
            error_message=error_message
        )

    def _generate_plan(self, state: ReasonerState) -> ReasonerState:
        """Generate a new plan using the plan parser."""
        logger.info("=== PLAN PHASE ===")
        
        bullet_plan_template = load_prompt("bullet_plan")
        if isinstance(bullet_plan_template, dict):
            bullet_plan_template["inputs"]["goal"] = state.goal
            prompt = json.dumps(bullet_plan_template, ensure_ascii=False)
        else:
            prompt = bullet_plan_template.format(goal=state.goal)
            
        logger.debug(f"Planning prompt:\n{prompt}")
        response = self.safe_llm_call([{"role": "user", "content": prompt}])
        logger.info(f"LLM planning response:\n{response}")
        
        # Extract and parse plan
        plan_md = extract_fenced_code(response)
        plan = self.plan_parser.parse(plan_md)
        
        logger.info(f"Generated plan with {len(plan)} steps:")
        for i, step in enumerate(plan):
            logger.info(f"  Step {i+1}: {step.text}")
            
        history_entry = f"Plan generated ({len(plan)} steps)"
        return state.with_plan(plan).with_history(history_entry)

    def _classify_step(self, step: Step) -> str:
        """Classify step type for execution routing."""
        if hasattr(step, 'step_type') and step.step_type:
            return step.step_type.upper()
            
        # Rule-based classification
        text_lower = step.text.lower()
        
        tool_verbs = ["send", "post", "create", "add", "upload", "delete", "get", "retrieve", "access", "list", "search", "find"]
        reasoning_verbs = ["analyze", "extract", "identify", "summarize"]
        
        if any(v in text_lower for v in tool_verbs):
            return "TOOL_USING"
        elif any(v in text_lower for v in reasoning_verbs) and any(key in text_lower for key in self.memory.keys()):
            return "REASONING"
        else:
            return "TOOL_USING"  # Default

    def _execute_step(self, step: Step, step_type: str, state: ReasonerState) -> Any:
        """Execute a step based on its type."""
        if step_type in ["SEARCH"]:
            # Tool selection only
            tool_id = self.tool_selector.select_tool(step, state)
            logger.info(f"Tool selected: {tool_id}")
            
            # Store tool_id in memory if requested  
            result = {"id": tool_id}
            if step.store_key:
                self.memory.set(
                    key=step.store_key,
                    value=result,
                    description=f"Tool ID for step '{step.text}'"
                )
            return result
            
        elif step_type in ["EXECUTE"]:
            # Tool execution - use same tool selection logic as SEARCH
            tool_id = self.tool_selector.select_tool(step, state)
            tool_info = self.jentic_client.load(tool_id)
            params = self.parameter_generator.generate_and_validate_parameters(tool_id, tool_info, state)
            return self.step_executor.execute_tool_step(tool_id, params, step)
            
        elif step_type in ["REASON", "REASONING"]:
            # Reasoning step
            return self.step_executor.execute_reasoning_step(step, state)
            
        else:
            # Default to full tool workflow (select + execute)
            tool_id = self.tool_selector.select_tool(step, state)
            tool_info = self.jentic_client.load(tool_id)
            params = self.parameter_generator.generate_and_validate_parameters(tool_id, tool_info, state)
            return self.step_executor.execute_tool_step(tool_id, params, step)

    def _process_step_result(self, step: Step, result: Any, state: ReasonerState) -> ReasonerState:
        """Process step execution result and update state."""
        logger.info("=== OBSERVATION PHASE ===")
        
        # Determine success
        success = self._is_result_successful(result)
        
        if success:
            # Ensure result is JSON-serializable before storing
            serializable_result = make_json_serializable(result)
            updated_step = step.mark_done(serializable_result)
            logger.info(f"Step completed successfully: {step.text}")
            
            # Store result in memory if requested
            if step.store_key:
                self.memory.set(
                    key=step.store_key,
                    value=serializable_result,
                    description=f"Result from step '{step.text}'"
                )
            
            history_entry = f"{step.text} -> done"
            return state.update_current_step(updated_step).with_history(history_entry).advance_plan()
        else:
            # Ensure result is JSON-serializable even for failed steps
            serializable_result = make_json_serializable(result)
            logger.warning(f"Step failed: {step.text}")
            
            # Extract simple error message for reflection
            error_msg = str(serializable_result.get("error", serializable_result)) if isinstance(serializable_result, dict) else str(serializable_result)
            
            # Try reflection on failed result
            logger.info("Attempting reflection on failed step result.")
            success, revised_step = self.reflection_engine.reflect_on_failure(step, error_msg, state)
            
            if success and revised_step:
                logger.info("Step revised after failure, retrying.")
                return state.update_current_step(revised_step)
            else:
                logger.warning("Reflection failed. Step cannot be recovered.")
                updated_step = step.mark_failed(serializable_result)
                history_entry = f"{step.text} -> failed"
                return state.update_current_step(updated_step).with_history(history_entry).mark_failed()

    def _handle_step_failure(self, step: Step, error_msg: str, state: ReasonerState) -> ReasonerState:
        """Handle step failure through reflection."""
        logger.info("Attempting reflection on failed step.")
        
        success, revised_step = self.reflection_engine.reflect_on_failure(step, error_msg, state)
        
        if success and revised_step:
            logger.info("Step revised after failure, retrying.")
            return state.update_current_step(revised_step).mark_failed()  # Remove failed flag
        else:
            logger.warning("Reflection failed. Step cannot be recovered.")
            failed_step = step.mark_failed()
            return state.update_current_step(failed_step).mark_failed()

    def _evaluate_goal_completion(self, state: ReasonerState) -> bool:
        """Evaluate if the goal has been completed based on current state."""
        try:
            # Simple heuristic: if plan is empty and we've made progress, goal is likely complete
            if not state.plan and hasattr(state, 'history') and state.history:
                return True
            
            # Use LLM to evaluate goal completion
            evaluation_prompt = load_prompt('goal_evaluation')
            
            # Build robust evaluation context
            completed_steps = []
            for step in state.plan:
                if hasattr(step.status, 'name') and step.status.name == "DONE":
                    completed_steps.append(step.text)
                elif step.status == StepStatus.DONE:
                    completed_steps.append(step.text)
            
            current_observations = []
            if hasattr(state, 'history') and state.history:
                current_observations = state.history[-5:]
            
            evaluation_context = {
                'goal': state.goal,
                'completed_steps': completed_steps,
                'current_observations': current_observations,
                'total_steps': len(state.plan),
                'history': current_observations  # Alternative field name
            }
            
            if isinstance(evaluation_prompt, dict):
                evaluation_prompt["inputs"].update(evaluation_context)
                prompt = json.dumps(evaluation_prompt, ensure_ascii=False)
            else:
                prompt = evaluation_prompt.format(**evaluation_context)
            
            response = self.safe_llm_call([{"role": "user", "content": prompt}])
            
            # Parse response for completion status
            return 'yes' in response.lower() or 'completed' in response.lower()
            
        except Exception as e:
            self.logger.error(f"Error evaluating goal completion: {e}")
            # Fallback: if plan is empty, assume goal is complete
            return not state.plan

    def _generate_plan_text(self, goal: str, context: Dict[str, Any]) -> str:
        """Generate plan text using LLM for compatibility methods."""
        try:
            # Use bullet plan prompt to generate plan
            bullet_plan_prompt = load_prompt('bullet_plan')
            plan_context = {
                'goal': goal,
                'context': context,
                'available_tools': context.get('available_tools', []),
                'memory_items': context.get('memory_items', [])
            }
            
            if isinstance(bullet_plan_prompt, dict):
                bullet_plan_prompt["inputs"].update(plan_context)
                prompt = json.dumps(bullet_plan_prompt, ensure_ascii=False)
            else:
                prompt = bullet_plan_prompt.format(**plan_context)
            
            response = self.safe_llm_call([{"role": "user", "content": prompt}])
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating plan text: {e}")
            # Fallback to simple plan
            return f"- Analyze the goal: {goal}\n- Execute necessary actions\n- Verify completion"

    def _is_result_successful(self, result: Any) -> bool:
        """Determine if a result indicates success."""
        if isinstance(result, dict):
            if "error" in result:
                return False
            inner = result.get("result", result)
            if hasattr(inner, "success"):
                return inner.success
            return bool(inner)
        return getattr(result, "success", True)

    # BaseReasoner compatibility methods

    def plan(self, goal: str, context: Dict[str, Any]) -> str:
        """BaseReasoner compatibility - generate plan description."""
        try:
            state = ReasonerState(goal=goal)
            # Use our BulletPlanParser to generate a proper plan
            plan_text = self._generate_plan_text(goal, context)
            parsed_plan = self.plan_parser.parse_plan(plan_text, goal)
            
            if parsed_plan.steps:
                plan_summary = f"Generated {len(parsed_plan.steps)} step plan:\n"
                for i, step in enumerate(parsed_plan.steps, 1):
                    plan_summary += f"{i}. {step.description}\n"
                return plan_summary.strip()
            else:
                return f"Generated plan for: {goal}"
        except Exception as e:
            self.logger.error(f"Error in plan compatibility method: {e}")
            return f"Generated plan for: {goal}"

    def select_tool(self, plan: str, available_tools: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """BaseReasoner compatibility - select tool from available list."""
        if not available_tools:
            return None
        
        try:
            # Use our ToolSelector to intelligently choose the best tool
            step_description = plan.split('\n')[0] if plan else "Execute plan step"
            selected_tool = self.tool_selector.select_tool(
                step_description=step_description,
                available_tools=available_tools,
                context={}
            )
            return selected_tool
        except Exception as e:
            self.logger.error(f"Error in select_tool compatibility method: {e}")
            # Fallback to first available tool
            return available_tools[0] if available_tools else None

    def act(self, tool: Dict[str, Any], plan: str) -> Dict[str, Any]:
        """BaseReasoner compatibility - generate parameters for tool execution."""
        if not tool:
            return {}
        
        try:
            # Use our ParameterGenerator to create proper parameters
            step_description = plan.split('\n')[0] if plan else "Execute with tool"
            parameters = self.parameter_generator.generate_parameters(
                tool=tool,
                step_description=step_description,
                context={},
                memory=self.memory
            )
            return parameters
        except Exception as e:
            self.logger.error(f"Error in act compatibility method: {e}")
            return {}

    def observe(self, action_result: Dict[str, Any]) -> str:
        """BaseReasoner compatibility - process and format observation."""
        try:
            # Process the result similar to how we do in step execution
            if isinstance(action_result, dict):
                if 'error' in action_result:
                    return f"Error observed: {action_result['error']}"
                elif 'result' in action_result:
                    result = action_result['result']
                    if isinstance(result, str):
                        return f"Result: {result[:500]}..." if len(result) > 500 else f"Result: {result}"
                    else:
                        return f"Result: {str(result)[:500]}..." if len(str(result)) > 500 else f"Result: {str(result)}"
                else:
                    return f"Observed: {str(action_result)[:500]}..." if len(str(action_result)) > 500 else f"Observed: {str(action_result)}"
            else:
                result_str = str(action_result)
                return f"Observed: {result_str[:500]}..." if len(result_str) > 500 else f"Observed: {result_str}"
        except Exception as e:
            self.logger.error(f"Error in observe compatibility method: {e}")
            return f"Observed result: {action_result}"

    def evaluate(self, goal: str, observations: List[str]) -> bool:
        """BaseReasoner compatibility - evaluate goal completion."""
        try:
            # Use our goal evaluation logic
            state = ReasonerState(goal=goal, history=observations)
            return self._evaluate_goal_completion(state)
        except Exception as e:
            self.logger.error(f"Error in evaluate compatibility method: {e}")
            # Conservative fallback - assume not completed if we can't evaluate
            return False

    def reflect(self, goal: str, observations: List[str], failed_attempts: List[str]) -> str:
        """BaseReasoner compatibility - generate reflection using ReflectionEngine."""
        try:
            # Create a failed step to trigger reflection
            failed_step = Step(
                description="Compatibility reflection step",
                step_type=StepType.REASONING,
                status=StepStatus.FAILED,
                error_message=f"Failed attempts: {failed_attempts}"
            )
            
            # Use our ReflectionEngine
            reflection_result = self.reflection_engine.reflect_on_failure(
                failed_step=failed_step,
                context={
                    'goal': goal,
                    'observations': observations,
                    'failed_attempts': failed_attempts
                },
                memory=self.memory
            )
            
            if reflection_result.revised_step:
                return f"Reflection: {reflection_result.revised_step.description}"
            elif reflection_result.analysis:
                return f"Analysis: {reflection_result.analysis}"
            else:
                return "Reflection completed - consider alternative approaches"
                
        except Exception as e:
            self.logger.error(f"Error in reflect compatibility method: {e}")
            return "Reflection completed" 