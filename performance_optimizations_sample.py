"""
Sample implementations of high-priority performance optimizations for the Jentic Agents system.
These can be integrated into the existing codebase to achieve immediate performance improvements.
"""

import re
import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache
from dataclasses import dataclass
from threading import Lock

# ============================================================================
# 1. OPTIMIZED TOOL SELECTOR WITH CACHING
# ============================================================================

class OptimizedToolSelector:
    """Enhanced ToolSelector with search result caching and smart selection strategies."""
    
    def __init__(self, jentic_client, memory, llm, search_top_k: int = 10):
        self.jentic_client = jentic_client
        self.memory = memory
        self.llm = llm
        self.search_top_k = search_top_k
        
        # Add caching mechanisms
        self._search_cache: Dict[str, Tuple[List[Dict], float]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._tool_pattern_cache: Dict[str, str] = {}  # step_pattern -> tool_id
        self._cache_lock = Lock()
        
    def select_tool(self, step, state) -> str:
        """Select tool with multiple optimization strategies."""
        
        # 1. Fast path: Check if tool is explicitly mentioned in step text
        tool_from_text = self._extract_tool_from_text(step.text)
        if tool_from_text:
            logger.info(f"Fast path: Tool explicitly mentioned: {tool_from_text}")
            return tool_from_text
        
        # 2. Check memory references (existing functionality)
        tool_id = self._check_execute_pattern(step)
        if tool_id:
            return tool_id
        
        # 3. Check pattern cache for similar steps
        similar_tool = self._find_cached_pattern_tool(step)
        if similar_tool:
            logger.info(f"Using cached tool for similar pattern: {similar_tool}")
            return similar_tool
        
        # 4. Smart search with adaptive parameters
        search_query = self._build_search_query(step, state)
        top_k = self._determine_adaptive_top_k(search_query, step.text)
        search_hits = self._search_tools_with_cache(search_query, top_k)
        
        if not search_hits:
            raise RuntimeError(f"No tools found for query: '{search_query}'")
        
        # 5. Prioritize and select
        prioritized_hits = self._prioritize_by_provider(search_hits, step)
        selected_tool_id = self._select_with_llm(step, prioritized_hits, state)
        
        # Cache the pattern for future use
        self._cache_step_pattern(step, selected_tool_id)
        
        return selected_tool_id
    
    def _extract_tool_from_text(self, step_text: str) -> Optional[str]:
        """Extract tool ID if explicitly mentioned in step text."""
        # Common patterns where tools are mentioned
        patterns = [
            r"use\s+tool\s+['\"]?([a-zA-Z0-9\-_\.]+)['\"]?",
            r"call\s+['\"]?([a-zA-Z0-9\-_\.]+)['\"]?\s+(?:tool|api|operation)",
            r"execute\s+['\"]?([a-zA-Z0-9\-_\.]+)['\"]?\s+(?:tool|api|operation)",
        ]
        
        step_lower = step_text.lower()
        for pattern in patterns:
            match = re.search(pattern, step_lower)
            if match:
                potential_tool = match.group(1)
                # Verify it looks like a tool ID
                if '.' in potential_tool and len(potential_tool) > 5:
                    return potential_tool
        return None
    
    def _search_tools_with_cache(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search for tools with caching."""
        cache_key = f"{query}:{top_k}"
        
        with self._cache_lock:
            # Check cache
            if cache_key in self._search_cache:
                results, timestamp = self._search_cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    logger.info(f"Cache hit for search query: {query}")
                    return results
            
            # Perform search
            logger.info(f"Cache miss, searching: {query} (top_k={top_k})")
            results = self.jentic_client.search(query, top_k=top_k)
            
            # Cache results
            self._search_cache[cache_key] = (results, time.time())
            
            # Cleanup old cache entries
            self._cleanup_cache()
            
        return results
    
    def _determine_adaptive_top_k(self, query: str, step_text: str) -> int:
        """Dynamically determine search result count based on query specificity."""
        query_lower = query.lower()
        step_lower = step_text.lower()
        
        # Very specific operations need fewer results
        specific_operations = [
            "send email", "create user", "delete record", 
            "update profile", "fetch data", "post message"
        ]
        
        for op in specific_operations:
            if op in query_lower or op in step_lower:
                return min(3, self.search_top_k)
        
        # Short, specific queries
        if len(query.split()) <= 3:
            return min(5, self.search_top_k)
        
        # Default to configured value
        return self.search_top_k
    
    def _find_cached_pattern_tool(self, step) -> Optional[str]:
        """Find tool from cached patterns of similar steps."""
        step_pattern = self._generate_step_pattern(step.text)
        return self._tool_pattern_cache.get(step_pattern)
    
    def _generate_step_pattern(self, step_text: str) -> str:
        """Generate a normalized pattern from step text for caching."""
        # Remove specific values but keep structure
        pattern = step_text.lower()
        # Replace emails, IDs, numbers with placeholders
        pattern = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Z|a-z]{2,}\b', '<email>', pattern)
        pattern = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '<uuid>', pattern)
        pattern = re.sub(r'\b\d+\b', '<number>', pattern)
        pattern = re.sub(r'"[^"]*"', '<string>', pattern)
        pattern = re.sub(r"'[^']*'", '<string>', pattern)
        return pattern
    
    def _cache_step_pattern(self, step, tool_id: str):
        """Cache the tool selection for a step pattern."""
        pattern = self._generate_step_pattern(step.text)
        self._tool_pattern_cache[pattern] = tool_id
        
        # Limit cache size
        if len(self._tool_pattern_cache) > 100:
            # Remove oldest entries (simple FIFO for now)
            oldest_key = next(iter(self._tool_pattern_cache))
            del self._tool_pattern_cache[oldest_key]
    
    def _cleanup_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._search_cache.items()
            if current_time - timestamp > self._cache_ttl
        ]
        for key in expired_keys:
            del self._search_cache[key]


# ============================================================================
# 2. OPTIMIZED HYBRID REASONER WITH FAST-PATH HEURISTICS
# ============================================================================

class OptimizedHybridReasoner:
    """HybridReasoner with fast-path classification heuristics."""
    
    def __init__(self, jentic, memory, llm=None, model=None, intervention_hub=None, **kwargs):
        # Initialize parent components
        self.jentic = jentic
        self.memory = memory
        self.llm = llm
        self.intervention_hub = intervention_hub
        
        # Initialize sub-reasoners
        self.freeform = FreeformReasoner(jentic, memory, llm, intervention_hub, **kwargs)
        self.bullet = BulletPlanReasoner(jentic, memory, llm, intervention_hub, **kwargs)
        
        # Cache for classification results
        self._classification_cache = {}
        
    def _is_simple_task(self, goal: str) -> bool:
        """Classify task complexity with fast heuristics before LLM call."""
        goal_lower = goal.lower()
        
        # Check cache first
        if goal_lower in self._classification_cache:
            logger.info(f"Using cached classification for goal")
            return self._classification_cache[goal_lower]
        
        # Fast path: Simple single-action patterns
        simple_patterns = [
            r"^(get|fetch|retrieve|find|search for|look up)\s+\w+",
            r"^(what|who|when|where|which|how many)\s+",
            r"^(send|post)\s+(?:a\s+)?(message|email|notification|alert)",
            r"^(list|show|display)\s+(?:all\s+)?\w+",
            r"^(check|verify|validate)\s+\w+",
            r"^(create|make|add)\s+(?:a\s+)?(?:new\s+)?\w+\s+(?:with|named|called)",
            r"^(delete|remove)\s+\w+",
            r"^(update|modify|change|edit)\s+\w+\s+\w+",
            r"^(tell me|show me|give me)\s+",
        ]
        
        for pattern in simple_patterns:
            if re.match(pattern, goal_lower):
                logger.info(f"Fast-path classification: SIMPLE task (pattern match)")
                self._classification_cache[goal_lower] = True
                return True
        
        # Fast path: Complex multi-step indicators
        complex_indicators = [
            # Explicit multi-step words
            "and then", "after that", "afterwards", "next",
            "followed by", "subsequently", "finally",
            # Planning words
            "plan", "steps", "procedure", "workflow",
            # Multiple actions
            "and also", "as well as", "in addition",
            # Conditional logic
            "if", "when", "unless", "depending on",
            # Iteration
            "for each", "for all", "for every",
            # Complex goals
            "analyze", "investigate", "research", "compare"
        ]
        
        if any(indicator in goal_lower for indicator in complex_indicators):
            logger.info(f"Fast-path classification: COMPLEX task (indicator found)")
            self._classification_cache[goal_lower] = False
            return False
        
        # Count action verbs - multiple verbs usually indicate complexity
        action_verbs = re.findall(
            r'\b(get|fetch|send|create|update|delete|find|search|list|check|verify|analyze|post|make)\b',
            goal_lower
        )
        if len(set(action_verbs)) > 2:  # More than 2 different action verbs
            logger.info(f"Fast-path classification: COMPLEX task (multiple actions)")
            self._classification_cache[goal_lower] = False
            return False
        
        # Fall back to LLM classification only for ambiguous cases
        logger.info("Ambiguous task, falling back to LLM classification")
        is_simple = self._llm_classify_task(goal)
        self._classification_cache[goal_lower] = is_simple
        return is_simple
    
    def _llm_classify_task(self, goal: str) -> bool:
        """Original LLM-based classification as fallback."""
        prompt_template = load_prompt("hybrid_classifier")
        prompt = prompt_template.format(goal=goal)
        
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}]).strip().upper()
            logger.info(f"LLM classification response: {response}")
            return "SINGLE-STEP" in response
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}. Defaulting to complex.")
            return False


# ============================================================================
# 3. OPTIMIZED PARAMETER GENERATOR WITH CACHING
# ============================================================================

@dataclass
class ParamCacheEntry:
    """Cache entry for parameter generation."""
    params: Dict[str, Any]
    timestamp: float
    context_hash: str


class OptimizedParameterGenerator:
    """Parameter generator with intelligent caching."""
    
    def __init__(self, memory, llm, max_retries: int = 3):
        self.memory = memory
        self.llm = llm
        self.max_retries = max_retries
        
        # Parameter cache: (tool_id, context_hash) -> ParamCacheEntry
        self._param_cache: Dict[Tuple[str, str], ParamCacheEntry] = {}
        self._cache_ttl = 600  # 10 minutes
        self._max_cache_size = 100
        
    def generate_and_validate_parameters(
        self, tool_id: str, tool_info: Dict[str, Any], state
    ) -> Dict[str, Any]:
        """Generate parameters with caching for similar contexts."""
        
        # Create cache key
        context_hash = self._create_context_hash(tool_id, tool_info, state)
        cache_key = (tool_id, context_hash)
        
        # Check cache
        cached_params = self._get_cached_params(cache_key)
        if cached_params is not None:
            logger.info(f"Using cached parameters for {tool_id}")
            return cached_params
        
        # Generate new parameters
        params = self._generate_new_parameters(tool_id, tool_info, state)
        
        # Cache the result
        self._cache_params(cache_key, params, context_hash)
        
        return params
    
    def _create_context_hash(self, tool_id: str, tool_info: Dict[str, Any], state) -> str:
        """Create a hash of the context for caching."""
        # Include relevant context that affects parameter generation
        context_data = {
            "goal": state.goal,
            "required_fields": tool_info.get("required", []),
            "memory_keys": sorted(self.memory.keys()),
            "current_step": state.current_step.text if state.current_step else "",
        }
        
        # Create stable hash
        context_str = json.dumps(context_data, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _get_cached_params(self, cache_key: Tuple[str, str]) -> Optional[Dict[str, Any]]:
        """Retrieve cached parameters if valid."""
        if cache_key not in self._param_cache:
            return None
        
        entry = self._param_cache[cache_key]
        if time.time() - entry.timestamp > self._cache_ttl:
            # Expired
            del self._param_cache[cache_key]
            return None
        
        # Validate that memory references still exist
        try:
            validated_params = self.memory.resolve_placeholders(entry.params)
            return validated_params
        except Exception:
            # Memory state has changed, invalidate cache
            del self._param_cache[cache_key]
            return None
    
    def _cache_params(self, cache_key: Tuple[str, str], params: Dict[str, Any], context_hash: str):
        """Cache generated parameters."""
        # Limit cache size
        if len(self._param_cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = min(self._param_cache.keys(), 
                           key=lambda k: self._param_cache[k].timestamp)
            del self._param_cache[oldest_key]
        
        self._param_cache[cache_key] = ParamCacheEntry(
            params=params,
            timestamp=time.time(),
            context_hash=context_hash
        )
    
    def _generate_new_parameters(self, tool_id: str, tool_info: Dict[str, Any], state) -> Dict[str, Any]:
        """Original parameter generation logic."""
        # ... existing implementation ...
        pass


# ============================================================================
# 4. PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """Track and report performance metrics."""
    
    def __init__(self):
        self.metrics = {
            "llm_calls": 0,
            "search_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "tool_executions": 0,
            "total_latency": 0,
            "classification_fast_path": 0,
            "classification_llm": 0,
        }
        self._start_times = {}
        
    def start_operation(self, operation_id: str):
        """Start timing an operation."""
        self._start_times[operation_id] = time.time()
    
    def end_operation(self, operation_id: str):
        """End timing an operation and record latency."""
        if operation_id in self._start_times:
            latency = time.time() - self._start_times[operation_id]
            self.metrics["total_latency"] += latency
            del self._start_times[operation_id]
            return latency
        return 0
    
    def increment(self, metric: str, count: int = 1):
        """Increment a counter metric."""
        if metric in self.metrics:
            self.metrics[metric] += count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary with calculated rates."""
        total_classifications = (
            self.metrics["classification_fast_path"] + 
            self.metrics["classification_llm"]
        )
        
        cache_total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        
        summary = {
            **self.metrics,
            "cache_hit_rate": (
                self.metrics["cache_hits"] / cache_total 
                if cache_total > 0 else 0
            ),
            "fast_path_rate": (
                self.metrics["classification_fast_path"] / total_classifications
                if total_classifications > 0 else 0
            ),
            "avg_latency_per_operation": (
                self.metrics["total_latency"] / self.metrics["tool_executions"]
                if self.metrics["tool_executions"] > 0 else 0
            ),
        }
        
        return summary
    
    def log_summary(self):
        """Log performance summary."""
        summary = self.get_summary()
        logger.info("=== Performance Summary ===")
        logger.info(f"LLM Calls: {summary['llm_calls']}")
        logger.info(f"Search Calls: {summary['search_calls']}")
        logger.info(f"Cache Hit Rate: {summary['cache_hit_rate']:.2%}")
        logger.info(f"Fast Path Classification Rate: {summary['fast_path_rate']:.2%}")
        logger.info(f"Average Operation Latency: {summary['avg_latency_per_operation']:.3f}s")
        logger.info(f"Total Latency: {summary['total_latency']:.3f}s")


# ============================================================================
# 5. CONFIGURATION UPDATES
# ============================================================================

OPTIMIZED_CONFIG = {
    "tool.actbots.reasoner.bullet_plan": {
        "max_reflection_attempts": 2,  # Reduced from 3
        "search_top_k": 5,  # Reduced from 10
        "max_iterations": 10,  # Reduced from 20
        "llm_timeout_seconds": 20,  # Reduced from 30
        "parameter_generation_retries": 2,  # Reduced from 3
        "enable_caching": True,
        "cache_ttl_seconds": 300,
        "enable_performance_monitoring": True,
        "enable_fast_path_classification": True,
        "adaptive_search_enabled": True,
    },
    "tool.actbots.performance": {
        "log_performance_summary": True,
        "performance_log_interval": 100,  # Log every 100 operations
        "enable_metrics_export": True,
    }
}


# ============================================================================
# Usage Example
# ============================================================================

def integrate_optimizations(existing_reasoner):
    """Example of how to integrate optimizations into existing code."""
    
    # Initialize performance monitor
    perf_monitor = PerformanceMonitor()
    
    # Wrap existing components with optimized versions
    if hasattr(existing_reasoner, 'tool_selector'):
        existing_reasoner.tool_selector = OptimizedToolSelector(
            existing_reasoner.jentic_client,
            existing_reasoner.memory,
            existing_reasoner.llm,
            search_top_k=5  # Use optimized value
        )
    
    if hasattr(existing_reasoner, 'parameter_generator'):
        existing_reasoner.parameter_generator = OptimizedParameterGenerator(
            existing_reasoner.memory,
            existing_reasoner.llm,
            max_retries=2  # Use optimized value
        )
    
    # Log performance periodically
    import atexit
    atexit.register(perf_monitor.log_summary)
    
    return existing_reasoner, perf_monitor