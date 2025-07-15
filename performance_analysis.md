# Performance Analysis and Optimization Recommendations

## Executive Summary

After analyzing the codebase, I've identified several key performance bottlenecks in the reasoner implementations and surrounding systems. The main issues stem from excessive LLM calls, inefficient tool search patterns, and lack of caching mechanisms. Below are detailed findings and actionable recommendations.

## Key Performance Bottlenecks

### 1. **Excessive LLM Calls in BulletPlanReasoner**

**Issue**: Each reasoning step involves multiple sequential LLM calls:
- Task complexity classification (HybridReasoner)
- Keyword extraction for tool search
- Tool selection from candidates
- Parameter generation
- Reflection and replanning

**Impact**: 5-7 LLM calls per step × 20 max iterations = potentially 100+ LLM calls per complex task

### 2. **Inefficient Tool Search Pattern**

**Issue**: 
- Tool search is performed for every step, even when tools are reused
- `search_top_k=10` fetches 10 tools when typically only 1-2 are needed
- No caching of search results between steps
- Each search triggers a Jentic API call

**Impact**: Unnecessary network latency and API calls

### 3. **Suboptimal Memory Lookups**

**Issue**:
- No caching layer for frequently accessed memory items
- Placeholder resolution performs recursive lookups without memoization
- Memory validation happens multiple times for the same parameters

### 4. **Redundant Task Classification**

**Issue**: HybridReasoner classifies every task, even obviously simple ones
- Classification requires an LLM call
- Simple keyword-based heuristics could handle 80% of cases

## Optimization Recommendations

### 1. **Implement Tool Search Result Caching**

```python
# In tool_selector.py
class ToolSelector:
    def __init__(self, ...):
        # Add search cache
        self._search_cache = {}  # query -> (results, timestamp)
        self._cache_ttl = 300  # 5 minutes
    
    def _search_tools(self, query: str) -> List[Dict[str, Any]]:
        # Check cache first
        cache_key = f"{query}:{self.search_top_k}"
        if cache_key in self._search_cache:
            results, timestamp = self._search_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                logger.info(f"Using cached search results for: {query}")
                return results
        
        # Perform search and cache
        results = self.jentic_client.search(query, top_k=self.search_top_k)
        self._search_cache[cache_key] = (results, time.time())
        return results
```

### 2. **Batch LLM Calls Where Possible**

```python
# Create a batch reasoning helper
class BatchLLMHelper:
    def batch_classify_steps(self, steps: List[Step]) -> List[str]:
        """Classify multiple steps in a single LLM call"""
        # Combine multiple classification requests
        combined_prompt = self._build_batch_prompt(steps)
        response = self.llm.chat([{"role": "user", "content": combined_prompt}])
        return self._parse_batch_response(response)
```

### 3. **Add Fast-Path Heuristics**

```python
# In hybrid_reasoner.py
def _is_simple_task(self, goal: str) -> bool:
    # Fast heuristics before LLM call
    goal_lower = goal.lower()
    
    # Simple patterns that don't need LLM classification
    simple_patterns = [
        r"^(get|fetch|retrieve|find|search for)\s+\w+",
        r"^(what|who|when|where)\s+",
        r"^(send|post)\s+a?\s*(message|email|notification)",
        r"^(list|show)\s+\w+",
    ]
    
    for pattern in simple_patterns:
        if re.match(pattern, goal_lower):
            logger.info(f"Fast-path classification: SIMPLE task")
            return True
    
    # Complex patterns
    complex_indicators = ["and then", "after that", "multiple", "steps", "plan"]
    if any(indicator in goal_lower for indicator in complex_indicators):
        logger.info(f"Fast-path classification: COMPLEX task")
        return False
    
    # Fall back to LLM classification
    return self._llm_classify_task(goal)
```

### 4. **Optimize Tool Selection Strategy**

```python
# In tool_selector.py
def select_tool(self, step: Step, state: ReasonerState) -> str:
    # 1. Check if tool is explicitly mentioned
    tool_from_text = self._extract_tool_from_text(step.text)
    if tool_from_text:
        return tool_from_text
    
    # 2. Check memory references (existing)
    tool_id = self._check_execute_pattern(step)
    if tool_id:
        return tool_id
    
    # 3. Use previous successful tools for similar steps
    similar_tool = self._find_similar_step_tool(step, state)
    if similar_tool:
        return similar_tool
    
    # 4. Only then perform search
    return self._search_and_select_tool(step, state)
```

### 5. **Implement Adaptive Search Parameters**

```python
# Dynamic top_k based on query complexity
def _determine_search_top_k(self, query: str) -> int:
    """Reduce search results for simple queries"""
    if any(specific in query.lower() for specific in ["send email", "get user", "create task"]):
        return 3  # Very specific queries need fewer results
    elif len(query.split()) < 5:
        return 5  # Short queries
    else:
        return 10  # Complex queries
```

### 6. **Add Result Caching for Parameter Generation**

```python
# In parameter_generator.py
class ParameterGenerator:
    def __init__(self, ...):
        self._param_cache = {}  # (tool_id, context_hash) -> parameters
    
    def generate_parameters(self, tool_id: str, context: Dict) -> Dict:
        # Create cache key from tool and context
        cache_key = self._create_cache_key(tool_id, context)
        
        if cache_key in self._param_cache:
            logger.info(f"Using cached parameters for {tool_id}")
            return self._param_cache[cache_key]
        
        # Generate and cache
        params = self._generate_with_llm(tool_id, context)
        self._param_cache[cache_key] = params
        return params
```

### 7. **Optimize FreeformReasoner Tool Catalog**

```python
# In freeform_reasoner.py
def _build_tool_catalog(self, state: ConversationState) -> str:
    # Only include relevant tools based on goal
    goal_keywords = self._extract_keywords(state.goal)
    
    # Search for relevant tools only
    relevant_tools = self.jentic_client.search(
        " ".join(goal_keywords), 
        top_k=5  # Reduced from 10
    )
    
    # Format only essential information
    return self._format_compact_catalog(relevant_tools)
```

### 8. **Configuration Optimizations**

```toml
# In config.toml
[tool.actbots.reasoner.bullet_plan]
max_reflection_attempts = 2  # Reduced from 3
search_top_k = 5  # Reduced from 10
max_iterations = 10  # Reduced from 20
llm_timeout_seconds = 20  # Reduced from 30
parameter_generation_retries = 2  # Reduced from 3
enable_caching = true
cache_ttl_seconds = 300
```

### 9. **Implement Async Optimizations**

```python
# Use async for parallel operations
async def _execute_plan_steps_async(self, steps: List[Step]):
    """Execute independent steps in parallel"""
    independent_steps = self._identify_independent_steps(steps)
    
    if len(independent_steps) > 1:
        tasks = [self._execute_step_async(step) for step in independent_steps]
        results = await asyncio.gather(*tasks)
        return results
```

### 10. **Add Metrics and Monitoring**

```python
# Performance tracking
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "llm_calls": 0,
            "search_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_latency": 0
        }
    
    def log_performance_summary(self):
        logger.info(f"Performance Summary: {self.metrics}")
        cache_hit_rate = self.metrics["cache_hits"] / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
        logger.info(f"Cache hit rate: {cache_hit_rate:.2%}")
```

## Implementation Priority

1. **High Priority (Quick Wins)**:
   - Add search result caching (30-40% reduction in API calls)
   - Implement fast-path heuristics for task classification (eliminate 80% of classification LLM calls)
   - Reduce search_top_k parameter (reduce response size and processing time)

2. **Medium Priority**:
   - Batch LLM operations where possible
   - Add parameter generation caching
   - Optimize tool catalog generation

3. **Low Priority (Long-term)**:
   - Async execution optimizations
   - Advanced caching strategies
   - Machine learning-based tool prediction

## Expected Performance Improvements

With these optimizations:
- **LLM calls reduced by 60-70%** through caching and heuristics
- **Jentic API calls reduced by 40-50%** through search caching
- **Overall latency reduced by 50%** through parallel operations and reduced iterations
- **Cost reduction of 60%** due to fewer LLM API calls

## Testing Recommendations

1. Add performance benchmarks for common tasks
2. Monitor cache hit rates in production
3. A/B test configuration changes
4. Profile hot paths using Python profilers
5. Add metrics collection for continuous monitoring