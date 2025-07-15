# Prompt Optimization Strategy

## Frequently Used Prompts Analysis

Based on the codebase analysis, the most frequently called prompts are:

1. **keyword_extraction.txt** - Called for every tool search
2. **select_tool.txt** - Called after every tool search
3. **param_generation.txt** - Called for every tool execution
4. **hybrid_classifier.txt** - Called for every task in HybridReasoner
5. **reflection_prompt.txt** - Called on failures

## Prompt-Specific Optimizations

### 1. **Combine Keyword Extraction and Tool Selection**

Instead of two separate LLM calls:
```python
# Current flow:
# 1. LLM call for keyword extraction
# 2. Search tools
# 3. LLM call for tool selection

# Optimized flow:
class DirectToolSelector:
    def select_tool_direct(self, step, state):
        """Single LLM call that suggests tools directly"""
        # Preload top 20-30 most common tools in memory
        common_tools = self._get_common_tools_catalog()
        
        prompt = f"""
        Step: {step.text}
        Goal: {state.goal}
        
        Common tools available:
        {common_tools}
        
        If the needed tool is in the list above, return its ID.
        If not, return SEARCH: <keywords to search for>
        """
        
        response = self.llm.chat([{"role": "user", "content": prompt}])
        
        if response.startswith("SEARCH:"):
            keywords = response[7:].strip()
            return self._search_and_select(keywords)
        else:
            return response.strip()
```

### 2. **Batch Parameter Generation**

For workflows with multiple similar tools:
```python
class BatchParameterGenerator:
    def generate_parameters_batch(self, tool_steps: List[Tuple[str, Dict]]):
        """Generate parameters for multiple tools in one LLM call"""
        batch_prompt = self._build_batch_param_prompt(tool_steps)
        response = self.llm.chat([{"role": "user", "content": batch_prompt}])
        return self._parse_batch_params(response)
```

### 3. **Template-Based Fast Paths**

For common parameter patterns:
```python
PARAM_TEMPLATES = {
    "email": {
        "pattern": r"send.*email.*to\s+(\S+@\S+)",
        "template": {
            "to": "${1}",
            "subject": "${memory.subject|'No Subject'}",
            "body": "${memory.body|step.text}"
        }
    },
    "user_lookup": {
        "pattern": r"(get|find|fetch).*user.*(?:named|called|with id)\s+(.+)",
        "template": {
            "user_id": "${2}",
            "fields": ["id", "name", "email"]
        }
    }
}

def try_template_params(self, tool_id: str, step_text: str) -> Optional[Dict]:
    """Try to generate parameters using templates"""
    for template_name, config in PARAM_TEMPLATES.items():
        if re.match(config["pattern"], step_text, re.IGNORECASE):
            return self._apply_template(config["template"], step_text)
    return None
```

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
1. **Implement search result caching** in tool_selector.py
2. **Add fast-path heuristics** to hybrid_reasoner.py
3. **Reduce configuration parameters** in config.toml
4. **Add simple performance logging**

### Phase 2: Core Optimizations (3-5 days)
1. **Implement OptimizedToolSelector** with pattern caching
2. **Add parameter generation caching**
3. **Implement common tools catalog**
4. **Create batch processing utilities**

### Phase 3: Advanced Features (1 week)
1. **Implement direct tool selection** (combined prompt)
2. **Add template-based parameter generation**
3. **Create async execution framework**
4. **Build comprehensive monitoring**

## Estimated Performance Impact

### Immediate Gains (Phase 1):
- **30-40%** reduction in Jentic API calls
- **20-30%** reduction in LLM calls
- **25%** reduction in overall latency

### Full Implementation:
- **60-70%** reduction in LLM calls
- **50%** reduction in Jentic API calls
- **50-60%** reduction in overall latency
- **60%** cost reduction

## Integration Example

```python
# In main.py, modify the agent_setup function:
def agent_setup(controller, jentic_client, memory, llm_wrapper):
    # Import optimizations
    from performance_optimizations_sample import (
        OptimizedToolSelector,
        OptimizedHybridReasoner,
        OptimizedParameterGenerator,
        PerformanceMonitor
    )
    
    # Initialize performance monitor
    perf_monitor = PerformanceMonitor()
    
    # Create optimized reasoner
    if ENABLE_OPTIMIZATIONS:
        reasoner = OptimizedHybridReasoner(
            jentic=jentic_client,
            memory=memory,
            llm=llm_wrapper,
            intervention_hub=controller.intervention_hub if controller else None,
        )
        
        # Inject optimized components
        if hasattr(reasoner.bullet, 'tool_selector'):
            reasoner.bullet.tool_selector = OptimizedToolSelector(
                jentic_client, memory, llm_wrapper, search_top_k=5
            )
        
        if hasattr(reasoner.bullet, 'parameter_generator'):
            reasoner.bullet.parameter_generator = OptimizedParameterGenerator(
                memory, llm_wrapper, max_retries=2
            )
    else:
        # Original implementation
        reasoner = HybridReasoner(
            jentic=jentic_client,
            memory=memory,
            llm=llm_wrapper,
            intervention_hub=controller.intervention_hub if controller else None,
        )
    
    agent = InteractiveCLIAgent(
        reasoner=reasoner,
        memory=memory,
        controller=controller,
        jentic_client=jentic_client,
    )
    
    # Add performance monitoring hooks
    import atexit
    atexit.register(perf_monitor.log_summary)
    
    return agent
```

## Testing Strategy

1. **Benchmark Suite**: Create standard test scenarios
2. **A/B Testing**: Run optimized vs original with same inputs
3. **Metrics Collection**: Track all performance indicators
4. **Regression Testing**: Ensure accuracy is maintained

## Monitoring and Metrics

```python
# Add to config.toml
[tool.actbots.monitoring]
enabled = true
metrics_backend = "prometheus"  # or "cloudwatch", "datadog"
export_interval = 60  # seconds

[tool.actbots.monitoring.alerts]
high_latency_threshold = 10.0  # seconds
high_cost_threshold = 1.0  # dollars per operation
low_cache_hit_threshold = 0.3  # 30%
```

## Conclusion

The proposed optimizations target the most impactful areas of the codebase:
1. Reducing redundant LLM calls through caching and smart heuristics
2. Minimizing Jentic API calls through result caching
3. Optimizing prompt strategies to combine multiple operations
4. Adding comprehensive monitoring for continuous improvement

Start with Phase 1 for immediate gains, then progressively implement more advanced optimizations based on observed bottlenecks and usage patterns.