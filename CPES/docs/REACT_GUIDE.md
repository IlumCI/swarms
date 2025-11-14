# ReAct (Reasoning and Acting) Guide for CPES

## Overview

ReAct is a powerful pattern that enables language models to perform structured reasoning by interleaving **Thought**, **Action**, and **Observation** steps. In CPES, ReAct is integrated into the cognitive loop to provide intelligent, tool-augmented reasoning capabilities.

## How ReAct Works

### The Core Pattern

ReAct follows a simple but powerful pattern:

1. **Thought**: The agent reasons about the problem
2. **Action**: The agent uses a tool or takes an action
3. **Observation**: The agent observes the result
4. **Repeat**: Continue until the problem is solved

### Example ReAct Session

```
Question: What is the square root of 144 plus 25?

Thought: I need to calculate the square root of 144 first, then add 25.
Action: CALC[144 ** 0.5]
Observation: 12.0

Thought: Now I need to add 25 to 12.
Action: CALC[12 + 25]
Observation: 37

Thought: I have the final answer.
Action: Finish[The answer is 37]
```

## ReAct in CPES

### Integration with Cognitive Loop

CPES automatically determines when to use ReAct based on input analysis:

- **Mathematical expressions**: `"What is 15 * 23 + 45?"`
- **Complex questions**: `"Compare the population of Vilnius with Riga"`
- **Search requests**: `"Find information about portal technology"`
- **Analysis tasks**: `"Analyze the pros and cons of this approach"`

### Available Tools

#### Basic Tools
- `CALC[expression]` - Mathematical calculations
- `COMPARE[a, b]` - Compare two values
- `FORMAT_TEXT[text, type]` - Format text
- `EXTRACT_NUMBERS[text]` - Extract numbers from text
- `COUNT_WORDS[text]` - Count words
- `SPLIT_TEXT[text, delimiter]` - Split text

#### Search Tools
- `SEARCH[query]` - Search knowledge base
- `LOOKUP[key]` - Look up specific key
- `LIST_KEYS[pattern]` - List available keys
- `ADD_KNOWLEDGE[key, value]` - Add knowledge

#### Memory Tools
- `RECALL_EPISODIC[query, k]` - Recall episodic memories
- `RECALL_SEMANTIC[entity]` - Recall semantic knowledge
- `RECALL_PROCEDURAL[task]` - Recall procedures
- `STORE_EPISODIC[text, tags]` - Store episodic memory
- `STORE_SEMANTIC[subject, predicate, object]` - Store semantic knowledge

## Using ReAct in CPES

### Basic Usage

```python
from CPES import Persona, ReActAgent, BasicTools, SearchTools
from CPES.utils import LLMWrapper

# Load persona
persona = Persona("path/to/persona.yaml")

# Initialize LLM
llm_wrapper = LLMWrapper("gpt-4o-mini", "openai")

# Create ReAct agent
react_agent = ReActAgent(
    llm_wrapper=llm_wrapper,
    tools={
        **BasicTools.get_tools(),
        **SearchTools().get_tools()
    },
    max_steps=6,
    persona_context=persona.get_identity_context()
)

# Use ReAct reasoning
result = react_agent.reason("What is 15 * 23 + 45?")
print(result.final_answer)
```

### Advanced Usage with Memory Integration

```python
from CPES import Persona, ReActAgent, MemoryTools
from CPES.memory import SemanticKnowledgeGraph, EpisodicMemoryStore

# Initialize memory systems
semantic_kg = SemanticKnowledgeGraph()
episodic_memory = EpisodicMemoryStore()

# Create memory tools
memory_tools = MemoryTools(
    episodic_memory=episodic_memory,
    semantic_kg=semantic_kg
).get_tools()

# Create ReAct agent with memory
react_agent = ReActAgent(
    llm_wrapper=llm_wrapper,
    tools={**basic_tools, **memory_tools},
    persona_context=persona.get_identity_context()
)

# Use ReAct with memory
result = react_agent.reason("What do I know about portal technology?")
```

## ReAct Configuration

### Customizing the Agent

```python
# Custom max steps
react_agent = ReActAgent(
    llm_wrapper=llm_wrapper,
    tools=tools,
    max_steps=10,  # Allow more reasoning steps
    persona_context=persona_context
)

# Add custom tools
def custom_tool(args):
    return f"Custom tool result: {args}"

react_agent.add_tool("custom", custom_tool, "A custom tool")
```

### Persona-Aware Reasoning

The ReAct agent uses the persona's context to maintain consistent reasoning style:

```yaml
# persona.yaml
name: Caroline
style:
  syntax: "crisply technical with sardonic asides"
  cadence: "short, declarative sentences"
  tics: ["Let's proceed.", "As expected."]
```

This influences the ReAct reasoning:

```
Thought: As expected, this requires a systematic approach.
Action: CALC[144 ** 0.5]
Observation: 12.0
Thought: Let's proceed with the addition.
Action: CALC[12 + 25]
Observation: 37
Action: Finish[The result is 37, as expected.]
```

## ReAct Patterns

### Mathematical Reasoning

```
Question: What is the area of a circle with radius 5?

Thought: I need to calculate the area using the formula π * r²
Action: CALC[3.14159 * 5 * 5]
Observation: 78.53975

Thought: I have the area calculation
Action: Finish[The area is approximately 78.54 square units]
```

### Information Gathering

```
Question: Find information about quantum computing

Thought: I should search for information about quantum computing
Action: SEARCH[quantum computing basics]
Observation: Found information about qubits, superposition, and entanglement

Thought: I have good information about quantum computing
Action: Finish[Quantum computing uses qubits that can exist in superposition states...]
```

### Comparative Analysis

```
Question: Compare Python and JavaScript for web development

Thought: I need to search for information about both languages
Action: SEARCH[Python web development pros cons]
Observation: Python is good for backend, has Django/Flask

Thought: Now I need information about JavaScript
Action: SEARCH[JavaScript web development pros cons]
Observation: JavaScript is essential for frontend, has many frameworks

Thought: I can now compare them
Action: Finish[Python excels at backend development with frameworks like Django, while JavaScript is essential for frontend development...]
```

## Error Handling and Fallbacks

### ReAct Failure Handling

```python
# ReAct automatically falls back to standard reasoning
if not result.success:
    print("ReAct failed, using standard reasoning")
    # The cognitive loop will use standard deliberation
```

### Tool Error Handling

```python
# Tools return error messages on failure
result = react_agent.reason("Calculate 1/0")
# Result: "Error: division by zero"
```

## Best Practices

### 1. Tool Design

- **Clear descriptions**: Tools should have clear docstrings
- **Error handling**: Always handle errors gracefully
- **Consistent interfaces**: Use consistent parameter patterns

### 2. Prompt Engineering

- **Clear instructions**: Be specific about the ReAct format
- **Persona context**: Include persona style in prompts
- **Tool descriptions**: Provide clear tool descriptions

### 3. Step Limits

- **Reasonable limits**: Set max_steps based on complexity
- **Timeout handling**: Implement timeouts for long operations
- **Resource management**: Monitor tool usage

### 4. Memory Integration

- **Selective storage**: Only store salient ReAct sessions
- **Context preservation**: Maintain context across steps
- **Knowledge updates**: Update knowledge bases with new information

## Debugging ReAct

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# ReAct will log detailed reasoning steps
result = react_agent.reason("What is 2+2?")
```

### Inspect Reasoning Process

```python
result = react_agent.reason("What is 2+2?")

# Get detailed summary
summary = react_agent.get_reasoning_summary(result)
print(summary)

# Inspect individual steps
for step in result.steps:
    print(f"Step {step.step_number}:")
    print(f"  Thought: {step.thought}")
    print(f"  Action: {step.action}")
    print(f"  Observation: {step.observation}")
```

## Advanced Features

### Custom Tool Development

```python
def advanced_calculator(expression):
    """Advanced calculator with scientific functions."""
    try:
        import math
        # Safe evaluation with math functions
        result = eval(expression, {"__builtins__": {}, "math": math}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Add to ReAct agent
react_agent.add_tool("advanced_calc", advanced_calculator)
```

### Multi-Step Planning

```python
def plan_task(task_description):
    """Plan a complex task by breaking it into steps."""
    # This could integrate with a planning system
    return f"Planned steps for: {task_description}"

react_agent.add_tool("plan", plan_task)
```

### Integration with External APIs

```python
def weather_lookup(city):
    """Get weather information for a city."""
    # This would call a real weather API
    return f"Weather in {city}: 72°F, sunny"

react_agent.add_tool("weather", weather_lookup)
```

## Conclusion

ReAct provides a powerful framework for structured reasoning within CPES. By combining persona-aware reasoning with tool use, it enables sophisticated problem-solving while maintaining character consistency. The key is to design good tools, provide clear context, and let the agent reason step-by-step to solve complex problems.
