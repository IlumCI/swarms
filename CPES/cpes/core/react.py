"""
ReAct (Reasoning and Acting) Module for CPES.

This module implements the ReAct pattern for structured reasoning and tool use,
allowing the persona to think, act, and observe in a controlled loop.
"""

import re
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from loguru import logger
import time


@dataclass
class ReActStep:
    """Represents a single ReAct step."""
    step_number: int
    thought: str
    action: str
    observation: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReActResult:
    """Represents the result of a ReAct reasoning session."""
    question: str
    steps: List[ReActStep]
    final_answer: str
    success: bool
    total_steps: int
    reasoning_time: float


class ReActAgent:
    """
    ReAct agent that implements structured reasoning and acting.
    
    This class provides a framework for agents to reason about problems
    by interleaving thoughts, actions, and observations in a controlled loop.
    """
    
    def __init__(self, llm_wrapper, tools: Dict[str, Callable], 
                 max_steps: int = 8, persona_context: str = ""):
        """
        Initialize the ReAct agent.
        
        Args:
            llm_wrapper: LLM wrapper for text generation
            tools: Dictionary of available tools (name -> function)
            max_steps: Maximum number of reasoning steps
            persona_context: Persona context for reasoning style
        """
        self.llm_wrapper = llm_wrapper
        self.tools = tools
        self.max_steps = max_steps
        self.persona_context = persona_context
        
        # ReAct prompt template
        self.prompt_template = self._build_prompt_template()
        
        logger.info(f"Initialized ReAct agent with {len(tools)} tools")
    
    def _build_prompt_template(self) -> str:
        """Build the ReAct prompt template."""
        tools_description = self._format_tools_description()
        
        template = f"""You are a reasoning agent with a specific personality and expertise.

{self.persona_context}

For each input, follow this exact pattern:

Question: {{question}}
Thought: {{your reasoning here}}
Action: {{choose one of the tools and use it, or say "Finish[final answer]"}}
Observation: {{the result from the tool}}
Thought: ...
Action: ...
...
Answer: {{final answer}}

Available tools:
{tools_description}

Rules:
1. Always follow the Thought -> Action -> Observation pattern
2. Use tools to gather information or perform calculations
3. Think step by step and be methodical
4. When you have enough information, use Finish[your answer]
5. Do not output anything outside this format
6. Be concise but thorough in your reasoning

Let's begin:

Question: {{question}}
Thought:"""
        
        return template
    
    def _format_tools_description(self) -> str:
        """Format tools description for the prompt."""
        descriptions = []
        for i, (tool_name, tool_func) in enumerate(self.tools.items(), 1):
            # Get tool description from docstring or use name
            description = getattr(tool_func, '__doc__', tool_name)
            if description:
                description = description.strip().split('\n')[0]
            else:
                description = f"Tool for {tool_name}"
            
            descriptions.append(f"{i}. {tool_name.upper()}[args] — {description}")
        
        return "\n".join(descriptions)
    
    def reason(self, question: str) -> ReActResult:
        """
        Perform ReAct reasoning on a question.
        
        Args:
            question: The question to reason about
            
        Returns:
            ReActResult with the reasoning process and final answer
        """
        logger.info(f"Starting ReAct reasoning for: {question[:100]}...")
        
        start_time = time.time()
        steps = []
        context = self.prompt_template.format(question=question)
        
        for step_num in range(1, self.max_steps + 1):
            try:
                # Generate model response
                response = self.llm_wrapper.generate(
                    prompt=context,
                    temperature=0.3,  # Lower temperature for more structured reasoning
                    max_tokens=500
                )
                
                # Parse the response
                thought, action = self._parse_response(response.content)
                
                if not thought and not action:
                    logger.warning(f"Failed to parse response at step {step_num}")
                    break
                
                # Execute action
                observation = self._execute_action(action)
                
                # Create step record
                step = ReActStep(
                    step_number=step_num,
                    thought=thought,
                    action=action,
                    observation=observation
                )
                steps.append(step)
                
                # Update context
                context += f"\nThought: {thought}\nAction: {action}\nObservation: {observation}\nThought:"
                
                # Check if we're done
                if action.startswith("Finish["):
                    final_answer = action[7:-1]  # Extract from Finish[answer]
                    success = True
                    break
                
                # Check for completion signal
                if "Answer:" in response.content:
                    final_answer = self._extract_final_answer(response.content)
                    success = True
                    break
                
            except Exception as e:
                logger.error(f"Error in ReAct step {step_num}: {e}")
                step = ReActStep(
                    step_number=step_num,
                    thought="Error occurred",
                    action="Error",
                    observation=f"Error: {str(e)}"
                )
                steps.append(step)
                break
        
        else:
            # Max steps reached without completion
            final_answer = "Unable to reach a conclusion within the maximum steps."
            success = False
        
        reasoning_time = time.time() - start_time
        
        result = ReActResult(
            question=question,
            steps=steps,
            final_answer=final_answer,
            success=success,
            total_steps=len(steps),
            reasoning_time=reasoning_time
        )
        
        logger.info(f"ReAct reasoning completed: {len(steps)} steps, success: {success}")
        return result
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse the model response to extract thought and action."""
        # Extract thought
        thought_match = re.search(r'Thought:\s*(.*?)(?=\nAction:|$)', response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        # Extract action
        action_match = re.search(r'Action:\s*(.*?)(?=\nObservation:|$)', response, re.DOTALL)
        action = action_match.group(1).strip() if action_match else ""
        
        return thought, action
    
    def _execute_action(self, action: str) -> str:
        """Execute the specified action using available tools."""
        if not action:
            return "No action specified"
        
        # Check for Finish action
        if action.startswith("Finish["):
            return "Final answer provided"
        
        # Parse tool call
        tool_match = re.match(r'(\w+)\[(.*?)\]', action)
        if not tool_match:
            return f"Invalid action format: {action}"
        
        tool_name, args = tool_match.groups()
        tool_name = tool_name.lower()
        
        # Execute tool
        if tool_name in self.tools:
            try:
                result = self.tools[tool_name](args)
                return str(result)
            except Exception as e:
                return f"Tool execution error: {str(e)}"
        else:
            return f"Unknown tool: {tool_name}"
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from the response."""
        answer_match = re.search(r'Answer:\s*(.*?)$', response, re.DOTALL)
        return answer_match.group(1).strip() if answer_match else "No answer found"
    
    def add_tool(self, name: str, func: Callable, description: str = "") -> None:
        """Add a new tool to the agent."""
        self.tools[name] = func
        if description:
            func.__doc__ = description
        logger.info(f"Added tool: {name}")
    
    def get_reasoning_summary(self, result: ReActResult) -> str:
        """Get a summary of the reasoning process."""
        summary = f"ReAct Reasoning Summary:\n"
        summary += f"Question: {result.question}\n"
        summary += f"Steps: {result.total_steps}\n"
        summary += f"Success: {result.success}\n"
        summary += f"Time: {result.reasoning_time:.2f}s\n\n"
        
        for step in result.steps:
            summary += f"Step {step.step_number}:\n"
            summary += f"  Thought: {step.thought}\n"
            summary += f"  Action: {step.action}\n"
            summary += f"  Observation: {step.observation}\n\n"
        
        summary += f"Final Answer: {result.final_answer}\n"
        return summary
    
    def __str__(self) -> str:
        """String representation."""
        return f"ReActAgent(tools={len(self.tools)}, max_steps={self.max_steps})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"ReActAgent(tools={list(self.tools.keys())}, max_steps={self.max_steps})"
