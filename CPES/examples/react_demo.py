"""
ReAct Demo for CPES.

This demo shows how the ReAct (Reasoning and Acting) system works
within the CPES framework, demonstrating structured reasoning and tool use.
"""

import sys
from pathlib import Path

# Add CPES to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.persona import Persona
from core.react import ReActAgent
from tools.basic_tools import BasicTools
from tools.search_tools import SearchTools
from tools.memory_tools import MemoryTools
from utils.llm_wrapper import LLMWrapper


def create_react_demo():
    """Create a ReAct demo with Caroline persona."""
    print("🔬 Initializing ReAct Demo with Caroline Persona...")
    
    # Load Caroline persona
    persona_path = Path(__file__).parent / "caroline_persona.yaml"
    persona = Persona(persona_path)
    print(f"✅ Loaded persona: {persona.spec.name}")
    
    # Initialize LLM wrapper (mock for demo)
    class MockLLMWrapper:
        def generate(self, prompt, temperature=0.7, max_tokens=500):
            # Simple mock response based on prompt content
            if "calculate" in prompt.lower():
                return "Thought: I need to calculate this step by step.\nAction: CALC[2 + 2]\nObservation: 4\nThought: Now I have the result.\nAction: Finish[The answer is 4]"
            elif "search" in prompt.lower():
                return "Thought: I should search for this information.\nAction: SEARCH[test query]\nObservation: Found some results.\nThought: I have the information I need.\nAction: Finish[Based on my search, here's what I found]"
            else:
                return "Thought: Let me think about this.\nAction: Finish[I need more information to answer this properly]"
    
    llm_wrapper = MockLLMWrapper()
    
    # Initialize tools
    basic_tools = BasicTools.get_tools()
    search_tools = SearchTools().get_tools()
    
    # Create ReAct agent
    react_agent = ReActAgent(
        llm_wrapper=llm_wrapper,
        tools={**basic_tools, **search_tools},
        max_steps=5,
        persona_context=persona.get_identity_context()
    )
    
    print("✅ ReAct agent initialized")
    return react_agent, persona


def run_react_examples(react_agent, persona):
    """Run example ReAct reasoning sessions."""
    print("\n🎭 ReAct Reasoning Examples")
    print("=" * 50)
    
    # Example 1: Mathematical reasoning
    print("\n📊 Example 1: Mathematical Reasoning")
    print("-" * 30)
    question1 = "What is 15 * 23 + 45?"
    print(f"Question: {question1}")
    
    result1 = react_agent.reason(question1)
    print(f"Answer: {result1.final_answer}")
    print(f"Steps: {result1.total_steps}")
    print(f"Success: {result1.success}")
    
    # Example 2: Information search
    print("\n🔍 Example 2: Information Search")
    print("-" * 30)
    question2 = "Search for information about portal technology"
    print(f"Question: {question2}")
    
    result2 = react_agent.reason(question2)
    print(f"Answer: {result2.final_answer}")
    print(f"Steps: {result2.total_steps}")
    print(f"Success: {result2.success}")
    
    # Example 3: Complex reasoning
    print("\n🧠 Example 3: Complex Reasoning")
    print("-" * 30)
    question3 = "Compare the population of Vilnius (593,000) with the population of Riga (632,000) and tell me which is larger"
    print(f"Question: {question3}")
    
    result3 = react_agent.reason(question3)
    print(f"Answer: {result3.final_answer}")
    print(f"Steps: {result3.total_steps}")
    print(f"Success: {result3.success}")
    
    return [result1, result2, result3]


def demonstrate_reasoning_process(react_agent, results):
    """Demonstrate the detailed reasoning process."""
    print("\n🔬 Detailed Reasoning Process")
    print("=" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\nExample {i} Reasoning Steps:")
        print("-" * 20)
        
        for step in result.steps:
            print(f"Step {step.step_number}:")
            print(f"  Thought: {step.thought}")
            print(f"  Action: {step.action}")
            print(f"  Observation: {step.observation}")
            print()


def show_react_capabilities(react_agent):
    """Show ReAct agent capabilities."""
    print("\n🛠️ ReAct Agent Capabilities")
    print("=" * 50)
    
    print("Available Tools:")
    for tool_name, tool_func in react_agent.tools.items():
        description = getattr(tool_func, '__doc__', 'No description available')
        print(f"  - {tool_name}: {description}")
    
    print(f"\nConfiguration:")
    print(f"  - Max Steps: {react_agent.max_steps}")
    print(f"  - Total Tools: {len(react_agent.tools)}")
    print(f"  - Persona Context: {len(react_agent.persona_context)} characters")


def main():
    """Main demo function."""
    print("🚀 CPES ReAct Demo")
    print("=" * 50)
    
    try:
        # Create ReAct demo
        react_agent, persona = create_react_demo()
        
        # Run examples
        results = run_react_examples(react_agent, persona)
        
        # Show detailed reasoning
        demonstrate_reasoning_process(react_agent, results)
        
        # Show capabilities
        show_react_capabilities(react_agent)
        
        print("\n🎉 ReAct Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Structured reasoning (Thought -> Action -> Observation)")
        print("✅ Tool integration and execution")
        print("✅ Persona-aware reasoning style")
        print("✅ Step-by-step problem solving")
        print("✅ Fallback mechanisms")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
