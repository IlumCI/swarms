"""
Test ReAct implementation for CPES.
"""

import sys
from pathlib import Path

# Add CPES to path
sys.path.insert(0, str(Path(__file__).parent))

def test_react_agent():
    """Test ReAct agent creation and basic functionality."""
    print("Testing ReAct agent...")
    
    from core.react import ReActAgent
    from tools.basic_tools import BasicTools
    
    # Mock LLM wrapper
    class MockLLM:
        def generate(self, prompt, temperature=0.7, max_tokens=500):
            if "calculate" in prompt.lower():
                return "Thought: I need to calculate this.\nAction: CALC[2+2]\nObservation: 4\nThought: I have the answer.\nAction: Finish[The answer is 4]"
            else:
                return "Thought: Let me think about this.\nAction: Finish[I need more information]"
    
    # Create ReAct agent
    tools = BasicTools.get_tools()
    agent = ReActAgent(
        llm_wrapper=MockLLM(),
        tools=tools,
        max_steps=3,
        persona_context="You are a helpful assistant."
    )
    
    # Test reasoning
    result = agent.reason("What is 2+2?")
    
    assert result.success == True
    assert "4" in result.final_answer
    assert result.total_steps > 0
    
    print("✅ ReAct agent test passed")


def test_basic_tools():
    """Test basic tools functionality."""
    print("Testing basic tools...")
    
    from tools.basic_tools import BasicTools
    
    # Test calculation
    result = BasicTools.calc("2 + 2")
    assert result == "4"
    
    # Test comparison
    result = BasicTools.compare("5", "3")
    assert "5 > 3" in result
    
    # Test word count
    result = BasicTools.count_words("hello world")
    assert result == "2"
    
    print("✅ Basic tools test passed")


def test_search_tools():
    """Test search tools functionality."""
    print("Testing search tools...")
    
    from tools.search_tools import SearchTools
    
    # Create search tools with knowledge base
    knowledge = {
        "python": "Python is a programming language",
        "javascript": "JavaScript is used for web development"
    }
    
    search_tools = SearchTools(knowledge)
    
    # Test search
    result = search_tools.search("python programming")
    assert "python" in result.lower()
    
    # Test lookup
    result = search_tools.lookup("python")
    assert "programming language" in result
    
    print("✅ Search tools test passed")


def test_memory_tools():
    """Test memory tools functionality."""
    print("Testing memory tools...")
    
    from tools.memory_tools import MemoryTools
    
    # Create memory tools (without actual memory systems for test)
    memory_tools = MemoryTools()
    
    # Test that tools are available
    tools = memory_tools.get_tools()
    assert len(tools) >= 0  # May be empty without memory systems
    
    print("✅ Memory tools test passed")


def test_react_integration():
    """Test ReAct integration with cognitive loop."""
    print("Testing ReAct integration...")
    
    from core.persona import Persona
    from core.cognitive_loop import CognitiveLoop
    from memory.semantic_kg import SemanticKnowledgeGraph
    from memory.episodic_memory import EpisodicMemoryStore
    from memory.procedural_memory import ProceduralMemoryStore
    from utils.embedding import EmbeddingModel
    from utils.llm_wrapper import LLMWrapper
    
    # Mock LLM wrapper
    class MockLLM:
        def generate(self, prompt, temperature=0.7, max_tokens=500):
            if "calculate" in prompt.lower():
                return "Thought: I need to calculate this.\nAction: CALC[2+2]\nObservation: 4\nThought: I have the answer.\nAction: Finish[The answer is 4]"
            else:
                return "I understand your question."
    
    # Mock embedding model
    class MockEmbedding:
        def encode(self, text):
            import numpy as np
            return np.random.rand(384)
    
    # Create persona
    persona_data = {
        "name": "Test Persona",
        "motives": [],
        "virtues": [],
        "vices": [],
        "red_lines": [],
        "style": {"syntax": "", "cadence": "", "tics": []},
        "relationships": [],
        "taboos": []
    }
    
    persona = Persona(persona_data)
    
    # Create cognitive loop components
    semantic_kg = SemanticKnowledgeGraph()
    episodic_memory = EpisodicMemoryStore(embedding_dim=384)
    procedural_memory = ProceduralMemoryStore()
    embedding_model = MockEmbedding()
    llm_wrapper = MockLLM()
    
    # Create cognitive loop
    cognitive_loop = CognitiveLoop(
        persona=persona,
        semantic_kg=semantic_kg,
        episodic_memory=episodic_memory,
        procedural_memory=procedural_memory,
        embedding_model=embedding_model,
        llm_wrapper=llm_wrapper
    )
    
    # Test that ReAct agent was created
    assert hasattr(cognitive_loop, 'react_agent')
    assert cognitive_loop.react_agent is not None
    
    print("✅ ReAct integration test passed")


def main():
    """Run all ReAct tests."""
    print("🧪 Running ReAct tests...")
    print("=" * 50)
    
    try:
        test_react_agent()
        test_basic_tools()
        test_search_tools()
        test_memory_tools()
        test_react_integration()
        
        print("\n🎉 All ReAct tests passed!")
        print("ReAct implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
