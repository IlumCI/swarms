"""
Basic test for CPES implementation.
"""

import sys
from pathlib import Path

# Add CPES to path
sys.path.insert(0, str(Path(__file__).parent))

from core.persona import Persona
from memory.semantic_kg import SemanticKnowledgeGraph
from memory.episodic_memory import EpisodicMemoryStore
from memory.procedural_memory import ProceduralMemoryStore
from utils.embedding import EmbeddingModel
from utils.llm_wrapper import LLMWrapper


def test_persona_loading():
    """Test persona loading from YAML."""
    print("Testing persona loading...")
    
    # Create a simple persona for testing
    persona_data = {
        "name": "Test Persona",
        "motives": [
            {"description": "test motive", "rank": 0.8}
        ],
        "virtues": ["test virtue"],
        "vices": ["test vice"],
        "red_lines": ["never test"],
        "style": {
            "syntax": "test syntax",
            "cadence": "test cadence",
            "tics": ["test tic"]
        },
        "relationships": [
            {"who": "test entity", "valence": "test", "strength": 0.5}
        ],
        "taboos": ["test taboo"]
    }
    
    persona = Persona(persona_data)
    assert persona.spec.name == "Test Persona"
    assert len(persona.spec.motives) == 1
    assert persona.spec.motives[0].description == "test motive"
    print("✅ Persona loading test passed")


def test_semantic_kg():
    """Test semantic knowledge graph."""
    print("Testing semantic knowledge graph...")
    
    kg = SemanticKnowledgeGraph()
    
    # Add some triples
    kg.add_triple("A", "relates_to", "B")
    kg.add_triple("B", "relates_to", "C")
    
    # Query
    beliefs = kg.get_beliefs_about("A")
    assert len(beliefs) > 0
    
    # Check statistics
    stats = kg.get_statistics()
    assert stats['num_triples'] == 2
    
    print("✅ Semantic KG test passed")


def test_episodic_memory():
    """Test episodic memory store."""
    print("Testing episodic memory...")
    
    # Create a simple embedding model for testing
    class MockEmbeddingModel:
        def __init__(self):
            self.dimension = 384
        
        def encode(self, text):
            import numpy as np
            return np.random.rand(384)
    
    embedding_model = MockEmbeddingModel()
    memory = EpisodicMemoryStore(embedding_dim=384)
    
    # Add memory
    memory_id = memory.add_memory(
        text="Test memory",
        embedding=embedding_model.encode("test"),
        tags=["test"],
        people=["test_person"]
    )
    
    assert memory_id is not None
    assert len(memory) == 1
    
    # Search
    results = memory.search(embedding_model.encode("test"), k=1)
    assert len(results) == 1
    
    print("✅ Episodic memory test passed")


def test_procedural_memory():
    """Test procedural memory store."""
    print("Testing procedural memory...")
    
    memory = ProceduralMemoryStore()
    
    # Add skill
    skill_id = memory.add_skill(
        name="Test Skill",
        description="A test skill",
        steps=["Step 1", "Step 2"],
        tools=["Tool 1"],
        category="test"
    )
    
    assert skill_id is not None
    assert len(memory) == 1
    
    # Search
    skills = memory.search_skills("test", limit=1)
    assert len(skills) == 1
    assert skills[0].name == "Test Skill"
    
    print("✅ Procedural memory test passed")


def test_value_gate():
    """Test value gate controller."""
    print("Testing value gate...")
    
    # Create test persona
    persona_data = {
        "name": "Test Persona",
        "motives": [],
        "virtues": [],
        "vices": [],
        "red_lines": ["never apologize"],
        "style": {"syntax": "", "cadence": "", "tics": []},
        "relationships": [],
        "taboos": []
    }
    
    persona = Persona(persona_data)
    
    from controllers.value_gate import ValueGate
    value_gate = ValueGate(persona)
    
    # Test violation detection
    violations = value_gate.check_text("I'm sorry about that")
    assert len(violations) > 0
    assert any(v.type == "red_line" for v in violations)
    
    print("✅ Value gate test passed")


def test_style_adapter():
    """Test style adapter controller."""
    print("Testing style adapter...")
    
    # Create test persona
    persona_data = {
        "name": "Test Persona",
        "motives": [],
        "virtues": [],
        "vices": [],
        "red_lines": [],
        "style": {
            "syntax": "technical",
            "cadence": "short sentences",
            "tics": ["Let's proceed."]
        },
        "relationships": [],
        "taboos": []
    }
    
    persona = Persona(persona_data)
    
    from controllers.style_adapter import StyleAdapter
    style_adapter = StyleAdapter(persona)
    
    # Test style adaptation
    adapted_text = style_adapter.adapt_text("This is a very long sentence that should be shortened.")
    assert len(adapted_text) > 0
    
    print("✅ Style adapter test passed")


def main():
    """Run all tests."""
    print("🧪 Running CPES basic tests...")
    print("=" * 50)
    
    try:
        test_persona_loading()
        test_semantic_kg()
        test_episodic_memory()
        test_procedural_memory()
        test_value_gate()
        test_style_adapter()
        
        print("\n🎉 All tests passed!")
        print("CPES implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
