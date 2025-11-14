"""
Simple test for CPES core components.
"""

import sys
from pathlib import Path

# Add CPES to path
sys.path.insert(0, str(Path(__file__).parent))

def test_persona_loading():
    """Test persona loading from dictionary."""
    print("Testing persona loading...")
    
    from core.persona import Persona
    
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
    
    from memory.semantic_kg import SemanticKnowledgeGraph
    
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
    
    from memory.episodic_memory import EpisodicMemoryStore
    import numpy as np
    
    memory = EpisodicMemoryStore(embedding_dim=384)
    
    # Add memory with random embedding
    memory_id = memory.add_memory(
        text="Test memory",
        embedding=np.random.rand(384),
        tags=["test"],
        people=["test_person"]
    )
    
    assert memory_id is not None
    assert len(memory) == 1
    
    # Search
    results = memory.search(np.random.rand(384), k=1)
    assert len(results) == 1
    
    print("✅ Episodic memory test passed")


def test_procedural_memory():
    """Test procedural memory store."""
    print("Testing procedural memory...")
    
    from memory.procedural_memory import ProceduralMemoryStore
    
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
    
    from core.persona import Persona
    from controllers.value_gate import ValueGate
    
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
    value_gate = ValueGate(persona)
    
    # Test violation detection
    violations = value_gate.check_text("I'm sorry about that")
    assert len(violations) > 0
    assert any(v.type == "red_line" for v in violations)
    
    print("✅ Value gate test passed")


def test_style_adapter():
    """Test style adapter controller."""
    print("Testing style adapter...")
    
    from core.persona import Persona
    from controllers.style_adapter import StyleAdapter
    
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
    style_adapter = StyleAdapter(persona)
    
    # Test style adaptation
    adapted_text = style_adapter.adapt_text("This is a very long sentence that should be shortened.")
    assert len(adapted_text) > 0
    
    print("✅ Style adapter test passed")


def main():
    """Run all tests."""
    print("🧪 Running CPES simple tests...")
    print("=" * 50)
    
    try:
        test_persona_loading()
        test_semantic_kg()
        test_episodic_memory()
        test_procedural_memory()
        test_value_gate()
        test_style_adapter()
        
        print("\n🎉 All tests passed!")
        print("CPES core components are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
