"""
Test script to verify CPES package functionality.
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from cpes import (
            Persona, CognitiveLoop, ReActAgent,
            SemanticKnowledgeGraph, EpisodicMemoryStore, ProceduralMemoryStore,
            ValueGate, StyleAdapter, EmbeddingModel, LLMWrapper,
            BasicTools, SearchTools, MemoryTools
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_persona_creation():
    """Test persona creation from data."""
    print("\nTesting persona creation...")
    
    try:
        from cpes import Persona
        
        # Test persona data
        persona_data = {
            "name": "TestPersona",
            "motives": [
                {"description": "test motive", "rank": 0.8}
            ],
            "virtues": ["test_virtue"],
            "vices": ["test_vice"],
            "red_lines": ["test red line"],
            "style": {
                "syntax": "test syntax",
                "cadence": "test cadence",
                "tics": ["test tic"]
            },
            "relationships": [
                {"who": "test_entity", "valence": "neutral", "strength": 0.5}
            ],
            "taboos": ["test taboo"]
        }
        
        persona = Persona(persona_data)
        print(f"✅ Persona created: {persona.spec.name}")
        return True
    except Exception as e:
        print(f"❌ Persona creation failed: {e}")
        return False

def test_llm_wrapper():
    """Test LLM wrapper initialization."""
    print("\nTesting LLM wrapper...")
    
    try:
        from cpes import LLMWrapper
        
        # Test with mock API key
        llm_wrapper = LLMWrapper(
            model_name="gpt-4o-mini",
            provider="openai",
            api_key="test-key"
        )
        print(f"✅ LLM wrapper created: {llm_wrapper.provider}/{llm_wrapper.model_name}")
        
        # Test LiteLLM wrapper
        llm_wrapper_litellm = LLMWrapper(
            model_name="gpt-4o-mini",
            provider="litellm",
            api_key="test-key",
            use_litellm=True
        )
        print(f"✅ LiteLLM wrapper created: {llm_wrapper_litellm.provider}/{llm_wrapper_litellm.model_name}")
        return True
    except Exception as e:
        print(f"❌ LLM wrapper creation failed: {e}")
        return False

def test_memory_systems():
    """Test memory systems initialization."""
    print("\nTesting memory systems...")
    
    try:
        from cpes import SemanticKnowledgeGraph, EpisodicMemoryStore, ProceduralMemoryStore
        
        semantic_kg = SemanticKnowledgeGraph()
        episodic_memory = EpisodicMemoryStore()
        procedural_memory = ProceduralMemoryStore()
        
        print("✅ All memory systems initialized")
        return True
    except Exception as e:
        print(f"❌ Memory systems initialization failed: {e}")
        return False

def test_embedding_model():
    """Test embedding model initialization."""
    print("\nTesting embedding model...")
    
    try:
        from cpes import EmbeddingModel
        
        embedding_model = EmbeddingModel("all-MiniLM-L6-v2")
        print("✅ Embedding model initialized")
        return True
    except Exception as e:
        print(f"❌ Embedding model initialization failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 CPES Package Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_persona_creation,
        test_llm_wrapper,
        test_memory_systems,
        test_embedding_model,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CPES package is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
