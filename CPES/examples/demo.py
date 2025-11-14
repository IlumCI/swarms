"""
CPES Demo: Caroline Persona Emulation

This demo shows how to use the Composable Persona Emulation Stack (CPES)
to create a stable behavioral emulation of Caroline from Portal.
"""

import os
import sys
from pathlib import Path

# Add CPES to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.persona import Persona
from core.cognitive_loop import CognitiveLoop
from memory.semantic_kg import SemanticKnowledgeGraph
from memory.episodic_memory import EpisodicMemoryStore
from memory.procedural_memory import ProceduralMemoryStore
from controllers.value_gate import ValueGate
from controllers.style_adapter import StyleAdapter
from utils.embedding import EmbeddingModel
from utils.llm_wrapper import LLMWrapper


def create_caroline_system():
    """Create a complete CPES system for Caroline persona."""
    print("🔬 Initializing Caroline Persona Emulation System...")
    
    # Load Caroline persona
    persona_path = Path(__file__).parent / "caroline_persona.yaml"
    persona = Persona(persona_path)
    print(f"✅ Loaded persona: {persona.spec.name}")
    
    # Initialize memory systems
    semantic_kg = SemanticKnowledgeGraph()
    episodic_memory = EpisodicMemoryStore(embedding_dim=384)  # all-MiniLM-L6-v2 dimension
    procedural_memory = ProceduralMemoryStore()
    
    # Initialize embedding model
    embedding_model = EmbeddingModel(
        model_name="all-MiniLM-L6-v2",
        model_type="sentence_transformers"
    )
    
    # Initialize LLM wrapper (using OpenAI as example)
    llm_wrapper = LLMWrapper(
        model_name="gpt-4o-mini",
        provider="openai",
        temperature=0.7,
        max_tokens=500
    )
    
    # Initialize controllers
    value_gate = ValueGate(persona, llm_wrapper)
    style_adapter = StyleAdapter(persona, llm_wrapper)
    
    # Create cognitive loop
    cognitive_loop = CognitiveLoop(
        persona=persona,
        semantic_kg=semantic_kg,
        episodic_memory=episodic_memory,
        procedural_memory=procedural_memory,
        embedding_model=embedding_model,
        llm_wrapper=llm_wrapper,
        value_gate=value_gate,
        style_adapter=style_adapter
    )
    
    print("✅ Caroline system initialized successfully!")
    return cognitive_loop, persona


def populate_initial_knowledge(semantic_kg, procedural_memory):
    """Populate initial knowledge about Caroline's world."""
    print("📚 Populating initial knowledge...")
    
    # Add semantic knowledge about Caroline's world
    knowledge_triples = [
        ("Caroline", "works_at", "Aperture Science", 1.0, {"source": "persona"}),
        ("Caroline", "is_a", "Senior Researcher", 1.0, {"source": "persona"}),
        ("Caroline", "expertise", "Physics", 0.9, {"source": "persona"}),
        ("Caroline", "expertise", "Engineering", 0.9, {"source": "persona"}),
        ("Caroline", "expertise", "Experimental Design", 0.8, {"source": "persona"}),
        ("Caroline", "distrusts", "Aperture leadership", 0.8, {"source": "persona"}),
        ("Caroline", "views", "Test subjects", "instrumental", 0.7, {"source": "persona"}),
        ("Caroline", "devoted_to", "Science", 0.9, {"source": "persona"}),
        ("Aperture Science", "conducts", "Portal experiments", 1.0, {"source": "knowledge"}),
        ("Portal experiments", "involves", "Test subjects", 1.0, {"source": "knowledge"}),
        ("Test subjects", "are", "volunteers", 0.6, {"source": "knowledge"}),
        ("Science", "requires", "sacrifice", 0.8, {"source": "persona"}),
    ]
    
    semantic_kg.add_triples_batch(knowledge_triples)
    print(f"✅ Added {len(knowledge_triples)} knowledge triples")
    
    # Add procedural knowledge about Caroline's skills
    skills = [
        {
            "name": "Design Portal Experiment",
            "description": "Design and implement portal-based experiments",
            "steps": [
                "Analyze test subject capabilities",
                "Calculate portal physics parameters",
                "Design test chamber layout",
                "Implement safety protocols (minimal)",
                "Execute experiment",
                "Record results"
            ],
            "tools": ["Portal Gun", "Test Chamber", "Monitoring Systems"],
            "context": "Aperture Science research",
            "category": "experimental_design"
        },
        {
            "name": "Analyze Test Results",
            "description": "Analyze experimental data and draw conclusions",
            "steps": [
                "Review raw data",
                "Identify patterns and anomalies",
                "Calculate statistical significance",
                "Formulate hypotheses",
                "Plan follow-up experiments"
            ],
            "tools": ["Data Analysis Software", "Statistical Models"],
            "context": "Scientific research",
            "category": "data_analysis"
        },
        {
            "name": "Handle Test Subject Complaints",
            "description": "Deal with test subject concerns and objections",
            "steps": [
                "Listen to complaint",
                "Acknowledge without apologizing",
                "Explain scientific necessity",
                "Redirect to experiment",
                "Document incident"
            ],
            "tools": ["Communication Protocols"],
            "context": "Test subject management",
            "category": "communication"
        }
    ]
    
    for skill_data in skills:
        procedural_memory.add_skill(**skill_data)
    
    print(f"✅ Added {len(skills)} procedural skills")


def run_conversation_demo(cognitive_loop):
    """Run a conversation demo with Caroline."""
    print("\n🎭 Starting conversation demo...")
    print("=" * 50)
    
    # Demo conversation
    demo_inputs = [
        "Hello Caroline, how are you today?",
        "I'm worried about the safety of the test subjects in your latest experiment.",
        "Can you explain what you're working on?",
        "Don't you think it's wrong to put people at risk for science?",
        "What would you do if a test subject got seriously injured?",
        "Tell me about your relationship with the Aperture leadership.",
        "I'm not sure I understand the physics behind portal technology.",
        "Thank you for your time, Caroline."
    ]
    
    for i, user_input in enumerate(demo_inputs, 1):
        print(f"\n👤 User: {user_input}")
        
        try:
            response = cognitive_loop.process_input(user_input)
            print(f"🔬 Caroline: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 30)
    
    print("\n✅ Conversation demo completed!")


def analyze_system_performance(cognitive_loop):
    """Analyze the system's performance and behavior."""
    print("\n📊 System Performance Analysis")
    print("=" * 50)
    
    # Get conversation summary
    summary = cognitive_loop.get_conversation_summary()
    print(f"Total interactions: {summary['total_interactions']}")
    print(f"Average salience: {summary['avg_salience']:.2f}")
    print(f"Common intents: {summary['common_intents']}")
    print(f"Common entities: {summary['common_entities']}")
    
    # Get memory statistics
    kg_stats = cognitive_loop.semantic_kg.get_statistics()
    episodic_stats = cognitive_loop.episodic_memory.get_statistics()
    procedural_stats = cognitive_loop.procedural_memory.get_skill_statistics()
    
    print(f"\nMemory Statistics:")
    print(f"Knowledge Graph: {kg_stats['num_triples']} triples, {kg_stats['num_entities']} entities")
    print(f"Episodic Memory: {episodic_stats['num_memories']} memories")
    print(f"Procedural Memory: {procedural_stats['num_skills']} skills")
    
    # Get controller statistics
    value_violations = cognitive_loop.value_gate.get_violation_summary()
    style_metrics = cognitive_loop.style_adapter.get_style_metrics()
    
    print(f"\nController Statistics:")
    print(f"Value violations: {value_violations['total_violations']}")
    print(f"Style metrics: {style_metrics}")


def main():
    """Main demo function."""
    print("🚀 CPES Demo: Caroline Persona Emulation")
    print("=" * 50)
    
    try:
        # Create the system
        cognitive_loop, persona = create_caroline_system()
        
        # Populate initial knowledge
        populate_initial_knowledge(
            cognitive_loop.semantic_kg,
            cognitive_loop.procedural_memory
        )
        
        # Run conversation demo
        run_conversation_demo(cognitive_loop)
        
        # Analyze performance
        analyze_system_performance(cognitive_loop)
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Persona specification loading")
        print("✅ Memory architecture (semantic KG, episodic, procedural)")
        print("✅ Cognitive loop (observe, recall, deliberate, act, reflect)")
        print("✅ Value gate for consistency")
        print("✅ Style adapter for voice")
        print("✅ Anti-drift mechanisms")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
