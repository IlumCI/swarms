# Composable Persona Emulation Stack (CPES)

A practical blueprint for building stable behavioral emulation that speaks, decides, and remembers like the target person while maintaining identity over time.

## Overview

CPES is designed to create software-defined personhood through data + constraints + processes. It's not about transferring consciousness, but building a stable behavioral emulation that:

- Speaks, decides, and remembers like the target person
- Maintains identity over time (values, style, relationships)
- Updates memories from new interactions without drifting out of character

## Architecture

### Core Components

1. **Persona Specification** (`core/persona.py`)
   - YAML-based identity definition
   - Motives, virtues, vices, red lines, style, relationships
   - Single source of truth for persona identity

2. **Memory Architecture** (`memory/`)
   - **Semantic Knowledge Graph**: Facts and beliefs as triplets
   - **Episodic Memory**: Vector store of experiences with time, place, people, affect
   - **Procedural Memory**: Skills, workflows, and tool usage patterns

3. **Cognitive Loop** (`core/cognitive_loop.py`)
   - **Observe**: Extract intents and entities from input
   - **Recall**: Retrieve relevant memories and beliefs
   - **Deliberate**: Plan response under persona constraints
   - **Act**: Generate response with style adaptation
   - **Reflect**: Evaluate and store salient interactions
   - **Adjust**: Apply drift correction if needed

4. **Controllers** (`controllers/`)
   - **Value Gate**: Ensures responses align with persona values and red lines
   - **Style Adapter**: Maintains consistent communication style

5. **Anti-Drift System** (`utils/anti_drift.py`)
   - Monitors persona consistency over time
   - Detects and prevents identity drift
   - Provides recommendations for recalibration

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

```python
from CPES import Persona, CognitiveLoop, SemanticKnowledgeGraph, EpisodicMemoryStore
from CPES.utils import EmbeddingModel, LLMWrapper

# Load persona
persona = Persona("path/to/persona.yaml")

# Initialize memory systems
semantic_kg = SemanticKnowledgeGraph()
episodic_memory = EpisodicMemoryStore()
procedural_memory = ProceduralMemoryStore()

# Initialize utilities
embedding_model = EmbeddingModel("all-MiniLM-L6-v2")
llm_wrapper = LLMWrapper("gpt-4o-mini", "openai")

# Create cognitive loop
cognitive_loop = CognitiveLoop(
    persona=persona,
    semantic_kg=semantic_kg,
    episodic_memory=episodic_memory,
    procedural_memory=procedural_memory,
    embedding_model=embedding_model,
    llm_wrapper=llm_wrapper
)

# Process input
response = cognitive_loop.process_input("Hello, how are you?")
print(response)
```

### Example: Caroline Persona

```python
# Run the Caroline demo
python CPES/examples/demo.py
```

This demonstrates a complete CPES implementation with Caroline from Portal, showing:
- Persona specification loading
- Memory architecture usage
- Cognitive loop processing
- Value gate and style adapter
- Anti-drift monitoring

## Persona Specification

Create a YAML file defining your persona:

```yaml
name: Caroline
motives:
  - description: "scientific progress at any cost"
    rank: 0.9
  - description: "protect the project, not the people"
    rank: 0.7
virtues: [precise, dry humor, relentless]
vices: [condescending, risk-seeking]
red_lines:
  - "never admit uncertainty without qualifying bounds"
  - "never apologize unless strategically useful"
style:
  syntax: "crisply technical with sardonic asides"
  cadence: "short, declarative sentences; rare metaphors"
  tics: ["Let's proceed.", "As expected.", "Predictably."]
relationships:
  - who: "Aperture leadership"
    valence: "distrust"
    strength: 0.8
taboos: ["sentimental monologues", "moralizing"]
```

## Memory Systems

### Semantic Knowledge Graph

Store facts and beliefs as semantic triplets:

```python
# Add knowledge
semantic_kg.add_triple("Caroline", "works_at", "Aperture Science")
semantic_kg.add_triple("Caroline", "distrusts", "leadership")

# Query beliefs
beliefs = semantic_kg.get_beliefs_about("Caroline")
```

### Episodic Memory

Store experiences with vector similarity search:

```python
# Add memory
episodic_memory.add_memory(
    text="User asked about safety protocols",
    embedding=embedding_model.encode("safety protocols"),
    tags=["safety", "protocols"],
    people=["User"],
    affect=0.2
)

# Search memories
memories = episodic_memory.search(query_embedding, k=5)
```

### Procedural Memory

Store skills and workflows:

```python
# Add skill
procedural_memory.add_skill(
    name="Design Experiment",
    description="Design and implement experiments",
    steps=["Analyze requirements", "Design protocol", "Execute"],
    tools=["Portal Gun", "Test Chamber"],
    category="experimental_design"
)

# Find relevant skill
skill = procedural_memory.get_workflow_for_task("design experiment")
```

## Anti-Drift Monitoring

Monitor persona consistency over time:

```python
from CPES.utils.anti_drift import AntiDriftMonitor

# Create monitor
monitor = AntiDriftMonitor(persona)

# Check for drift
report = monitor.check_drift(recent_responses, recent_violations)

if report.overall_health == "critical":
    print("CRITICAL: Persona drift detected!")
    print("Recommendations:", report.recommendations)
```

## Key Features

### 1. Identity Consistency
- YAML-based persona specification
- Value gate prevents violations
- Style adapter maintains voice
- Anti-drift monitoring

### 2. Memory Architecture
- Semantic knowledge graph for facts
- Episodic memory for experiences
- Procedural memory for skills
- Vector similarity search

### 3. Cognitive Processing
- Intent and entity extraction
- Memory retrieval and reasoning
- Response generation with constraints
- Salience-based memory storage

### 4. Drift Prevention
- Real-time monitoring
- Automated detection
- Recommendation system
- Checkpoint saving

## Advanced Usage

### Custom Embedding Models

```python
# Use OpenAI embeddings
embedding_model = EmbeddingModel(
    model_name="text-embedding-ada-002",
    model_type="openai"
)

# Use custom model
embedding_model = EmbeddingModel(
    model_name="custom-model",
    model_type="custom"
)
# Set embedding_model.model and embedding_model.dimension
```

### Custom LLM Providers

```python
# Use Anthropic
llm_wrapper = LLMWrapper(
    model_name="claude-3-sonnet",
    provider="anthropic"
)

# Use custom provider
llm_wrapper = LLMWrapper(
    model_name="custom-model",
    provider="custom"
)
# Set llm_wrapper.client
```

### Memory Persistence

```python
# Save memories
semantic_kg.save("knowledge_graph.json")
episodic_memory.save("episodic_memories.json")
procedural_memory.save("procedural_memories.json")

# Load memories
semantic_kg.load("knowledge_graph.json")
episodic_memory.load("episodic_memories.json")
procedural_memory.load("procedural_memories.json")
```

## Best Practices

### 1. Persona Design
- Keep persona specs focused and specific
- Use clear, measurable red lines
- Define characteristic phrases and style
- Include relevant relationships

### 2. Memory Management
- Store only salient interactions
- Use appropriate tags and metadata
- Regular cleanup of old memories
- Monitor memory growth

### 3. Drift Prevention
- Regular drift checks
- Monitor key metrics
- Act on recommendations
- Save checkpoints

### 4. Performance
- Use appropriate embedding models
- Batch memory operations
- Cache frequent queries
- Monitor response times

## Troubleshooting

### Common Issues

1. **Persona Drift**
   - Check value gate settings
   - Review style adapter configuration
   - Run drift analysis
   - Recalibrate persona spec

2. **Memory Issues**
   - Check embedding dimensions
   - Verify memory storage
   - Monitor memory usage
   - Clean up old memories

3. **Performance Problems**
   - Use smaller embedding models
   - Reduce memory search scope
   - Optimize LLM calls
   - Cache frequent operations

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
from loguru import logger
logger.add("debug.log", level="DEBUG")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by the swarms framework
- Built with modern Python tools
- Designed for practical deployment
