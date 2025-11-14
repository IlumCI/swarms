"""
Composable Persona Emulation Stack (CPES)

A practical blueprint for building stable behavioral emulation that:
- Speaks, decides, and remembers like the target person
- Maintains identity over time (values, style, relationships)
- Updates memories from new interactions without drifting out of character

Think of it as software-defined personhood: data + constraints + processes.
"""

from .core.persona import Persona
from .core.cognitive_loop import CognitiveLoop
from .core.react import ReActAgent
from .memory.semantic_kg import SemanticKnowledgeGraph
from .memory.episodic_memory import EpisodicMemoryStore
from .memory.procedural_memory import ProceduralMemoryStore
from .controllers.value_gate import ValueGate
from .controllers.style_adapter import StyleAdapter
from .utils.embedding import EmbeddingModel
from .utils.llm_wrapper import LLMWrapper
from .tools.basic_tools import BasicTools
from .tools.search_tools import SearchTools
from .tools.memory_tools import MemoryTools

__version__ = "0.1.0"
__author__ = "CPES Team"

__all__ = [
    "Persona",
    "CognitiveLoop",
    "ReActAgent",
    "SemanticKnowledgeGraph",
    "EpisodicMemoryStore",
    "ProceduralMemoryStore",
    "ValueGate",
    "StyleAdapter",
    "EmbeddingModel",
    "LLMWrapper",
    "BasicTools",
    "SearchTools",
    "MemoryTools",
]
