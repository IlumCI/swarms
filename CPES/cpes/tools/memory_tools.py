"""
Memory tools for ReAct agents.

This module provides memory-related tools that ReAct agents can use
to store and retrieve information from various memory systems.
"""

from typing import Any, Dict, List, Optional
from loguru import logger
import numpy as np


class MemoryTools:
    """Collection of memory tools for ReAct agents."""
    
    def __init__(self, episodic_memory=None, semantic_kg=None, procedural_memory=None):
        """
        Initialize memory tools.
        
        Args:
            episodic_memory: Episodic memory store
            semantic_kg: Semantic knowledge graph
            procedural_memory: Procedural memory store
        """
        self.episodic_memory = episodic_memory
        self.semantic_kg = semantic_kg
        self.procedural_memory = procedural_memory
    
    def recall_episodic(self, query: str, k: int = 3) -> str:
        """
        Recall episodic memories based on query.
        
        Args:
            query: Query to search memories
            k: Number of memories to retrieve
            
        Returns:
            Retrieved memories
        """
        try:
            if not self.episodic_memory:
                return "Episodic memory not available"
            
            # Generate embedding for query (simplified)
            query_embedding = np.random.rand(384)  # Placeholder
            
            memories = self.episodic_memory.search(query_embedding, k=k)
            
            if not memories:
                return f"No memories found for: {query}"
            
            results = []
            for i, (memory, score) in enumerate(memories, 1):
                results.append(f"{i}. {memory.text[:100]}... (score: {score:.2f})")
            
            return f"Found {len(memories)} memories:\n" + "\n".join(results)
            
        except Exception as e:
            return f"Episodic recall error: {str(e)}"
    
    def recall_semantic(self, entity: str) -> str:
        """
        Recall semantic knowledge about an entity.
        
        Args:
            entity: Entity to get knowledge about
            
        Returns:
            Semantic knowledge
        """
        try:
            if not self.semantic_kg:
                return "Semantic knowledge graph not available"
            
            beliefs = self.semantic_kg.get_beliefs_about(entity, max_results=5)
            
            if not beliefs:
                return f"No knowledge found about: {entity}"
            
            return f"Knowledge about {entity}:\n" + "\n".join(f"- {belief}" for belief in beliefs)
            
        except Exception as e:
            return f"Semantic recall error: {str(e)}"
    
    def recall_procedural(self, task: str) -> str:
        """
        Recall procedural knowledge for a task.
        
        Args:
            task: Task to get procedures for
            
        Returns:
            Procedural knowledge
        """
        try:
            if not self.procedural_memory:
                return "Procedural memory not available"
            
            skills = self.procedural_memory.search_skills(task, limit=3)
            
            if not skills:
                return f"No procedures found for: {task}"
            
            results = []
            for i, skill in enumerate(skills, 1):
                results.append(f"{i}. {skill.name}: {skill.description}")
                results.append(f"   Steps: {' -> '.join(skill.steps[:3])}")
            
            return f"Found {len(skills)} procedures:\n" + "\n".join(results)
            
        except Exception as e:
            return f"Procedural recall error: {str(e)}"
    
    def store_episodic(self, text: str, tags: List[str] = None) -> str:
        """
        Store information in episodic memory.
        
        Args:
            text: Text to store
            tags: Optional tags for the memory
            
        Returns:
            Confirmation message
        """
        try:
            if not self.episodic_memory:
                return "Episodic memory not available"
            
            # Generate embedding (simplified)
            embedding = np.random.rand(384)  # Placeholder
            
            memory_id = self.episodic_memory.add_memory(
                text=text,
                embedding=embedding,
                tags=tags or [],
                people=["ReAct Agent"]
            )
            
            return f"Stored episodic memory: {memory_id}"
            
        except Exception as e:
            return f"Episodic store error: {str(e)}"
    
    def store_semantic(self, subject: str, predicate: str, object: str) -> str:
        """
        Store semantic knowledge.
        
        Args:
            subject: Subject of the triple
            predicate: Predicate of the triple
            object: Object of the triple
            
        Returns:
            Confirmation message
        """
        try:
            if not self.semantic_kg:
                return "Semantic knowledge graph not available"
            
            triple_id = self.semantic_kg.add_triple(subject, predicate, object)
            return f"Stored semantic knowledge: {subject} {predicate} {object}"
            
        except Exception as e:
            return f"Semantic store error: {str(e)}"
    
    def get_tools(self) -> Dict[str, callable]:
        """Get dictionary of available memory tools."""
        tools = {}
        
        if self.episodic_memory:
            tools.update({
                "recall_episodic": self.recall_episodic,
                "store_episodic": self.store_episodic
            })
        
        if self.semantic_kg:
            tools.update({
                "recall_semantic": self.recall_semantic,
                "store_semantic": self.store_semantic
            })
        
        if self.procedural_memory:
            tools["recall_procedural"] = self.recall_procedural
        
        return tools
