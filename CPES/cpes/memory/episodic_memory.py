"""
Episodic Memory for CPES.

This module implements episodic memory using vector similarity search for storing
and retrieving specific experiences, moments, and interactions with time, place, people, and affect.
"""

import numpy as np
import faiss
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
from loguru import logger


@dataclass
class EpisodicMemory:
    """Represents a single episodic memory."""
    id: str
    text: str
    embedding: np.ndarray
    timestamp: float
    tags: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)
    place: Optional[str] = None
    affect: float = 0.0  # -1.0 to 1.0, negative = negative emotion, positive = positive
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'text': self.text,
            'embedding': self.embedding.tolist(),
            'timestamp': self.timestamp,
            'tags': self.tags,
            'people': self.people,
            'place': self.place,
            'affect': self.affect,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodicMemory':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            text=data['text'],
            embedding=np.array(data['embedding']),
            timestamp=data['timestamp'],
            tags=data.get('tags', []),
            people=data.get('people', []),
            place=data.get('place'),
            affect=data.get('affect', 0.0),
            metadata=data.get('metadata', {})
        )


class EpisodicMemoryStore:
    """
    Episodic memory store using FAISS for vector similarity search.
    
    This class manages episodic memories with semantic and temporal retrieval,
    supporting both content-based and time-based queries.
    """
    
    def __init__(self, embedding_dim: int = 1536, index_type: str = "flat"):
        """
        Initialize the episodic memory store.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            index_type: Type of FAISS index ("flat" or "ivf")
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Store memories and metadata
        self.memories: List[EpisodicMemory] = []
        self.id_to_index: Dict[str, int] = {}
        
        logger.info(f"Initialized EpisodicMemoryStore with {embedding_dim}D embeddings")
    
    def add_memory(self, text: str, embedding: np.ndarray, 
                   tags: Optional[List[str]] = None,
                   people: Optional[List[str]] = None,
                   place: Optional[str] = None,
                   affect: float = 0.0,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new episodic memory.
        
        Args:
            text: Text content of the memory
            embedding: Vector embedding of the text
            tags: Optional tags for the memory
            people: Optional list of people involved
            place: Optional place where it occurred
            affect: Emotional affect (-1.0 to 1.0)
            metadata: Optional additional metadata
            
        Returns:
            ID of the created memory
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension {embedding.shape[0]} doesn't match expected {self.embedding_dim}")
        
        # Create memory
        memory_id = str(uuid.uuid4())
        memory = EpisodicMemory(
            id=memory_id,
            text=text,
            embedding=embedding.astype(np.float32),
            timestamp=time.time(),
            tags=tags or [],
            people=people or [],
            place=place,
            affect=affect,
            metadata=metadata or {}
        )
        
        # Add to store
        self.memories.append(memory)
        self.id_to_index[memory_id] = len(self.memories) - 1
        
        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1).astype(np.float32))
        
        logger.debug(f"Added episodic memory: {memory_id}")
        return memory_id
    
    def search(self, query_embedding: np.ndarray, k: int = 8, 
               min_affect: Optional[float] = None,
               max_affect: Optional[float] = None,
               tags: Optional[List[str]] = None,
               people: Optional[List[str]] = None,
               time_range: Optional[Tuple[float, float]] = None) -> List[Tuple[EpisodicMemory, float]]:
        """
        Search for similar memories.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            min_affect: Minimum affect score filter
            max_affect: Maximum affect score filter
            tags: Filter by tags (any match)
            people: Filter by people (any match)
            time_range: Filter by time range (start, end) timestamps
            
        Returns:
            List of (memory, similarity_score) tuples
        """
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} doesn't match expected {self.embedding_dim}")
        
        if len(self.memories) == 0:
            return []
        
        # Search in FAISS index
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vector, min(k * 2, len(self.memories)))  # Get more to filter
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            memory = self.memories[idx]
            
            # Apply filters
            if min_affect is not None and memory.affect < min_affect:
                continue
            
            if max_affect is not None and memory.affect > max_affect:
                continue
            
            if tags and not any(tag in memory.tags for tag in tags):
                continue
            
            if people and not any(person in memory.people for person in people):
                continue
            
            if time_range:
                start_time, end_time = time_range
                if not (start_time <= memory.timestamp <= end_time):
                    continue
            
            results.append((memory, float(score)))
            
            if len(results) >= k:
                break
        
        logger.debug(f"Found {len(results)} memories matching query")
        return results
    
    def search_by_semantic_similarity(self, query_embedding: np.ndarray, k: int = 8) -> List[Tuple[EpisodicMemory, float]]:
        """Search by semantic similarity only."""
        return self.search(query_embedding, k)
    
    def search_by_recency(self, k: int = 8, hours: Optional[float] = None) -> List[EpisodicMemory]:
        """
        Search by recency.
        
        Args:
            k: Number of results to return
            hours: Optional time window in hours
            
        Returns:
            List of recent memories
        """
        if not self.memories:
            return []
        
        # Sort by timestamp (most recent first)
        sorted_memories = sorted(self.memories, key=lambda m: m.timestamp, reverse=True)
        
        if hours is not None:
            cutoff_time = time.time() - (hours * 3600)
            sorted_memories = [m for m in sorted_memories if m.timestamp >= cutoff_time]
        
        return sorted_memories[:k]
    
    def search_by_affect(self, k: int = 8, min_affect: float = 0.5) -> List[EpisodicMemory]:
        """
        Search for high-affect memories.
        
        Args:
            k: Number of results to return
            min_affect: Minimum affect score
            
        Returns:
            List of high-affect memories
        """
        high_affect_memories = [m for m in self.memories if m.affect >= min_affect]
        return sorted(high_affect_memories, key=lambda m: m.affect, reverse=True)[:k]
    
    def search_by_people(self, people: List[str], k: int = 8) -> List[EpisodicMemory]:
        """
        Search for memories involving specific people.
        
        Args:
            people: List of people to search for
            k: Number of results to return
            
        Returns:
            List of memories involving the people
        """
        matching_memories = []
        for memory in self.memories:
            if any(person in memory.people for person in people):
                matching_memories.append(memory)
        
        # Sort by recency
        return sorted(matching_memories, key=lambda m: m.timestamp, reverse=True)[:k]
    
    def get_memory_by_id(self, memory_id: str) -> Optional[EpisodicMemory]:
        """Get a specific memory by ID."""
        if memory_id in self.id_to_index:
            return self.memories[self.id_to_index[memory_id]]
        return None
    
    def update_memory(self, memory_id: str, **kwargs) -> bool:
        """
        Update a memory's metadata.
        
        Args:
            memory_id: ID of the memory to update
            **kwargs: Fields to update
            
        Returns:
            True if updated successfully, False if memory not found
        """
        if memory_id not in self.id_to_index:
            return False
        
        memory = self.memories[self.id_to_index[memory_id]]
        
        for key, value in kwargs.items():
            if hasattr(memory, key):
                setattr(memory, key, value)
        
        logger.debug(f"Updated memory: {memory_id}")
        return True
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deleted successfully, False if memory not found
        """
        if memory_id not in self.id_to_index:
            return False
        
        # Remove from memory list
        idx = self.id_to_index[memory_id]
        del self.memories[idx]
        
        # Update indices
        del self.id_to_index[memory_id]
        for mid, i in self.id_to_index.items():
            if i > idx:
                self.id_to_index[mid] = i - 1
        
        # Rebuild FAISS index
        self._rebuild_index()
        
        logger.debug(f"Deleted memory: {memory_id}")
        return True
    
    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from current memories."""
        if not self.memories:
            return
        
        # Reinitialize index
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        
        # Add all memories
        embeddings = np.array([m.embedding for m in self.memories])
        self.index.add(embeddings)
        
        logger.debug("Rebuilt FAISS index")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        if not self.memories:
            return {
                'num_memories': 0,
                'avg_affect': 0.0,
                'time_span_hours': 0.0,
                'unique_people': 0,
                'unique_places': 0
            }
        
        timestamps = [m.timestamp for m in self.memories]
        affects = [m.affect for m in self.memories]
        all_people = set()
        all_places = set()
        
        for memory in self.memories:
            all_people.update(memory.people)
            if memory.place:
                all_places.add(memory.place)
        
        return {
            'num_memories': len(self.memories),
            'avg_affect': np.mean(affects),
            'time_span_hours': (max(timestamps) - min(timestamps)) / 3600 if timestamps else 0,
            'unique_people': len(all_people),
            'unique_places': len(all_places)
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export memories to dictionary format."""
        return {
            'memories': [m.to_dict() for m in self.memories],
            'statistics': self.get_statistics()
        }
    
    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """Import memories from dictionary format."""
        self.clear()
        
        for memory_data in data.get('memories', []):
            memory = EpisodicMemory.from_dict(memory_data)
            self.memories.append(memory)
            self.id_to_index[memory.id] = len(self.memories) - 1
        
        # Rebuild index
        self._rebuild_index()
        
        logger.info(f"Imported {len(data.get('memories', []))} memories")
    
    def clear(self) -> None:
        """Clear all memories."""
        self.memories.clear()
        self.id_to_index.clear()
        
        # Reinitialize index
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        
        logger.info("Cleared episodic memory store")
    
    def save(self, filepath: str) -> None:
        """Save memories to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.export_to_dict(), f, indent=2)
        
        logger.info(f"Saved episodic memories to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load memories from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.import_from_dict(data)
        logger.info(f"Loaded episodic memories from {filepath}")
    
    def __len__(self) -> int:
        """Return number of memories."""
        return len(self.memories)
    
    def __str__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return f"EpisodicMemoryStore(memories={stats['num_memories']}, people={stats['unique_people']})"


# Import time module for timestamps
import time
