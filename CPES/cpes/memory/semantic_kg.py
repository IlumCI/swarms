"""
Semantic Knowledge Graph for CPES.

This module implements a knowledge graph for storing and querying semantic relationships
and beliefs about the persona. Used for factual grounding and consistent beliefs.
"""

import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from loguru import logger
import json
import time


@dataclass
class Triple:
    """Represents a semantic triple (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SemanticKnowledgeGraph:
    """
    Semantic Knowledge Graph for storing and querying persona beliefs and facts.
    
    This class maintains a graph of semantic relationships that represent the persona's
    knowledge, beliefs, and understanding of the world. It provides methods for
    adding, querying, and reasoning about these relationships.
    """
    
    def __init__(self, directed: bool = True):
        """
        Initialize the semantic knowledge graph.
        
        Args:
            directed: Whether to use a directed graph (default: True)
        """
        self.graph = nx.MultiDiGraph() if directed else nx.MultiGraph()
        self.triples: List[Triple] = []
        self._index: Dict[str, Set[int]] = {}  # Entity -> triple indices
        
        logger.info("Initialized Semantic Knowledge Graph")
    
    def add_triple(self, subject: str, predicate: str, object: str, 
                   confidence: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a semantic triple to the knowledge graph.
        
        Args:
            subject: Subject of the triple
            predicate: Predicate/relationship
            object: Object of the triple
            confidence: Confidence score (0.0 to 1.0)
            metadata: Additional metadata for the triple
            
        Returns:
            Index of the added triple
        """
        if metadata is None:
            metadata = {}
        
        # Create triple
        triple = Triple(
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=confidence,
            metadata=metadata
        )
        
        # Add to list
        triple_index = len(self.triples)
        self.triples.append(triple)
        
        # Add to graph
        self.graph.add_edge(subject, object, key=predicate, 
                          confidence=confidence, metadata=metadata)
        
        # Update index
        for entity in [subject, object]:
            if entity not in self._index:
                self._index[entity] = set()
            self._index[entity].add(triple_index)
        
        logger.debug(f"Added triple: {subject} -> {predicate} -> {object}")
        return triple_index
    
    def add_triples_batch(self, triples: List[Tuple[str, str, str, float, Dict[str, Any]]]) -> List[int]:
        """
        Add multiple triples in batch.
        
        Args:
            triples: List of (subject, predicate, object, confidence, metadata) tuples
            
        Returns:
            List of triple indices
        """
        indices = []
        for subject, predicate, object, confidence, metadata in triples:
            idx = self.add_triple(subject, predicate, object, confidence, metadata)
            indices.append(idx)
        
        logger.info(f"Added {len(triples)} triples in batch")
        return indices
    
    def query_triples(self, subject: Optional[str] = None, 
                     predicate: Optional[str] = None,
                     object: Optional[str] = None,
                     min_confidence: float = 0.0) -> List[Triple]:
        """
        Query triples matching the given pattern.
        
        Args:
            subject: Subject to match (None for any)
            predicate: Predicate to match (None for any)
            object: Object to match (None for any)
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matching triples
        """
        matches = []
        
        for triple in self.triples:
            if triple.confidence < min_confidence:
                continue
            
            if subject is not None and triple.subject != subject:
                continue
            
            if predicate is not None and triple.predicate != predicate:
                continue
            
            if object is not None and triple.object != object:
                continue
            
            matches.append(triple)
        
        logger.debug(f"Found {len(matches)} matching triples")
        return matches
    
    def get_entity_relationships(self, entity: str, 
                               relationship_types: Optional[List[str]] = None) -> List[Triple]:
        """
        Get all relationships for a specific entity.
        
        Args:
            entity: Entity to get relationships for
            relationship_types: Optional list of relationship types to filter by
            
        Returns:
            List of triples involving the entity
        """
        if entity not in self._index:
            return []
        
        entity_triples = [self.triples[i] for i in self._index[entity]]
        
        if relationship_types:
            entity_triples = [t for t in entity_triples if t.predicate in relationship_types]
        
        return entity_triples
    
    def get_beliefs_about(self, topic: str, max_results: int = 10) -> List[str]:
        """
        Get beliefs about a specific topic.
        
        Args:
            topic: Topic to get beliefs about
            max_results: Maximum number of results to return
            
        Returns:
            List of belief statements
        """
        beliefs = []
        
        # Get direct relationships
        relationships = self.get_entity_relationships(topic)
        
        for rel in relationships[:max_results]:
            if rel.subject == topic:
                beliefs.append(f"{rel.subject} {rel.predicate} {rel.object}")
            else:
                beliefs.append(f"{rel.object} {rel.predicate} {rel.subject}")
        
        # Get indirect relationships (2-hop)
        if len(beliefs) < max_results:
            for rel in relationships:
                if rel.subject == topic:
                    # Follow outgoing relationships
                    next_rels = self.get_entity_relationships(rel.object)
                    for next_rel in next_rels[:max_results - len(beliefs)]:
                        beliefs.append(f"{topic} -> {rel.object} -> {next_rel.object}")
                        if len(beliefs) >= max_results:
                            break
                elif rel.object == topic:
                    # Follow incoming relationships
                    next_rels = self.get_entity_relationships(rel.subject)
                    for next_rel in next_rels[:max_results - len(beliefs)]:
                        beliefs.append(f"{next_rel.subject} -> {rel.subject} -> {topic}")
                        if len(beliefs) >= max_results:
                            break
        
        logger.debug(f"Found {len(beliefs)} beliefs about {topic}")
        return beliefs[:max_results]
    
    def find_path(self, start: str, end: str, max_length: int = 3) -> Optional[List[str]]:
        """
        Find a path between two entities.
        
        Args:
            start: Starting entity
            end: Ending entity
            max_length: Maximum path length
            
        Returns:
            List of entities in the path, or None if no path found
        """
        try:
            path = nx.shortest_path(self.graph, start, end)
            if len(path) <= max_length + 1:  # +1 because path includes both endpoints
                return path
        except nx.NetworkXNoPath:
            pass
        
        return None
    
    def get_related_entities(self, entity: str, max_degree: int = 2) -> Set[str]:
        """
        Get entities related to the given entity within max_degree hops.
        
        Args:
            entity: Entity to find related entities for
            max_degree: Maximum degree of separation
            
        Returns:
            Set of related entity names
        """
        if entity not in self.graph:
            return set()
        
        related = set()
        current_level = {entity}
        
        for degree in range(max_degree):
            next_level = set()
            for node in current_level:
                # Add neighbors
                neighbors = set(self.graph.neighbors(node))
                next_level.update(neighbors)
            
            related.update(next_level)
            current_level = next_level - related  # Avoid revisiting
            
            if not current_level:
                break
        
        related.discard(entity)  # Remove the original entity
        return related
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        return {
            'num_triples': len(self.triples),
            'num_entities': len(self.graph.nodes),
            'num_relationships': len(self.graph.edges),
            'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes) if self.graph.nodes else 0,
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph) if self.graph.is_directed() else nx.is_connected(self.graph)
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export knowledge graph to dictionary format."""
        return {
            'triples': [
                {
                    'subject': t.subject,
                    'predicate': t.predicate,
                    'object': t.object,
                    'confidence': t.confidence,
                    'metadata': t.metadata
                }
                for t in self.triples
            ],
            'statistics': self.get_statistics()
        }
    
    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """Import knowledge graph from dictionary format."""
        self.clear()
        
        for triple_data in data.get('triples', []):
            self.add_triple(
                subject=triple_data['subject'],
                predicate=triple_data['predicate'],
                object=triple_data['object'],
                confidence=triple_data.get('confidence', 1.0),
                metadata=triple_data.get('metadata', {})
            )
        
        logger.info(f"Imported {len(data.get('triples', []))} triples")
    
    def clear(self) -> None:
        """Clear all data from the knowledge graph."""
        self.graph.clear()
        self.triples.clear()
        self._index.clear()
        logger.info("Cleared knowledge graph")
    
    def save(self, filepath: str) -> None:
        """Save knowledge graph to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.export_to_dict(), f, indent=2)
        
        logger.info(f"Saved knowledge graph to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load knowledge graph from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.import_from_dict(data)
        logger.info(f"Loaded knowledge graph from {filepath}")
    
    def __len__(self) -> int:
        """Return number of triples in the graph."""
        return len(self.triples)
    
    def __str__(self) -> str:
        """String representation of the knowledge graph."""
        stats = self.get_statistics()
        return f"SemanticKnowledgeGraph(triples={stats['num_triples']}, entities={stats['num_entities']})"
