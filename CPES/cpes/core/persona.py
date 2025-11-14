"""
Persona specification system for CPES.

This module handles the core persona definition, loading, and validation.
The persona spec is the single source of identity that controls voice and values.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Motive:
    """Represents a core motive with ranking."""
    description: str
    rank: float  # 0.0 to 1.0, higher = more important


@dataclass
class Relationship:
    """Represents a relationship with another entity."""
    who: str
    valence: str  # e.g., "trust", "distrust", "neutral", "love", "hate"
    strength: float = 0.5  # 0.0 to 1.0


@dataclass
class StyleSpec:
    """Defines the persona's communication style."""
    syntax: str  # e.g., "crisply technical with sardonic asides"
    cadence: str  # e.g., "short, declarative sentences; rare metaphors"
    tics: List[str] = field(default_factory=list)  # Characteristic phrases


@dataclass
class PersonaSpec:
    """Complete persona specification."""
    name: str
    motives: List[Motive]
    virtues: List[str]
    vices: List[str]
    red_lines: List[str]  # Things they never do
    style: StyleSpec
    relationships: List[Relationship]
    taboos: List[str]  # Things they avoid saying/doing
    metadata: Dict[str, Any] = field(default_factory=dict)


class Persona:
    """
    Core persona management class for CPES.
    
    This class handles loading, validating, and managing persona specifications
    that serve as the single source of identity for behavioral emulation.
    """
    
    def __init__(self, persona_path: Union[str, Path, Dict[str, Any]]):
        """
        Initialize a persona from a file path or dictionary.
        
        Args:
            persona_path: Path to YAML file or dictionary containing persona spec
        """
        self.spec: Optional[PersonaSpec] = None
        self._load_persona(persona_path)
        self._validate_persona()
    
    def _load_persona(self, persona_path: Union[str, Path, Dict[str, Any]]) -> None:
        """Load persona specification from file or dictionary."""
        if isinstance(persona_path, dict):
            data = persona_path
        else:
            persona_path = Path(persona_path)
            if not persona_path.exists():
                raise FileNotFoundError(f"Persona file not found: {persona_path}")
            
            with open(persona_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        
        # Parse motives
        motives = []
        for motive_data in data.get('motives', []):
            if isinstance(motive_data, dict):
                motives.append(Motive(
                    description=motive_data['description'],
                    rank=motive_data.get('rank', 0.5)
                ))
            else:
                # Handle simple string format: "description (rank: 0.9)"
                if '(' in motive_data and 'rank:' in motive_data:
                    desc, rank_part = motive_data.split('(', 1)
                    rank = float(rank_part.split('rank:')[1].split(')')[0].strip())
                    motives.append(Motive(description=desc.strip(), rank=rank))
                else:
                    motives.append(Motive(description=motive_data, rank=0.5))
        
        # Parse relationships
        relationships = []
        for rel_data in data.get('relationships', []):
            if isinstance(rel_data, dict):
                relationships.append(Relationship(
                    who=rel_data['who'],
                    valence=rel_data['valence'],
                    strength=rel_data.get('strength', 0.5)
                ))
            else:
                # Handle simple format: "who: valence"
                if ':' in rel_data:
                    who, valence = rel_data.split(':', 1)
                    relationships.append(Relationship(
                        who=who.strip(),
                        valence=valence.strip()
                    ))
        
        # Parse style
        style_data = data.get('style', {})
        style = StyleSpec(
            syntax=style_data.get('syntax', ''),
            cadence=style_data.get('cadence', ''),
            tics=style_data.get('tics', [])
        )
        
        # Create persona spec
        self.spec = PersonaSpec(
            name=data['name'],
            motives=motives,
            virtues=data.get('virtues', []),
            vices=data.get('vices', []),
            red_lines=data.get('red_lines', []),
            style=style,
            relationships=relationships,
            taboos=data.get('taboos', []),
            metadata=data.get('metadata', {})
        )
        
        logger.info(f"Loaded persona: {self.spec.name}")
    
    def _validate_persona(self) -> None:
        """Validate persona specification for completeness and consistency."""
        if not self.spec:
            raise ValueError("Persona specification not loaded")
        
        # Check required fields
        if not self.spec.name:
            raise ValueError("Persona name is required")
        
        if not self.spec.motives:
            logger.warning("No motives defined for persona")
        
        if not self.spec.virtues and not self.spec.vices:
            logger.warning("No virtues or vices defined for persona")
        
        # Validate motive ranks
        for motive in self.spec.motives:
            if not 0.0 <= motive.rank <= 1.0:
                raise ValueError(f"Invalid motive rank {motive.rank} for '{motive.description}'. Must be 0.0-1.0")
        
        # Validate relationship strengths
        for rel in self.spec.relationships:
            if not 0.0 <= rel.strength <= 1.0:
                raise ValueError(f"Invalid relationship strength {rel.strength} for '{rel.who}'. Must be 0.0-1.0")
        
        logger.debug("Persona validation passed")
    
    def get_identity_context(self) -> str:
        """
        Get formatted identity context for LLM prompts.
        
        Returns:
            Formatted string containing persona identity information
        """
        if not self.spec:
            return ""
        
        context = f"Identity: {self.spec.name}\n\n"
        
        # Motives (sorted by rank)
        if self.spec.motives:
            context += "Core Motives:\n"
            sorted_motives = sorted(self.spec.motives, key=lambda m: m.rank, reverse=True)
            for motive in sorted_motives:
                context += f"- {motive.description} (priority: {motive.rank:.1f})\n"
            context += "\n"
        
        # Virtues and vices
        if self.spec.virtues:
            context += f"Virtues: {', '.join(self.spec.virtues)}\n"
        
        if self.spec.vices:
            context += f"Vices: {', '.join(self.spec.vices)}\n"
        
        # Red lines
        if self.spec.red_lines:
            context += "\nRed Lines (never do):\n"
            for line in self.spec.red_lines:
                context += f"- {line}\n"
        
        # Taboos
        if self.spec.taboos:
            context += "\nTaboos (avoid):\n"
            for taboo in self.spec.taboos:
                context += f"- {taboo}\n"
        
        # Style
        if self.spec.style.syntax:
            context += f"\nCommunication Style: {self.spec.style.syntax}\n"
        
        if self.spec.style.cadence:
            context += f"Cadence: {self.spec.style.cadence}\n"
        
        if self.spec.style.tics:
            context += f"Characteristic phrases: {', '.join(self.spec.style.tics)}\n"
        
        # Relationships
        if self.spec.relationships:
            context += "\nKey Relationships:\n"
            for rel in self.spec.relationships:
                context += f"- {rel.who}: {rel.valence} (strength: {rel.strength:.1f})\n"
        
        return context
    
    def get_motives_by_rank(self, min_rank: float = 0.0) -> List[Motive]:
        """Get motives above a minimum rank threshold."""
        return [m for m in self.spec.motives if m.rank >= min_rank]
    
    def get_relationships_by_valence(self, valence: str) -> List[Relationship]:
        """Get relationships with specific valence."""
        return [r for r in self.spec.relationships if r.valence == valence]
    
    def check_red_line_violation(self, text: str) -> List[str]:
        """
        Check if text violates any red lines.
        
        Args:
            text: Text to check
            
        Returns:
            List of violated red lines
        """
        violations = []
        text_lower = text.lower()
        
        for red_line in self.spec.red_lines:
            # Simple keyword matching - could be enhanced with more sophisticated NLP
            if any(word in text_lower for word in red_line.lower().split()):
                violations.append(red_line)
        
        return violations
    
    def check_taboo_violation(self, text: str) -> List[str]:
        """
        Check if text contains any taboo content.
        
        Args:
            text: Text to check
            
        Returns:
            List of violated taboos
        """
        violations = []
        text_lower = text.lower()
        
        for taboo in self.spec.taboos:
            if any(word in text_lower for word in taboo.lower().split()):
                violations.append(taboo)
        
        return violations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary format."""
        if not self.spec:
            return {}
        
        return {
            'name': self.spec.name,
            'motives': [
                {'description': m.description, 'rank': m.rank} 
                for m in self.spec.motives
            ],
            'virtues': self.spec.virtues,
            'vices': self.spec.vices,
            'red_lines': self.spec.red_lines,
            'style': {
                'syntax': self.spec.style.syntax,
                'cadence': self.spec.style.cadence,
                'tics': self.spec.style.tics
            },
            'relationships': [
                {'who': r.who, 'valence': r.valence, 'strength': r.strength}
                for r in self.spec.relationships
            ],
            'taboos': self.spec.taboos,
            'metadata': self.spec.metadata
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save persona to YAML file."""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved persona to {path}")
    
    def __str__(self) -> str:
        """String representation of persona."""
        if not self.spec:
            return "Uninitialized Persona"
        
        return f"Persona({self.spec.name})"
    
    def __repr__(self) -> str:
        """Detailed representation of persona."""
        if not self.spec:
            return "Persona(spec=None)"
        
        return f"Persona(name='{self.spec.name}', motives={len(self.spec.motives)}, virtues={len(self.spec.virtues)})"
