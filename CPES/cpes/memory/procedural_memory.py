"""
Procedural Memory for CPES.

This module implements procedural memory for storing and retrieving skills,
workflows, and tool usage patterns specific to the persona.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
from loguru import logger


@dataclass
class Skill:
    """Represents a procedural skill or workflow."""
    id: str
    name: str
    description: str
    steps: List[str]
    tools: List[str] = field(default_factory=list)
    context: str = ""
    success_rate: float = 0.0  # 0.0 to 1.0
    last_used: Optional[float] = None
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'steps': self.steps,
            'tools': self.tools,
            'context': self.context,
            'success_rate': self.success_rate,
            'last_used': self.last_used,
            'usage_count': self.usage_count,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Skill':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            steps=data['steps'],
            tools=data.get('tools', []),
            context=data.get('context', ''),
            success_rate=data.get('success_rate', 0.0),
            last_used=data.get('last_used'),
            usage_count=data.get('usage_count', 0),
            metadata=data.get('metadata', {})
        )


class ProceduralMemoryStore:
    """
    Procedural memory store for managing skills and workflows.
    
    This class stores and retrieves procedural knowledge about how the persona
    performs specific tasks, uses tools, and follows workflows.
    """
    
    def __init__(self):
        """Initialize the procedural memory store."""
        self.skills: Dict[str, Skill] = {}
        self.skill_categories: Dict[str, List[str]] = {}  # category -> skill_ids
        self.tool_skills: Dict[str, List[str]] = {}  # tool -> skill_ids
        
        logger.info("Initialized ProceduralMemoryStore")
    
    def add_skill(self, name: str, description: str, steps: List[str],
                  tools: Optional[List[str]] = None,
                  context: str = "",
                  category: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new skill to procedural memory.
        
        Args:
            name: Name of the skill
            description: Description of what the skill does
            steps: List of steps to perform the skill
            tools: Optional list of tools used
            context: Context where this skill is used
            category: Optional category for the skill
            metadata: Optional additional metadata
            
        Returns:
            ID of the created skill
        """
        skill_id = str(uuid.uuid4())
        skill = Skill(
            id=skill_id,
            name=name,
            description=description,
            steps=steps,
            tools=tools or [],
            context=context,
            metadata=metadata or {}
        )
        
        self.skills[skill_id] = skill
        
        # Add to categories
        if category:
            if category not in self.skill_categories:
                self.skill_categories[category] = []
            self.skill_categories[category].append(skill_id)
        
        # Add to tool mapping
        for tool in skill.tools:
            if tool not in self.tool_skills:
                self.tool_skills[tool] = []
            self.tool_skills[tool].append(skill_id)
        
        logger.debug(f"Added skill: {name}")
        return skill_id
    
    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        return self.skills.get(skill_id)
    
    def find_skills_by_name(self, name: str) -> List[Skill]:
        """Find skills by name (partial match)."""
        name_lower = name.lower()
        return [skill for skill in self.skills.values() 
                if name_lower in skill.name.lower()]
    
    def find_skills_by_tool(self, tool: str) -> List[Skill]:
        """Find skills that use a specific tool."""
        skill_ids = self.tool_skills.get(tool, [])
        return [self.skills[sid] for sid in skill_ids if sid in self.skills]
    
    def find_skills_by_category(self, category: str) -> List[Skill]:
        """Find skills in a specific category."""
        skill_ids = self.skill_categories.get(category, [])
        return [self.skills[sid] for sid in skill_ids if sid in self.skills]
    
    def search_skills(self, query: str, limit: int = 10) -> List[Skill]:
        """
        Search skills by name, description, or context.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching skills
        """
        query_lower = query.lower()
        matches = []
        
        for skill in self.skills.values():
            score = 0
            
            # Check name
            if query_lower in skill.name.lower():
                score += 3
            
            # Check description
            if query_lower in skill.description.lower():
                score += 2
            
            # Check context
            if query_lower in skill.context.lower():
                score += 1
            
            # Check steps
            for step in skill.steps:
                if query_lower in step.lower():
                    score += 1
            
            if score > 0:
                matches.append((skill, score))
        
        # Sort by score and return top results
        matches.sort(key=lambda x: x[1], reverse=True)
        return [skill for skill, _ in matches[:limit]]
    
    def get_workflow_for_task(self, task: str, context: str = "") -> Optional[Skill]:
        """
        Get the best workflow for a specific task.
        
        Args:
            task: Description of the task
            context: Optional context for the task
            
        Returns:
            Best matching skill, or None if no match found
        """
        # Search for skills that match the task
        candidates = self.search_skills(task, limit=5)
        
        if not candidates:
            return None
        
        # Score candidates based on relevance and success rate
        best_skill = None
        best_score = 0
        
        for skill in candidates:
            score = skill.success_rate * 0.7  # Weight success rate heavily
            
            # Bonus for context match
            if context and context.lower() in skill.context.lower():
                score += 0.3
            
            # Bonus for recent usage
            if skill.last_used:
                # Simple recency bonus (could be more sophisticated)
                score += 0.1
            
            if score > best_score:
                best_score = score
                best_skill = skill
        
        return best_skill
    
    def record_skill_usage(self, skill_id: str, success: bool = True) -> bool:
        """
        Record usage of a skill and update success rate.
        
        Args:
            skill_id: ID of the skill used
            success: Whether the skill usage was successful
            
        Returns:
            True if recorded successfully, False if skill not found
        """
        if skill_id not in self.skills:
            return False
        
        skill = self.skills[skill_id]
        skill.usage_count += 1
        skill.last_used = time.time()
        
        # Update success rate (exponential moving average)
        alpha = 0.1  # Learning rate
        if success:
            skill.success_rate = skill.success_rate + alpha * (1.0 - skill.success_rate)
        else:
            skill.success_rate = skill.success_rate + alpha * (0.0 - skill.success_rate)
        
        logger.debug(f"Recorded skill usage: {skill.name} (success: {success})")
        return True
    
    def update_skill(self, skill_id: str, **kwargs) -> bool:
        """
        Update a skill's properties.
        
        Args:
            skill_id: ID of the skill to update
            **kwargs: Fields to update
            
        Returns:
            True if updated successfully, False if skill not found
        """
        if skill_id not in self.skills:
            return False
        
        skill = self.skills[skill_id]
        
        # Update allowed fields
        allowed_fields = ['name', 'description', 'steps', 'tools', 'context', 'metadata']
        for key, value in kwargs.items():
            if key in allowed_fields and hasattr(skill, key):
                setattr(skill, key, value)
        
        logger.debug(f"Updated skill: {skill.name}")
        return True
    
    def delete_skill(self, skill_id: str) -> bool:
        """
        Delete a skill by ID.
        
        Args:
            skill_id: ID of the skill to delete
            
        Returns:
            True if deleted successfully, False if skill not found
        """
        if skill_id not in self.skills:
            return False
        
        skill = self.skills[skill_id]
        
        # Remove from categories
        for category, skill_ids in self.skill_categories.items():
            if skill_id in skill_ids:
                skill_ids.remove(skill_id)
        
        # Remove from tool mapping
        for tool, skill_ids in self.tool_skills.items():
            if skill_id in skill_ids:
                skill_ids.remove(skill_id)
        
        # Remove from skills
        del self.skills[skill_id]
        
        logger.debug(f"Deleted skill: {skill.name}")
        return True
    
    def get_skill_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored skills."""
        if not self.skills:
            return {
                'num_skills': 0,
                'avg_success_rate': 0.0,
                'total_usage': 0,
                'categories': 0,
                'tools': 0
            }
        
        success_rates = [skill.success_rate for skill in self.skills.values()]
        total_usage = sum(skill.usage_count for skill in self.skills.values())
        all_tools = set()
        
        for skill in self.skills.values():
            all_tools.update(skill.tools)
        
        return {
            'num_skills': len(self.skills),
            'avg_success_rate': sum(success_rates) / len(success_rates),
            'total_usage': total_usage,
            'categories': len(self.skill_categories),
            'tools': len(all_tools)
        }
    
    def get_skills_by_usage(self, limit: int = 10) -> List[Skill]:
        """Get most frequently used skills."""
        skills = list(self.skills.values())
        skills.sort(key=lambda s: s.usage_count, reverse=True)
        return skills[:limit]
    
    def get_skills_by_success_rate(self, min_rate: float = 0.0, limit: int = 10) -> List[Skill]:
        """Get skills with high success rates."""
        skills = [s for s in self.skills.values() if s.success_rate >= min_rate]
        skills.sort(key=lambda s: s.success_rate, reverse=True)
        return skills[:limit]
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export skills to dictionary format."""
        return {
            'skills': {sid: skill.to_dict() for sid, skill in self.skills.items()},
            'categories': self.skill_categories,
            'tool_skills': self.tool_skills,
            'statistics': self.get_skill_statistics()
        }
    
    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """Import skills from dictionary format."""
        self.clear()
        
        # Import skills
        for skill_id, skill_data in data.get('skills', {}).items():
            skill = Skill.from_dict(skill_data)
            self.skills[skill_id] = skill
        
        # Import categories and tool mappings
        self.skill_categories = data.get('categories', {})
        self.tool_skills = data.get('tool_skills', {})
        
        logger.info(f"Imported {len(self.skills)} skills")
    
    def clear(self) -> None:
        """Clear all skills."""
        self.skills.clear()
        self.skill_categories.clear()
        self.tool_skills.clear()
        logger.info("Cleared procedural memory store")
    
    def save(self, filepath: str) -> None:
        """Save skills to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.export_to_dict(), f, indent=2)
        
        logger.info(f"Saved procedural memories to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load skills from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.import_from_dict(data)
        logger.info(f"Loaded procedural memories from {filepath}")
    
    def __len__(self) -> int:
        """Return number of skills."""
        return len(self.skills)
    
    def __str__(self) -> str:
        """String representation."""
        stats = self.get_skill_statistics()
        return f"ProceduralMemoryStore(skills={stats['num_skills']}, tools={stats['tools']})"


# Import time module for timestamps
import time
