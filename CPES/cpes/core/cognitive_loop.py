"""
Cognitive Loop for CPES.

This module implements the core cognitive loop that orchestrates the persona's
decision-making process: observe, recall, deliberate, act, reflect, adjust.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
from loguru import logger

from .persona import Persona
from .react import ReActAgent
from ..memory.semantic_kg import SemanticKnowledgeGraph
from ..memory.episodic_memory import EpisodicMemoryStore
from ..memory.procedural_memory import ProceduralMemoryStore
from ..controllers.value_gate import ValueGate
from ..controllers.style_adapter import StyleAdapter
from ..utils.embedding import EmbeddingModel
from ..utils.llm_wrapper import LLMWrapper
from ..tools.basic_tools import BasicTools
from ..tools.search_tools import SearchTools
from ..tools.memory_tools import MemoryTools


@dataclass
class CognitiveState:
    """Represents the current state of the cognitive loop."""
    input_text: str
    extracted_intents: List[str] = field(default_factory=list)
    extracted_entities: List[str] = field(default_factory=list)
    recalled_memories: List[Any] = field(default_factory=list)
    relevant_beliefs: List[str] = field(default_factory=list)
    draft_response: str = ""
    final_response: str = ""
    salience_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    react_result: Optional[Any] = None  # ReAct reasoning result
    use_react: bool = False  # Whether to use ReAct for this input


class CognitiveLoop:
    """
    Core cognitive loop for CPES persona emulation.
    
    This class orchestrates the complete cognitive process:
    1. Observe: Extract intents and entities from input
    2. Recall: Retrieve relevant memories and beliefs
    3. Deliberate: Plan response under persona constraints
    4. Act: Generate response with style adaptation
    5. Reflect: Evaluate response and store if salient
    6. Adjust: Apply drift correction if needed
    """
    
    def __init__(self, persona: Persona,
                 semantic_kg: SemanticKnowledgeGraph,
                 episodic_memory: EpisodicMemoryStore,
                 procedural_memory: ProceduralMemoryStore,
                 embedding_model: EmbeddingModel,
                 llm_wrapper: LLMWrapper,
                 value_gate: Optional[ValueGate] = None,
                 style_adapter: Optional[StyleAdapter] = None):
        """
        Initialize the cognitive loop.
        
        Args:
            persona: Persona specification
            semantic_kg: Semantic knowledge graph
            episodic_memory: Episodic memory store
            procedural_memory: Procedural memory store
            embedding_model: Embedding model for vector operations
            llm_wrapper: LLM wrapper for text generation
            value_gate: Optional value gate controller
            style_adapter: Optional style adapter controller
        """
        self.persona = persona
        self.semantic_kg = semantic_kg
        self.episodic_memory = episodic_memory
        self.procedural_memory = procedural_memory
        self.embedding_model = embedding_model
        self.llm_wrapper = llm_wrapper
        
        # Initialize controllers
        self.value_gate = value_gate or ValueGate(persona, llm_wrapper)
        self.style_adapter = style_adapter or StyleAdapter(persona, llm_wrapper)
        
        # Initialize ReAct agent
        self.react_agent = self._initialize_react_agent()
        
        # State tracking
        self.current_state: Optional[CognitiveState] = None
        self.conversation_history: List[CognitiveState] = []
        
        logger.info(f"Initialized CognitiveLoop for {persona.spec.name}")
    
    def _initialize_react_agent(self) -> ReActAgent:
        """Initialize the ReAct agent with available tools."""
        # Get persona context for reasoning style
        persona_context = self.persona.get_identity_context()
        
        # Initialize tool collections
        basic_tools = BasicTools.get_tools()
        search_tools = SearchTools().get_tools()
        memory_tools = MemoryTools(
            episodic_memory=self.episodic_memory,
            semantic_kg=self.semantic_kg,
            procedural_memory=self.procedural_memory
        ).get_tools()
        
        # Combine all tools
        all_tools = {**basic_tools, **search_tools, **memory_tools}
        
        # Create ReAct agent
        react_agent = ReActAgent(
            llm_wrapper=self.llm_wrapper,
            tools=all_tools,
            max_steps=6,  # Reasonable limit for persona responses
            persona_context=persona_context
        )
        
        logger.info(f"Initialized ReAct agent with {len(all_tools)} tools")
        return react_agent
    
    def _should_use_react(self, user_input: str, intents: List[str]) -> bool:
        """Determine if ReAct reasoning should be used for this input."""
        # Use ReAct for complex reasoning tasks
        react_keywords = [
            "calculate", "compute", "solve", "analyze", "compare",
            "find", "search", "look up", "research", "investigate",
            "plan", "design", "create", "build", "develop"
        ]
        
        input_lower = user_input.lower()
        
        # Check for reasoning keywords
        if any(keyword in input_lower for keyword in react_keywords):
            return True
        
        # Check for question patterns that might need reasoning
        if any(intent in intents for intent in ["question", "information_seeking"]):
            if len(user_input.split()) > 10:  # Complex questions
                return True
        
        # Check for mathematical expressions
        import re
        if re.search(r'\d+[\+\-\*/]\d+', user_input):
            return True
        
        return False
    
    def process_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process user input through the complete cognitive loop.
        
        Args:
            user_input: User's input text
            context: Optional context information
            
        Returns:
            Persona's response
        """
        logger.info(f"Processing input: {user_input[:100]}...")
        
        # Initialize cognitive state
        self.current_state = CognitiveState(input_text=user_input)
        
        try:
            # 1. Observe: Extract intents and entities
            self._observe(user_input, context)
            
            # Check if we should use ReAct reasoning
            self.current_state.use_react = self._should_use_react(
                user_input, 
                self.current_state.extracted_intents
            )
            
            if self.current_state.use_react:
                # Use ReAct reasoning
                self._react_reasoning()
            else:
                # Use standard cognitive loop
                # 2. Recall: Retrieve relevant memories and beliefs
                self._recall()
                
                # 3. Deliberate: Plan response under persona constraints
                self._deliberate()
            
            # 4. Act: Generate response with style adaptation
            self._act()
            
            # 5. Reflect: Evaluate response and store if salient
            self._reflect()
            
            # 6. Adjust: Apply drift correction if needed
            self._adjust()
            
            # Store state in conversation history
            self.conversation_history.append(self.current_state)
            
            logger.info("Cognitive loop completed successfully")
            return self.current_state.final_response
            
        except Exception as e:
            logger.error(f"Cognitive loop failed: {e}")
            return "I apologize, but I encountered an error processing your request."
    
    def _observe(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Extract intents and entities from user input."""
        logger.debug("Observing input...")
        
        # Simple intent extraction (could be enhanced with NLP)
        intents = self._extract_intents(user_input)
        entities = self._extract_entities(user_input)
        
        self.current_state.extracted_intents = intents
        self.current_state.extracted_entities = entities
        
        logger.debug(f"Extracted intents: {intents}, entities: {entities}")
    
    def _extract_intents(self, text: str) -> List[str]:
        """Extract intents from text (simple keyword-based)."""
        text_lower = text.lower()
        intents = []
        
        # Question patterns
        if any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            intents.append('question')
        
        # Request patterns
        if any(word in text_lower for word in ['can you', 'please', 'help', 'do']):
            intents.append('request')
        
        # Opinion patterns
        if any(word in text_lower for word in ['think', 'believe', 'opinion', 'feel']):
            intents.append('opinion_seeking')
        
        # Information patterns
        if any(word in text_lower for word in ['tell me', 'explain', 'describe', 'show']):
            intents.append('information_seeking')
        
        return intents
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text (simple pattern matching)."""
        import re
        
        entities = []
        
        # Proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(proper_nouns)
        
        # Technical terms (CamelCase)
        technical_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', text)
        entities.extend(technical_terms)
        
        # Numbers
        numbers = re.findall(r'\b\d+\b', text)
        entities.extend(numbers)
        
        return list(set(entities))  # Remove duplicates
    
    def _react_reasoning(self) -> None:
        """Perform ReAct reasoning for complex inputs."""
        logger.debug("Performing ReAct reasoning...")
        
        try:
            # Use ReAct agent to reason about the input
            react_result = self.react_agent.reason(self.current_state.input_text)
            self.current_state.react_result = react_result
            
            # Use the ReAct result as the draft response
            if react_result.success:
                self.current_state.draft_response = react_result.final_answer
                logger.debug(f"ReAct reasoning successful: {react_result.total_steps} steps")
            else:
                # Fallback to standard reasoning if ReAct fails
                logger.warning("ReAct reasoning failed, falling back to standard reasoning")
                self._recall()
                self._deliberate()
                
        except Exception as e:
            logger.error(f"ReAct reasoning failed: {e}")
            # Fallback to standard reasoning
            self._recall()
            self._deliberate()
    
    def _recall(self) -> None:
        """Recall relevant memories and beliefs."""
        logger.debug("Recalling relevant information...")
        
        # Get episodic memories
        query_embedding = self.embedding_model.encode(self.current_state.input_text)
        memories = self.episodic_memory.search(
            query_embedding=query_embedding,
            k=6  # Retrieve 6 most relevant memories
        )
        
        self.current_state.recalled_memories = memories
        
        # Get relevant beliefs from semantic KG
        beliefs = []
        for entity in self.current_state.extracted_entities:
            entity_beliefs = self.semantic_kg.get_beliefs_about(entity, max_results=3)
            beliefs.extend(entity_beliefs)
        
        self.current_state.relevant_beliefs = beliefs
        
        logger.debug(f"Recalled {len(memories)} memories and {len(beliefs)} beliefs")
    
    def _deliberate(self) -> None:
        """Plan response under persona constraints."""
        logger.debug("Deliberating response...")
        
        # Build context for response generation
        context_parts = []
        
        # Add persona identity
        context_parts.append(self.persona.get_identity_context())
        
        # Add relevant memories
        if self.current_state.recalled_memories:
            context_parts.append("\nRelevant Memories:")
            for i, (memory, score) in enumerate(self.current_state.recalled_memories[:3], 1):
                context_parts.append(f"{i}. {memory.text[:200]}... (relevance: {score:.2f})")
        
        # Add relevant beliefs
        if self.current_state.relevant_beliefs:
            context_parts.append("\nRelevant Knowledge:")
            for belief in self.current_state.relevant_beliefs[:3]:
                context_parts.append(f"- {belief}")
        
        # Add current context
        context_parts.append(f"\nCurrent Input: {self.current_state.input_text}")
        
        context_text = "\n".join(context_parts)
        
        # Generate draft response
        try:
            response = self.llm_wrapper.generate(
                prompt=context_text,
                temperature=0.7,
                max_tokens=500
            )
            
            self.current_state.draft_response = response.content
            logger.debug("Generated draft response")
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            self.current_state.draft_response = "I need to think about this more carefully."
    
    def _act(self) -> None:
        """Generate final response with style adaptation."""
        logger.debug("Acting - generating final response...")
        
        # Apply value gate
        violations = self.value_gate.check_text(
            self.current_state.draft_response,
            context=self.current_state.input_text
        )
        
        if violations:
            logger.warning(f"Found {len(violations)} value violations, enforcing values...")
            gated_response = self.value_gate.enforce_values(
                self.current_state.draft_response,
                context=self.current_state.input_text
            )
        else:
            gated_response = self.current_state.draft_response
        
        # Apply style adapter
        final_response = self.style_adapter.adapt_text(
            gated_response,
            context=self.current_state.input_text
        )
        
        self.current_state.final_response = final_response
        logger.debug("Generated final response")
    
    def _reflect(self) -> None:
        """Evaluate response and store if salient."""
        logger.debug("Reflecting on response...")
        
        # Calculate salience score
        salience_score = self._calculate_salience()
        self.current_state.salience_score = salience_score
        
        # Store as episodic memory if salient
        if salience_score > 0.5:  # Threshold for storing
            self._store_episodic_memory()
            logger.debug(f"Stored salient interaction (score: {salience_score:.2f})")
    
    def _calculate_salience(self) -> float:
        """Calculate salience score for the interaction."""
        score = 0.0
        
        # Base score from response length
        response_length = len(self.current_state.final_response.split())
        score += min(response_length / 100, 0.3)  # Cap at 0.3
        
        # Score from emotional content
        emotional_words = ['excited', 'frustrated', 'concerned', 'pleased', 'disappointed']
        response_lower = self.current_state.final_response.lower()
        emotional_count = sum(1 for word in emotional_words if word in response_lower)
        score += min(emotional_count * 0.1, 0.2)  # Cap at 0.2
        
        # Score from novelty (new entities)
        new_entities = set(self.current_state.extracted_entities)
        if hasattr(self, '_previous_entities'):
            new_entities -= self._previous_entities
        score += min(len(new_entities) * 0.1, 0.2)  # Cap at 0.2
        
        # Score from value violations (indicates important decisions)
        violations = self.value_gate.check_text(self.current_state.final_response)
        score += min(len(violations) * 0.1, 0.3)  # Cap at 0.3
        
        # Store entities for next comparison
        self._previous_entities = set(self.current_state.extracted_entities)
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _store_episodic_memory(self) -> None:
        """Store the interaction as episodic memory."""
        # Create memory text
        memory_text = f"User: {self.current_state.input_text}\n{self.persona.spec.name}: {self.current_state.final_response}"
        
        # Generate embedding
        memory_embedding = self.embedding_model.encode(memory_text)
        
        # Extract people and tags
        people = self.current_state.extracted_entities  # Simple heuristic
        tags = self.current_state.extracted_intents
        
        # Calculate affect (simple heuristic)
        affect = 0.0
        response_lower = self.current_state.final_response.lower()
        positive_words = ['good', 'great', 'excellent', 'pleased', 'happy']
        negative_words = ['bad', 'terrible', 'disappointed', 'frustrated', 'concerned']
        
        if any(word in response_lower for word in positive_words):
            affect = 0.5
        elif any(word in response_lower for word in negative_words):
            affect = -0.5
        
        # Store memory
        self.episodic_memory.add_memory(
            text=memory_text,
            embedding=memory_embedding,
            tags=tags,
            people=people,
            affect=affect,
            metadata={
                'salience_score': self.current_state.salience_score,
                'intents': self.current_state.extracted_intents,
                'entities': self.current_state.extracted_entities
            }
        )
    
    def _adjust(self) -> None:
        """Apply drift correction if needed."""
        logger.debug("Checking for drift...")
        
        # Check for value violations in final response
        violations = self.value_gate.check_text(self.current_state.final_response)
        
        if violations:
            logger.warning(f"Detected {len(violations)} value violations - applying correction")
            
            # Re-apply value gate
            corrected_response = self.value_gate.enforce_values(
                self.current_state.final_response,
                context=self.current_state.input_text
            )
            
            if corrected_response != self.current_state.final_response:
                self.current_state.final_response = corrected_response
                logger.info("Applied drift correction")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the conversation history."""
        if not self.conversation_history:
            return {"message": "No conversation history"}
        
        recent_states = self.conversation_history[-10:]  # Last 10 interactions
        
        return {
            "total_interactions": len(self.conversation_history),
            "recent_salience": [state.salience_score for state in recent_states],
            "avg_salience": sum(state.salience_score for state in recent_states) / len(recent_states),
            "common_intents": self._get_common_intents(),
            "common_entities": self._get_common_entities()
        }
    
    def _get_common_intents(self) -> Dict[str, int]:
        """Get frequency of intents in conversation."""
        intent_counts = {}
        for state in self.conversation_history:
            for intent in state.extracted_intents:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
        return intent_counts
    
    def _get_common_entities(self) -> Dict[str, int]:
        """Get frequency of entities in conversation."""
        entity_counts = {}
        for state in self.conversation_history:
            for entity in state.extracted_entities:
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
        return entity_counts
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        self.current_state = None
        logger.info("Cleared conversation history")
    
    def __str__(self) -> str:
        """String representation."""
        return f"CognitiveLoop(persona={self.persona.spec.name})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"CognitiveLoop(persona={self.persona.spec.name}, interactions={len(self.conversation_history)})"
