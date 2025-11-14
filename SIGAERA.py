"""
SIGAERA: Swarms-compatible wrapper for Σ-AERA (AERASIGMA) agent.

This module provides a Swarms-compatible interface to the Σ-AERA cognitive architecture,
allowing it to be used seamlessly within the Swarms framework.

Usage:
    >>> from swarms import SIGAERAAgent
    >>> from swarms.models import OpenAIChat
    >>> 
    >>> llm = OpenAIChat(model_name="gpt-4")
    >>> agent = SIGAERAAgent(llm=llm)
    >>> response = agent.run("Solve this problem: ...")
    >>> print(response)
"""

from typing import Any, Dict, List, Optional, Union, Callable
import time
import json
import os
import pickle
from pathlib import Path
from loguru import logger

from swarms.agents.AERASIGMA import (
    AERASigmaAgent,
    KnowledgeBase,
    Drive,
    CompositeState,
    AtomicPredicate,
    Entity,
    CRM,
    Mreq,
    AntiMreq,
)


class SIGAERAAgent:
    """
    Swarms-compatible wrapper for Σ-AERA cognitive architecture.
    
    This agent wraps the AERASigmaAgent to provide a Swarms-compatible interface
    while preserving all the advanced AGI capabilities of the underlying architecture.
    
    Args:
        llm: Language model backend (OpenAI, Anthropic, etc.) or callable
        model_name: Model name string (alternative to llm)
        max_iterations: Maximum cognitive cycles
        learning_enabled: Enable learning mechanisms (CTPX/PTPX/GTPX)
        analogy_enabled: Enable analogy mechanisms
        use_multi_modal: Enable multi-modal perception
        use_llm_reasoning: Enable LLM as core reasoning component
        use_neural_crms: Enable neural network CRMs
        seed_kb: Initial knowledge base (optional)
        agent_name: Name of the agent
        agent_description: Description of the agent
        system_prompt: System prompt (used for LLM integration)
        verbose: Enable verbose logging
        
    Attributes:
        name: Agent name
        description: Agent description
        system_prompt: System prompt
        
    Methods:
        run: Execute a task using the Σ-AERA cognitive cycle
        add_goal: Add a goal/drive to the agent
        get_knowledge_base_stats: Get statistics about the knowledge base
        generate_explanation: Generate self-explanation of recent behavior
        
    Examples:
        >>> from swarms import SIGAERAAgent
        >>> from swarms.models import OpenAIChat
        >>> 
        >>> llm = OpenAIChat(model_name="gpt-4")
        >>> agent = SIGAERAAgent(
        ...     llm=llm,
        ...     agent_name="sigma-aera-agent",
        ...     learning_enabled=True
        ... )
        >>> 
        >>> # Run a task
        >>> response = agent.run("Analyze the causal relationships in this scenario...")
        >>> 
        >>> # Add a goal
        >>> agent.add_goal("maximize_utility", {"var": "utility", "target": 1.0})
        >>> 
        >>> # Get stats
        >>> stats = agent.get_knowledge_base_stats()
        >>> print(f"CRMs: {stats['num_crms']}, CSTs: {stats['num_csts']}")
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        model_name: Optional[str] = None,
        max_iterations: int = 100,
        learning_enabled: bool = True,
        analogy_enabled: bool = True,
        seed_kb: Optional[KnowledgeBase] = None,
        agent_name: str = "sigma-aera-agent",
        agent_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
        storage_path: Optional[str] = None,
        auto_save: bool = True,
        **kwargs
    ):
        """
        Initialize SIGAERA agent.
        
        Args:
            llm: Language model backend (callable or object with .run() method)
            model_name: Model name (if llm not provided, will try to create from model_name)
            max_iterations: Maximum cognitive cycles
            learning_enabled: Enable learning mechanisms (CTPX/PTPX/GTPX)
            analogy_enabled: Enable analogy mechanisms
            seed_kb: Initial knowledge base
            agent_name: Name of the agent
            agent_description: Description of the agent
            system_prompt: System prompt for LLM
            verbose: Enable verbose logging
            storage_path: Path to storage directory (default: ~/.sigma_aera/{agent_name})
            auto_save: Automatically save knowledge base after learning updates
            **kwargs: Additional arguments (currently unused, reserved for future use)
        """
        # Store agent metadata
        self.name = agent_name
        self.description = agent_description or "Σ-AERA cognitive architecture agent with self-programming, causal reasoning, and multi-modal perception"
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.auto_save = auto_save
        
        # Setup storage path
        if storage_path is None:
            home = Path.home()
            storage_path = str(home / ".sigma_aera" / agent_name)
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.kb_file = self.storage_path / "knowledge_base.pkl"
        self.history_file = self.storage_path / "task_history.json"
        self.metadata_file = self.storage_path / "metadata.json"
        
        # Resolve LLM backend
        llm_backend = self._resolve_llm(llm, model_name)
        
        if verbose:
            logger.info(f"Initializing SIGAERA agent: {agent_name}")
            logger.info(f"Storage path: {self.storage_path}")
            logger.info(f"LLM backend: {llm_backend is not None}")
            logger.info(f"Learning enabled: {learning_enabled}")
            logger.info(f"Analogy enabled: {analogy_enabled}")
        
        # Try to load existing knowledge base
        loaded_kb = None
        if self.kb_file.exists() and seed_kb is None:
            try:
                loaded_kb = self._load_knowledge_base()
                if verbose:
                    logger.info(f"Loaded existing knowledge base from {self.kb_file}")
            except Exception as e:
                if verbose:
                    logger.warning(f"Could not load knowledge base: {e}")
        
        # Use loaded KB or provided seed KB
        final_kb = seed_kb or loaded_kb
        
        # Initialize the underlying AERASigmaAgent
        self.aera_agent = AERASigmaAgent(
            llm=llm_backend,
            model_name=model_name,
            seed_kb=final_kb,
            max_iterations=max_iterations,
            learning_enabled=learning_enabled,
            analogy_enabled=analogy_enabled,
            agent_name=agent_name,
            agent_description=agent_description,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs
        )
        
        # Initialize seed knowledge base if needed
        if final_kb is None:
            self.aera_agent.initialize_seed_kb()
            if auto_save:
                self.save_knowledge_base()
        
        # Load task history if exists
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.task_history = json.load(f)
                if verbose:
                    logger.info(f"Loaded {len(self.task_history)} tasks from history")
            except Exception as e:
                if verbose:
                    logger.warning(f"Could not load task history: {e}")
                self.task_history = []
        else:
            self.task_history = []
        
        # Save metadata
        self._save_metadata()
        
        if verbose:
            logger.info("SIGAERA agent initialized successfully")
    
    def _resolve_llm(self, llm: Optional[Any], model_name: Optional[str]) -> Optional[Any]:
        """
        Resolve LLM backend from various input formats.
        
        Args:
            llm: LLM object or callable
            model_name: Model name string
            
        Returns:
            LLM backend (callable or object with .run() method)
        """
        if llm is not None:
            return llm
        
        if model_name is not None:
            # Try to create LLM from model_name
            try:
                # Try different import paths
                try:
                    from swarms.models import OpenAIChat
                    return OpenAIChat(model_name=model_name)
                except ImportError:
                    # Fallback: create a simple callable wrapper
                    def llm_wrapper(prompt: str) -> str:
                        # This is a placeholder - user should provide actual LLM
                        return f"[LLM response for: {prompt[:50]}...]"
                    return llm_wrapper
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Could not create LLM from model_name '{model_name}': {e}")
                return None
        
        return None
    
    def run(
        self,
        task: str,
        img: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Execute a task using the Σ-AERA cognitive cycle.
        
        This method implements the Swarms-compatible run interface while
        leveraging the full cognitive architecture of Σ-AERA.
        
        Args:
            task: Task description or question
            img: Optional image path or image data (for multi-modal tasks)
            **kwargs: Additional arguments (e.g., available_actions, goals)
            
        Returns:
            Response string from the agent
            
        Examples:
            >>> agent = SIGAERAAgent(llm=llm)
            >>> response = agent.run("What are the causal factors affecting climate change?")
            >>> 
            >>> # Multi-modal task
            >>> response = agent.run(
            ...     "Describe what you see in this image",
            ...     img="/path/to/image.jpg"
            ... )
        """
        if self.verbose:
            logger.info(f"Executing task: {task[:100]}...")
        
        # Prepare observation (can be text, image, or both)
        observation = img if img else task
        
        # Get available actions if provided
        available_actions = kwargs.get("available_actions")
        
        # Execute cognitive cycle
        try:
            result = self.aera_agent.cognitive_cycle(
                observation=observation,
                available_actions=available_actions
            )
            
            # Extract response
            action = result.get("action", "noop")
            beliefs = result.get("beliefs", {})
            expected_utility = result.get("expected_utility", 0.0)
            
            # Format response
            if action != "noop":
                response = f"Action: {action}\n"
            else:
                response = ""
            
            # Add reasoning if available
            if beliefs:
                # Extract most likely beliefs
                key_beliefs = {}
                for var_id, belief in list(beliefs.items())[:5]:  # Top 5 beliefs
                    if belief:
                        best_val = max(belief.items(), key=lambda x: x[1])[0]
                        key_beliefs[str(var_id)] = best_val
                
                if key_beliefs:
                    response += f"Reasoning: {key_beliefs}\n"
            
            # Add utility if significant
            if expected_utility > 0.1:
                response += f"Expected utility: {expected_utility:.3f}\n"
            
            # If we have LLM reasoning, try to get a natural language response
            if self.aera_agent.llm_reasoning_engine and hasattr(self.aera_agent.llm_reasoning_engine.llm_backend, 'run'):
                try:
                    # Use LLM to generate natural language response
                    llm_prompt = f"""
                    Task: {task}
                    Agent's reasoning: {result.get('beliefs', {})}
                    Action taken: {action}
                    
                    Generate a clear, natural language response explaining the agent's reasoning and action.
                    """
                    llm_response = self.aera_agent.llm_reasoning_engine.llm_backend.run(llm_prompt)
                    if llm_response:
                        response = llm_response
                except Exception as e:
                    if self.verbose:
                        logger.debug(f"LLM response generation failed: {e}")
                    # Fall back to structured response
                    if not response or response.strip() == "":
                        response = f"Processed task: {task}\nAction: {action}"
            
            # Store in history
            self.task_history.append({
                "task": task,
                "observation": observation,
                "result": result,
                "response": response,
                "timestamp": time.time()
            })
            
            # Auto-save if enabled
            if self.auto_save:
                self._save_task_history()
                # Save KB if learning occurred
                if self.aera_agent.learning_enabled and result.get("learning_occurred", False):
                    self.save_knowledge_base()
            
            if self.verbose:
                logger.info(f"Task completed. Response length: {len(response)}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error executing task: {str(e)}"
            logger.error(error_msg)
            if self.verbose:
                import traceback
                logger.error(traceback.format_exc())
            return error_msg
    
    def add_goal(
        self,
        goal_id: str,
        goal_condition: Dict[str, Any],
        utility_function: Optional[Callable[[float], float]] = None,
        weight: float = 1.0
    ) -> bool:
        """
        Add a goal/drive to the agent.
        
        Args:
            goal_id: Unique identifier for the goal
            goal_condition: Dictionary mapping variable IDs to target values
            utility_function: Optional utility function (default: linear)
            weight: Weight for multi-goal scenarios
            
        Returns:
            True if goal was added successfully
            
        Examples:
            >>> agent.add_goal(
            ...     "maximize_utility",
            ...     {"utility_var": 1.0, "satisfaction": 0.9}
            ... )
        """
        try:
            # Create conditions for the goal CST
            conditions = []
            for var_id, target_value in goal_condition.items():
                conditions.append(AtomicPredicate(
                    var_id=var_id,
                    operator="=",
                    value=target_value
                ))
            
            # Create goal CST
            goal_cst_id = f"goal_cst_{goal_id}"
            goal_cst = CompositeState(
                id=goal_cst_id,
                conditions=conditions
            )
            
            # Add to knowledge base
            self.aera_agent.kb.add_cst(goal_cst)
            
            # Create drive
            if utility_function is None:
                utility_function = lambda sat: sat  # Linear utility
            
            drive = Drive(
                id=goal_id,
                goal_cst_id=goal_cst_id,
                utility_function=utility_function,
                weight=weight
            )
            
            # Add drive to agent
            self.aera_agent.add_drive(drive)
            
            if self.verbose:
                logger.info(f"Added goal: {goal_id} with {len(conditions)} conditions")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding goal: {e}")
            return False
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with knowledge base statistics
            
        Examples:
            >>> stats = agent.get_knowledge_base_stats()
            >>> print(f"CRMs: {stats['num_crms']}")
        """
        return self.aera_agent.get_knowledge_base_stats()
    
    def generate_explanation(
        self,
        num_steps: int = 5
    ) -> Dict[str, Any]:
        """
        Generate self-explanation of recent behavior.
        
        Args:
            num_steps: Number of recent steps to explain
            
        Returns:
            Explanation dictionary with narrative and CRM IDs
            
        Examples:
            >>> explanation = agent.generate_explanation(num_steps=10)
            >>> print(explanation['explanation'])
        """
        # Check if enhanced agent has explanation generator
        if hasattr(self.aera_agent, 'explanation_generator'):
            return self.aera_agent.generate_self_explanation(num_steps)
        else:
            # Fallback: simple explanation from history
            if len(self.task_history) < num_steps:
                steps = self.task_history
            else:
                steps = self.task_history[-num_steps:]
            
            explanation = {
                "explanation": f"Agent executed {len(steps)} recent tasks.",
                "tasks": [step["task"] for step in steps],
                "num_steps": len(steps)
            }
            
            return explanation
    
    def learn_from_experience(
        self,
        state_before: Dict[str, Any],
        action: str,
        state_after: Dict[str, Any]
    ):
        """
        Learn from an experience tuple.
        
        This allows the agent to learn from observed transitions.
        
        Args:
            state_before: State before action
            action: Action taken
            state_after: State after action
            
        Examples:
            >>> agent.learn_from_experience(
            ...     state_before={"position": 0},
            ...     action="move_right",
            ...     state_after={"position": 1}
            ... )
        """
        if self.aera_agent.learning_enabled:
            self.aera_agent.learn_from_experience(
                x_t=state_before,
                u_t=action,
                x_tp1=state_after
            )
            
            # Auto-save if enabled
            if self.auto_save:
                self.save_knowledge_base()
            
            if self.verbose:
                logger.info(f"Learned from experience: {action}")
    
    def save_knowledge_base(self) -> bool:
        """
        Save the knowledge base to disk.
        
        Returns:
            True if save was successful
        """
        try:
            kb = self.aera_agent.kb
            
            # Serialize knowledge base
            kb_data = {
                "entities": {str(k): self._serialize_entity(v) for k, v in kb.entities.items()},
                "crms": {str(k): self._serialize_crm(v) for k, v in kb.crms.items()},
                "csts": {str(k): self._serialize_cst(v) for k, v in kb.csts.items()},
                "mreqs": {str(k): self._serialize_mreq(v) for k, v in kb.mreqs.items()},
                "anti_mreqs": {str(k): self._serialize_anti_mreq(v) for k, v in kb.anti_mreqs.items()},
                "drives": {str(k): self._serialize_drive(v) for k, v in kb.drives.items()},
                "history": kb.history[-1000:],  # Keep last 1000 experiences
                "attention_policy": kb.attention_policy,
                "rl_modules": kb.rl_modules,
            }
            
            with open(self.kb_file, 'wb') as f:
                pickle.dump(kb_data, f)
            
            if self.verbose:
                logger.info(f"Saved knowledge base to {self.kb_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False
    
    def _load_knowledge_base(self) -> KnowledgeBase:
        """Load knowledge base from disk."""
        with open(self.kb_file, 'rb') as f:
            kb_data = pickle.load(f)
        
        kb = KnowledgeBase()
        
        # Deserialize entities
        for k, v in kb_data.get("entities", {}).items():
            kb.add_entity(self._deserialize_entity(k, v))
        
        # Deserialize CSTs first (CRMs depend on them)
        for k, v in kb_data.get("csts", {}).items():
            kb.add_cst(self._deserialize_cst(k, v))
        
        # Deserialize CRMs
        for k, v in kb_data.get("crms", {}).items():
            kb.add_crm(self._deserialize_crm(k, v))
        
        # Deserialize Mreqs
        for k, v in kb_data.get("mreqs", {}).items():
            kb.add_mreq(self._deserialize_mreq(k, v))
        
        # Deserialize AntiMreqs
        for k, v in kb_data.get("anti_mreqs", {}).items():
            kb.add_anti_mreq(self._deserialize_anti_mreq(k, v))
        
        # Deserialize drives
        for k, v in kb_data.get("drives", {}).items():
            kb.add_drive(self._deserialize_drive(k, v))
        
        # Restore history and other data
        kb.history = kb_data.get("history", [])
        kb.attention_policy = kb_data.get("attention_policy")
        kb.rl_modules = kb_data.get("rl_modules", {})
        
        return kb
    
    def _save_task_history(self):
        """Save task history to disk."""
        try:
            # Keep only last 1000 tasks
            history = self.task_history[-1000:]
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)
        except Exception as e:
            if self.verbose:
                logger.warning(f"Could not save task history: {e}")
    
    def _save_metadata(self):
        """Save agent metadata."""
        try:
            metadata = {
                "agent_name": self.name,
                "description": self.description,
                "created_at": time.time(),
                "last_updated": time.time(),
                "storage_path": str(self.storage_path),
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            if self.verbose:
                logger.warning(f"Could not save metadata: {e}")
    
    # Serialization helpers
    def _serialize_entity(self, entity: Entity) -> Dict:
        return {
            "id": str(entity.id),
            "ontologies": list(entity.ontologies),
            "attributes": {str(k): str(v) for k, v in entity.attributes.items()}
        }
    
    def _deserialize_entity(self, k: str, v: Dict) -> Entity:
        return Entity(
            id=v["id"],
            ontologies=set(v["ontologies"]),
            attributes={k: v for k, v in v["attributes"].items()}
        )
    
    def _serialize_cst(self, cst: CompositeState) -> Dict:
        return {
            "id": str(cst.id),
            "conditions": [
                {
                    "var_id": str(pred.var_id),
                    "operator": pred.operator,
                    "value": str(pred.value) if isinstance(pred.value, (int, float, str)) else pred.value
                }
                for pred in cst.conditions
            ]
        }
    
    def _deserialize_cst(self, k: str, v: Dict) -> CompositeState:
        conditions = [
            AtomicPredicate(
                var_id=cond["var_id"],
                operator=cond["operator"],
                value=cond["value"]
            )
            for cond in v["conditions"]
        ]
        return CompositeState(id=v["id"], conditions=conditions)
    
    def _serialize_crm(self, crm: CRM) -> Dict:
        return {
            "id": str(crm.id),
            "pre_cst_id": str(crm.pre_cst_id),
            "post_cst_id": str(crm.post_cst_id),
            "actions": list(crm.actions),
            "param_model": {
                "model_type": crm.param_model.model_type,
                "parameters": {k: v.tolist() if hasattr(v, 'tolist') else v 
                              for k, v in crm.param_model.parameters.items()}
            },
            "stats": crm.stats
        }
    
    def _deserialize_crm(self, k: str, v: Dict) -> CRM:
        from swarms.agents.AERASIGMA import ParametricModel
        import numpy as np
        
        param_model = ParametricModel(
            model_type=v["param_model"]["model_type"],
            parameters={
                k: np.array(v) if isinstance(v, list) else v
                for k, v in v["param_model"]["parameters"].items()
            }
        )
        
        return CRM(
            id=v["id"],
            pre_cst_id=v["pre_cst_id"],
            post_cst_id=v["post_cst_id"],
            actions=set(v["actions"]),
            param_model=param_model,
            stats=v.get("stats", {})
        )
    
    def _serialize_mreq(self, mreq: Mreq) -> Dict:
        return {
            "id": str(mreq.id),
            "cst_id": str(mreq.cst_id),
            "crm_id": str(mreq.crm_id),
            "confidence": mreq.confidence
        }
    
    def _deserialize_mreq(self, k: str, v: Dict) -> Mreq:
        return Mreq(
            id=v["id"],
            cst_id=v["cst_id"],
            crm_id=v["crm_id"],
            confidence=v.get("confidence", 0.5)
        )
    
    def _serialize_anti_mreq(self, anti_mreq: AntiMreq) -> Dict:
        return {
            "id": str(anti_mreq.id),
            "cst_id": str(anti_mreq.cst_id),
            "crm_id": str(anti_mreq.crm_id),
            "confidence": anti_mreq.confidence
        }
    
    def _deserialize_anti_mreq(self, k: str, v: Dict) -> AntiMreq:
        return AntiMreq(
            id=v["id"],
            cst_id=v["cst_id"],
            crm_id=v["crm_id"],
            confidence=v.get("confidence", 0.5)
        )
    
    def _serialize_drive(self, drive: Drive) -> Dict:
        return {
            "id": str(drive.id),
            "goal_cst_id": str(drive.goal_cst_id),
            "weight": drive.weight,
            "novelty_weight": drive.novelty_weight,
            "exploitation_weight": drive.exploitation_weight,
        }
    
    def _deserialize_drive(self, k: str, v: Dict) -> Drive:
        return Drive(
            id=v["id"],
            goal_cst_id=v["goal_cst_id"],
            weight=v.get("weight", 1.0),
            novelty_weight=v.get("novelty_weight", 0.3),
            exploitation_weight=v.get("exploitation_weight", 0.7),
        )
    
    def clear_storage(self) -> bool:
        """
        Clear all stored data (knowledge base, history, etc.).
        
        Returns:
            True if cleared successfully
        """
        try:
            if self.kb_file.exists():
                self.kb_file.unlink()
            if self.history_file.exists():
                self.history_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            
            if self.verbose:
                logger.info("Cleared all stored data")
            
            return True
        except Exception as e:
            logger.error(f"Error clearing storage: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        stats = self.get_knowledge_base_stats()
        storage_info = f", storage={self.storage_path}" if self.storage_path else ""
        return (
            f"SIGAERAAgent(name='{self.name}', "
            f"CRMs={stats['num_crms']}, "
            f"CSTs={stats['num_csts']}, "
            f"learning={'enabled' if self.aera_agent.learning_enabled else 'disabled'}"
            f"{storage_info})"
        )

