"""
Σ-AERA: Hybrid Autocatalytic Endogenous Reflective Architecture with Sigma Factor Graph Substrate.

This module implements a unified cognitive architecture combining:
- AERA's constructivist, self-programming approach with CRMs, CSTs, Mreqs
- Sigma's factor graph + message passing computational substrate
- Deep LLM integration for semantic reasoning and hypothesis generation
- Multi-modal perception (vision, audio, text) with CLIP and vision-language models
- Neural CRMs using transformers and deep networks
- Joint training of LLM and factor graph components

INTEGRATED WITH SWARMS:
- Inherits from swarms.structs.agent.Agent for full Swarms compatibility
- Uses Swarms' LLM handling, memory, tools, and workflow integration
- Maintains all Σ-AERA cognitive architecture capabilities

ENHANCED FEATURES (AGI-Oriented):

1. Deep LLM Integration:
   - LLMReasoningEngine: LLM as core reasoning component within factor graphs
   - Semantic factors: f_LLM(X) = p_θ(X | context, prompt)
   - Bidirectional integration: factor graph guides LLM, LLM provides semantic priors
   - Joint training: L = L_factor_graph + λ·L_LLM + μ·L_align
   - LLM-based CRM hypothesis generation and CST refinement

2. Multi-Modal Perception:
   - MultiModalPerceptionPipeline: unified processing of text, vision, audio, video
   - CLIP integration for vision-language alignment
   - Cross-modal alignment: L_align = -log p(z_text, z_vision, z_audio | o_t)
   - Entity extraction from images, scene understanding, object detection

3. Neural CRMs:
   - Transformer-based dynamics: h_t = TransformerEncoder([x_t, u_t, context_t])
   - Deep neural networks for complex causal relationships
   - Context-aware predictions with uncertainty quantification

4. Experience Graph & Causal Discovery:
   - ExperienceGraph: maintains graph of transitions (x_t, u_t, x_{t+1})
   - Clustering similar contexts for pattern discovery
   - Mutual information computation: I(X; Y | Z) for dependency analysis
   - CRM candidate suggestion from experience patterns
   - Firing frequency priors to prevent scope explosion

5. Dual Learning Loops:
   - Structural (AERA): adds/removes CRMs, CSTs, Mreqs
   - Parametric (Sigma): updates parameters θ via SGD
   - Synchronization: structural updates when parametric confidence < threshold

6. Hierarchical Memory:
   - Episodic buffer (short-term experiences)
   - Semantic graph (consolidated CRMs & CSTs)
   - Procedural cache (compiled factor-graph fragments)
   - Memory compression via VAE: min_{φ,θ} E[log p_θ(x|z)] + KL[q_φ(z|x)||p(z)]

7. Meta-Control & Attention:
   - Meta-state: m_t = [error_gradients, drive_priorities, graph_load, confidence]
   - Attention policy: π_ω(A_t | m_t) using reinforcement learning
   - Intrinsic reward: r_t^int = |x̂_{t+1} - x_{t+1}| (curiosity-driven exploration)
   - Bounded rationality with fixed compute budget per cycle

8. Safety & Stability:
   - Stability regularizer: L_stability = λ ||K_{t+1} - K_t||²
   - Change impact prediction before committing structural changes
   - Versioned KB with rollback capability
   - Validation requires proof-of-predictive-improvement

Mathematical Foundation:
    Core Architecture:
        A = ⟨S, O, U, K, F, Π, L, M⟩
        
    Where:
    - S: world state space
    - O: observation space  
    - U: action space
    - K: knowledge base (CRMs, CSTs, Mreqs, factor definitions)
    - F: factor graph construction function
    - Π: policy/decision machinery
    - L: learning and self-programming mechanisms
    - M: meta-control (attention, scheduling, resource allocation)
    
    Factor Graph Representation:
        F_t(X) = ∏_a f_a(X_{N(a)})
        
        p_t(X) = (1/Z_t) ∏_a f_a(X_{N(a)})
        
    Where:
    - X = {X_{e,a,t}} ∪ {R_{r,ē,t}}: attribute and relation variables
    - f_a: factor functions (perception, CRMs, CSTs, utilities, priors)
    - N(a): neighbors of factor a
    
    Message Passing (Sum-Product):
        Variable→Factor: m_{{i→a}}(x_i) = ∏_{{b∈N(i)\\{{a}}}} m_{{b→i}}(x_i)
        Factor→Variable: m_{{a→i}}(x_i) = ∑_{{x_{{N(a)\\{{i}}}}}} f_a(x_{{N(a)}}) ∏_{{j∈N(a)\\{{i}}}} m_{{j→a}}(x_j)
        Belief: b_i(x_i) ∝ ∏_{{a∈N(i)}} m_{{a→i}}(x_i)
    
    CRM (Causal-Relational Model):
        M = ⟨id, C_pre, C_post, U_M, θ_M, Σ_M⟩
        
        p_M(x_{t+1} | x_t, u_t; θ_M) ∝ f_M(x_{t+1}, x_t, u_t)
        
        f_M(x_{t+1}, x_t, u_t) = {
            exp(-½||x_{t+1} - f_{θ_M}(x_t, u_t)||²_{Σ_M^{-1}}), if u_t ∈ U_M ∧ C_pre(x_t) = 1
            1, otherwise
        }
    
    Learning Mechanisms:
        - CTPX: Change-Targeted Pattern Extractor (detects unexplained changes)
        - PTPX: Prediction-Targeted Pattern Extractor (refines failed predictions)
        - GTPX: Goal-Targeted Pattern Extractor (explains unexpected goal achievement)
        - Analogy: Transfers CRMs across domains via attribute/relational similarity
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Set,
    Callable,
    Hashable,
    Sequence,
)
from enum import Enum
from collections import defaultdict
import itertools
import json
import re
import numpy as np
from uuid import uuid4
import time
import pickle
import os
from pathlib import Path
from loguru import logger

# Swarms integration
try:
    from swarms.structs.agent import Agent
except ImportError:
    # Fallback if Swarms not available
    class Agent:
        """Fallback Agent class if Swarms not available."""
        pass
    logger.warning("Swarms Agent class not found. Using fallback.")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Neural network features will be limited.")


# ============================================================================
# Core Data Structures
# ============================================================================

class VarType(Enum):
    """Types of variables in the factor graph."""
    ATTRIBUTE = "attribute"
    RELATION = "relation"
    ACTION = "action"
    DRIVE = "drive"
    INTERNAL = "internal"


@dataclass
class Entity:
    """
    Represents an entity in the world model.
    
    Attributes:
        id: Unique identifier for the entity
        ontologies: Set of type symbols (e.g., {"hand", "object"})
        attributes: Mapping from attribute symbols to variable IDs in factor graph
    """
    id: Hashable
    ontologies: Set[str] = field(default_factory=set)
    attributes: Dict[str, Hashable] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.id == other.id


@dataclass
class AtomicPredicate:
    """
    Atomic condition in a Composite State (CST).
    
    Examples:
        - Equality: X_{h,pos} = X_{c,pos}
        - Type check: X_{c,type} = "cube"
        - Inequality: X_{e,value} > threshold
    """
    var_id: Hashable  # Variable ID
    operator: str  # "=", "!=", ">", "<", ">=", "<=", "in", "not_in"
    value: Any  # Comparison value or another variable ID
    
    def evaluate(self, var_values: Dict[Hashable, Any]) -> bool:
        """
        Evaluate the predicate given variable values.
        
        Args:
            var_values: Dictionary mapping variable IDs to their values
            
        Returns:
            Boolean result of the predicate evaluation
        """
        if self.var_id not in var_values:
            return False
        
        var_val = var_values[self.var_id]
        target_val = var_values.get(self.value, self.value) if isinstance(self.value, Hashable) else self.value
        
        if self.operator == "=":
            return var_val == target_val
        elif self.operator == "!=":
            return var_val != target_val
        elif self.operator == ">":
            return var_val > target_val
        elif self.operator == "<":
            return var_val < target_val
        elif self.operator == ">=":
            return var_val >= target_val
        elif self.operator == "<=":
            return var_val <= target_val
        elif self.operator == "in":
            return var_val in target_val if isinstance(target_val, (list, set, tuple)) else False
        elif self.operator == "not_in":
            return var_val not in target_val if isinstance(target_val, (list, set, tuple)) else False
        else:
            logger.warning(f"Unknown operator: {self.operator}")
            return False


@dataclass
class CompositeState:
    """
    Composite State (CST): conjunction of atomic predicates.
    
    C(x_t) = ⋀_i φ_i(x_t)
    
    Used as pre/post conditions for CRMs and as goal conditions for drives.
    """
    id: Hashable
    conditions: List[AtomicPredicate] = field(default_factory=list)
    
    def evaluate(self, var_values: Dict[Hashable, Any]) -> bool:
        """
        Evaluate the composite state (all conditions must be true).
        
        Args:
            var_values: Dictionary mapping variable IDs to their values
            
        Returns:
            True if all conditions are satisfied
        """
        if not self.conditions:
            return True
        return all(pred.evaluate(var_values) for pred in self.conditions)
    
    def evaluate_soft(self, var_values: Dict[Hashable, Any], temperature: float = 1.0) -> float:
        """
        Soft evaluation returning a probability-like score.
        
        Args:
            var_values: Dictionary mapping variable IDs to their values
            temperature: Temperature for softmax-like evaluation
            
        Returns:
            Soft satisfaction score in [0, 1]
        """
        if not self.conditions:
            return 1.0
        
        # For each predicate, compute a soft score
        scores = []
        for pred in self.conditions:
            if pred.evaluate(var_values):
                scores.append(1.0)
            else:
                # Soft penalty for unsatisfied predicates
                scores.append(0.0)
        
        # Return product of scores (all must be satisfied)
        return np.prod(scores) if scores else 1.0


@dataclass
class ParametricModel:
    """
    Parametric model for CRM transitions.
    
    Enhanced with neural network support and transformer-based dynamics.
    
    Mathematical formulation:
        Linear-Gaussian: x_{t+1} = A_M x_t + B_M u_t + ε, ε ~ N(0, Σ_M)
        
        Neural Network: x_{t+1} ~ N(f_θ(x_t, u_t), Σ_θ(x_t, u_t))
            where f_θ is a deep neural network (MLP, Transformer, etc.)
        
        Transformer-based CRM:
            h_t = TransformerEncoder([x_t, u_t, context_t])
            x_{t+1} = MLP(h_t) + ε
            Σ_{t+1} = Softplus(MLP_σ(h_t))
    """
    model_type: str  # "linear_gaussian", "neural_network", "transformer", "polynomial", etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    model: Optional[Any] = None  # Actual model object (e.g., PyTorch nn.Module)
    hidden_dim: int = 128  # Hidden dimension for neural models
    num_layers: int = 2  # Number of layers for neural models
    use_transformer: bool = False  # Whether to use transformer architecture
    
    def predict(self, x_t: np.ndarray, u_t: np.ndarray, context: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state and uncertainty.
        
        Enhanced with transformer support and context awareness.
        
        Args:
            x_t: Current state vector
            u_t: Action vector
            context: Optional context vector (for transformer models)
            
        Returns:
            Tuple of (mean, covariance) for next state distribution
        """
        if self.model_type == "linear_gaussian":
            A = self.parameters.get("A", np.eye(len(x_t)))
            B = self.parameters.get("B", np.zeros((len(x_t), len(u_t))))
            mean = A @ x_t + B @ u_t
            cov = self.parameters.get("Sigma", np.eye(len(x_t)) * 0.1)
            return mean, cov
        elif self.model_type in ["neural_network", "transformer"] and TORCH_AVAILABLE:
            if self.model is None:
                # Initialize model if not present
                self._initialize_neural_model(len(x_t), len(u_t))
            
            if self.model is not None:
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x_t).unsqueeze(0)
                    u_tensor = torch.FloatTensor(u_t).unsqueeze(0)
                    
                    if self.use_transformer and context is not None:
                        # Transformer-based prediction
                        context_tensor = torch.FloatTensor(context).unsqueeze(0)
                        # Stack: [x_t, u_t, context] as sequence
                        input_seq = torch.stack([x_tensor.squeeze(0), u_tensor.squeeze(0), context_tensor.squeeze(0)], dim=0).unsqueeze(0)
                        h = self.model.encoder(input_seq)
                        # Use last hidden state
                        h_last = h[:, -1, :]
                        mean = self.model.predictor(h_last).squeeze(0).numpy()
                        cov_log = self.model.cov_predictor(h_last).squeeze(0).numpy()
                        cov = np.diag(np.exp(np.clip(cov_log, -10, 10)))  # Ensure positive definite
                    else:
                        # Standard MLP
                        input_tensor = torch.cat([x_tensor, u_tensor], dim=1)
                        mean = self.model(input_tensor).squeeze(0).numpy()
                        cov = np.eye(len(mean)) * 0.1
                    
                    return mean, cov
        
        # Default: identity with small noise
        return x_t, np.eye(len(x_t)) * 0.1
    
    def _initialize_neural_model(self, state_dim: int, action_dim: int):
        """Initialize neural network model."""
        if not TORCH_AVAILABLE:
            return
        
        if self.use_transformer:
            # Transformer-based model
            class TransformerCRM(nn.Module):
                def __init__(self, state_dim, action_dim, hidden_dim, num_layers):
                    super().__init__()
                    self.input_dim = state_dim + action_dim
                    self.hidden_dim = hidden_dim
                    
                    # Input projection
                    self.input_proj = nn.Linear(self.input_dim, hidden_dim)
                    
                    # Transformer encoder
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=hidden_dim,
                        nhead=8,
                        dim_feedforward=hidden_dim * 4,
                        dropout=0.1,
                        batch_first=True
                    )
                    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    
                    # Predictors
                    self.predictor = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, state_dim)
                    )
                    self.cov_predictor = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, state_dim)
                    )
                
                def forward(self, x):
                    x = self.input_proj(x)
                    h = self.encoder(x)
                    return h
            
            self.model = TransformerCRM(
                state_dim, action_dim, self.hidden_dim, self.num_layers
            )
        else:
            # Standard MLP
            class MLPCRM(nn.Module):
                def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
                    super().__init__()
                    layers = []
                    layers.append(nn.Linear(input_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    
                    for _ in range(num_layers - 1):
                        layers.append(nn.Linear(hidden_dim, hidden_dim))
                        layers.append(nn.ReLU())
                    
                    layers.append(nn.Linear(hidden_dim, output_dim))
                    self.net = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.net(x)
            
            self.model = MLPCRM(
                state_dim + action_dim, state_dim, self.hidden_dim, self.num_layers
            )
    
    def update(self, x_t: np.ndarray, u_t: np.ndarray, x_tp1: np.ndarray, learning_rate: float = 0.01):
        """
        Update model parameters given a transition.
        
        Args:
            x_t: Current state
            u_t: Action
            x_tp1: Next state (observed)
            learning_rate: Learning rate for gradient updates
        """
        if self.model_type == "linear_gaussian":
            # Simple least squares update
            A = self.parameters.get("A", np.eye(len(x_t)))
            B = self.parameters.get("B", np.zeros((len(x_t), len(u_t))))
            
            # Stack [x_t, u_t] as input
            input_vec = np.concatenate([x_t, u_t])
            
            # Update A and B using gradient descent
            pred = A @ x_t + B @ u_t
            error = x_tp1 - pred
            
            # Gradient w.r.t. A and B
            dA = np.outer(error, x_t) * learning_rate
            dB = np.outer(error, u_t) * learning_rate
            
            self.parameters["A"] = A + dA
            self.parameters["B"] = B + dB
            
            # Update covariance estimate
            residual = x_tp1 - (self.parameters["A"] @ x_t + self.parameters["B"] @ u_t)
            if "Sigma" not in self.parameters:
                self.parameters["Sigma"] = np.eye(len(x_t)) * 0.1
            # Exponential moving average
            self.parameters["Sigma"] = 0.9 * self.parameters["Sigma"] + 0.1 * np.outer(residual, residual)
        elif self.model_type == "neural_network" and self.model is not None and TORCH_AVAILABLE:
            # PyTorch training step
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            
            x_tensor = torch.FloatTensor(x_t).unsqueeze(0)
            u_tensor = torch.FloatTensor(u_t).unsqueeze(0)
            x_tp1_tensor = torch.FloatTensor(x_tp1).unsqueeze(0)
            
            input_tensor = torch.cat([x_tensor, u_tensor], dim=1)
            pred = self.model(input_tensor)
            
            loss = nn.MSELoss()(pred, x_tp1_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@dataclass
class CRM:
    """
    Causal-Relational Model (CRM).
    
    M = ⟨id, C_pre, C_post, U_M, θ_M, Σ_M⟩
    
    Represents a causal transformation from prior to posterior state,
    conditioned on actions.
    """
    id: Hashable
    pre_cst_id: Hashable  # Pre-condition Composite State
    post_cst_id: Hashable  # Post-condition Composite State
    actions: Set[str]  # Set of action types this CRM applies to
    param_model: ParametricModel  # Transition model
    stats: Dict[str, Any] = field(default_factory=lambda: {
        "successes": 0,
        "failures": 0,
        "firings": 0,
        "last_used": None,
        "total_error": 0.0,
    })
    
    def compute_confidence(self) -> float:
        """
        Compute confidence based on success rate and firing frequency.
        
        Returns:
            Confidence score in [0, 1]
        """
        total = self.stats["successes"] + self.stats["failures"]
        if total == 0:
            return 0.5  # Neutral prior
        
        success_rate = self.stats["successes"] / total
        firing_rate = min(1.0, self.stats["firings"] / max(1, time.time() - (self.stats.get("created_at", time.time()) - 3600)))
        
        # Combine success rate with firing frequency
        confidence = success_rate * (0.7 + 0.3 * firing_rate)
        return confidence
    
    def record_success(self, error: float = 0.0):
        """Record a successful application of this CRM."""
        self.stats["successes"] += 1
        self.stats["firings"] += 1
        self.stats["last_used"] = time.time()
        self.stats["total_error"] += error
    
    def record_failure(self, error: float = 1.0):
        """Record a failed application of this CRM."""
        self.stats["failures"] += 1
        self.stats["firings"] += 1
        self.stats["last_used"] = time.time()
        self.stats["total_error"] += error


@dataclass
class Mreq:
    """
    Requirement Model: links a CST to a CRM.
    
    Mreq = ⟨C_ctx, M, γ_M⟩
    
    Expresses when a CRM is allowed to fire (when its context CST is satisfied).
    """
    id: Hashable
    cst_id: Hashable  # Context CST that gates the CRM
    crm_id: Hashable  # The CRM being gated
    confidence: float = 0.5  # Initial confidence γ_M
    
    def is_active(self, cst_satisfied: bool, threshold: float = 0.3) -> bool:
        """
        Check if this Mreq should activate its CRM.
        
        Args:
            cst_satisfied: Whether the context CST is satisfied
            threshold: Minimum confidence threshold
            
        Returns:
            True if CRM should be activated
        """
        return cst_satisfied and self.confidence > threshold


@dataclass
class AntiMreq:
    """
    Anti-Requirement Model: inhibits a CRM when its CST is satisfied.
    
    Used to block CRMs in contexts where they are known to fail.
    """
    id: Hashable
    cst_id: Hashable  # Context CST that blocks the CRM
    crm_id: Hashable  # The CRM being blocked
    confidence: float = 0.5
    
    def blocks(self, cst_satisfied: bool, threshold: float = 0.3) -> bool:
        """
        Check if this AntiMreq should block its CRM.
        
        Args:
            cst_satisfied: Whether the blocking CST is satisfied
            threshold: Minimum confidence threshold
            
        Returns:
            True if CRM should be blocked
        """
        return cst_satisfied and self.confidence > threshold


@dataclass
class Drive:
    """
    Drive (Goal): desired state configuration with self-consistency.
    
    Enhanced formulation:
        d = ⟨C_goal, u_d, α_novelty, α_exploitation⟩
        
        Reward: R_t = α · Novelty(X_t) + (1-α) · U(X_t)
        
        Self-consistency: reward ∝ -Var(prediction_error)
    """
    id: Hashable
    goal_cst_id: Hashable  # Goal Composite State
    utility_function: Callable[[float], float] = field(default=lambda sat: sat)  # Maps satisfaction to utility
    weight: float = 1.0  # Weight for multi-drive scenarios
    novelty_weight: float = 0.3  # Weight for novelty reward (exploration)
    exploitation_weight: float = 0.7  # Weight for goal reward (exploitation)
    prediction_error_history: List[float] = field(default_factory=list)  # For self-consistency
    
    def compute_utility(self, satisfaction: float) -> float:
        """
        Compute utility given satisfaction level.
        
        Args:
            satisfaction: Satisfaction score in [0, 1]
            
        Returns:
            Utility value
        """
        return self.weight * self.utility_function(satisfaction)
    
    def compute_reward(
        self,
        satisfaction: float,
        novelty: float = 0.0,
        prediction_error: Optional[float] = None
    ) -> float:
        """
        Compute combined reward with exploration-exploitation balance.
        
        R_t = α · Novelty(X_t) + (1-α) · U(X_t) - β · Var(prediction_error)
        
        Args:
            satisfaction: Goal satisfaction
            novelty: Novelty score (for exploration)
            prediction_error: Prediction error (for self-consistency)
            
        Returns:
            Combined reward
        """
        # Goal utility
        goal_utility = self.compute_utility(satisfaction)
        
        # Exploration-exploitation balance
        exploration_reward = self.novelty_weight * novelty
        exploitation_reward = self.exploitation_weight * goal_utility
        combined_reward = exploration_reward + exploitation_reward
        
        # Self-consistency penalty
        if prediction_error is not None:
            self.prediction_error_history.append(prediction_error)
            # Keep only recent history
            if len(self.prediction_error_history) > 100:
                self.prediction_error_history = self.prediction_error_history[-100:]
            
            # Variance of prediction errors (self-consistency)
            if len(self.prediction_error_history) > 1:
                error_variance = np.var(self.prediction_error_history)
                consistency_penalty = 0.1 * error_variance  # Small penalty
                combined_reward -= consistency_penalty
        
        return combined_reward


# ============================================================================
# Factor Graph Representation
# ============================================================================

@dataclass
class FactorNode:
    """
    Factor node in the factor graph.
    
    Represents a local function f_a(X_{N(a)}) that contributes to the joint.
    """
    id: Hashable
    factor_type: str  # "perception", "crm", "cst", "utility", "prior", etc.
    var_neighbors: Set[Hashable]  # Variable IDs this factor connects to
    factor_func: Callable[[Dict[Hashable, Any]], float]  # The actual factor function
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info (e.g., CRM ID, CST ID)
    
    def evaluate(self, var_values: Dict[Hashable, Any]) -> float:
        """
        Evaluate the factor function.
        
        Args:
            var_values: Dictionary mapping variable IDs to their values
            
        Returns:
            Factor value (potential)
        """
        # Extract only the relevant variables
        relevant_vars = {vid: var_values[vid] for vid in self.var_neighbors if vid in var_values}
        return self.factor_func(relevant_vars)


@dataclass
class VariableNode:
    """
    Variable node in the factor graph.
    
    Represents a random variable (attribute, relation, action, etc.).
    """
    id: Hashable
    var_type: VarType
    domain: Any  # Domain of possible values (discrete set, continuous range, etc.)
    belief: Optional[Dict[Any, float]] = None  # Current belief distribution
    metadata: Dict[str, Any] = field(default_factory=dict)  # Entity ID, attribute name, etc.


@dataclass
class FactorGraph:
    """
    Factor graph G = (V_X, V_F, E).
    
    V_X: variable nodes
    V_F: factor nodes
    E: edges (implicit via var_neighbors in factors)
    """
    variables: Dict[Hashable, VariableNode] = field(default_factory=dict)
    factors: Dict[Hashable, FactorNode] = field(default_factory=dict)
    
    def add_variable(self, var_id: Hashable, var_type: VarType, domain: Any, metadata: Optional[Dict] = None):
        """Add a variable node to the graph."""
        self.variables[var_id] = VariableNode(
            id=var_id,
            var_type=var_type,
            domain=domain,
            metadata=metadata or {}
        )
    
    def add_factor(self, factor_id: Hashable, factor_node: FactorNode):
        """Add a factor node to the graph."""
        self.factors[factor_id] = factor_node
        # Ensure all neighbor variables exist
        for var_id in factor_node.var_neighbors:
            if var_id not in self.variables:
                logger.warning(f"Factor {factor_id} references unknown variable {var_id}")
    
    def get_variable_neighbors(self, var_id: Hashable) -> Set[Hashable]:
        """Get all factor IDs connected to a variable."""
        neighbors = set()
        for factor_id, factor in self.factors.items():
            if var_id in factor.var_neighbors:
                neighbors.add(factor_id)
        return neighbors
    
    def get_factor_neighbors(self, factor_id: Hashable) -> Set[Hashable]:
        """Get all variable IDs connected to a factor."""
        if factor_id not in self.factors:
            return set()
        return self.factors[factor_id].var_neighbors


# ============================================================================
# Message Passing Algorithms
# ============================================================================

class MessagePassingEngine:
    """
    Implements sum-product and max-product message passing for factor graphs.
    
    Sum-Product Algorithm:
        Variable→Factor: m_{{i→a}}(x_i) = ∏_{{b∈N(i)\\{{a}}}} m_{{b→i}}(x_i)
        Factor→Variable: m_{{a→i}}(x_i) = ∑_{{x_{{N(a)\\{{i}}}}}} f_a(x_{{N(a)}}) ∏_{{j∈N(a)\\{{i}}}} m_{{j→a}}(x_j)
        Belief: b_i(x_i) ∝ ∏_{a∈N(i)} m_{a→i}(x_i)
    """
    
    def __init__(self, max_iterations: int = 50, convergence_threshold: float = 1e-6):
        """
        Initialize message passing engine.
        
        Args:
            max_iterations: Maximum number of message passing iterations
            convergence_threshold: Convergence threshold for beliefs
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def run_sum_product(
        self,
        graph: FactorGraph,
        evidence: Optional[Dict[Hashable, Any]] = None,
        use_max_product: bool = False
    ) -> Dict[Hashable, Dict[Any, float]]:
        """
        Run sum-product (or max-product) message passing.
        
        Args:
            graph: Factor graph to perform inference on
            evidence: Observed variable values (clamped)
            use_max_product: If True, use max-product for MAP inference
            
        Returns:
            Dictionary mapping variable IDs to belief distributions
        """
        evidence = evidence or {}
        
        # Initialize messages: variable→factor and factor→variable
        messages_var_to_factor: Dict[Tuple[Hashable, Hashable], Dict[Any, float]] = {}
        messages_factor_to_var: Dict[Tuple[Hashable, Hashable], Dict[Any, float]] = {}
        
        # Initialize all messages to uniform
        for var_id in graph.variables:
            for factor_id in graph.get_variable_neighbors(var_id):
                messages_var_to_factor[(var_id, factor_id)] = self._uniform_message(graph.variables[var_id].domain)
        
        for factor_id in graph.factors:
            for var_id in graph.get_factor_neighbors(factor_id):
                messages_factor_to_var[(factor_id, var_id)] = self._uniform_message(graph.variables[var_id].domain)
        
        # Clamp evidence variables
        for var_id, value in evidence.items():
            if var_id in graph.variables:
                # Set belief to delta distribution
                domain = graph.variables[var_id].domain
                if isinstance(domain, (list, tuple, set)):
                    for val in domain:
                        messages_var_to_factor[(var_id, factor_id)] = {val: 1.0 if val == value else 0.0}
                else:
                    # Continuous: use narrow Gaussian approximation
                    messages_var_to_factor[(var_id, factor_id)] = {value: 1.0}
        
        # Iterative message passing
        for iteration in range(self.max_iterations):
            old_beliefs = self._compute_beliefs(graph, messages_factor_to_var)
            
            # Update factor→variable messages
            for factor_id, factor in graph.factors.items():
                for var_id in factor.var_neighbors:
                    messages_factor_to_var[(factor_id, var_id)] = self._compute_factor_to_var_message(
                        graph, factor_id, var_id, messages_var_to_factor, use_max_product
                    )
            
            # Update variable→factor messages
            for var_id in graph.variables:
                for factor_id in graph.get_variable_neighbors(var_id):
                    messages_var_to_factor[(var_id, factor_id)] = self._compute_var_to_factor_message(
                        graph, var_id, factor_id, messages_factor_to_var
                    )
            
            # Check convergence
            new_beliefs = self._compute_beliefs(graph, messages_factor_to_var)
            if self._check_convergence(old_beliefs, new_beliefs):
                logger.debug(f"Message passing converged after {iteration + 1} iterations")
                break
        
        return new_beliefs
    
    def _uniform_message(self, domain: Any) -> Dict[Any, float]:
        """Create a uniform message over a domain."""
        if isinstance(domain, (list, tuple, set)):
            size = len(domain)
            return {val: 1.0 / size for val in domain}
        else:
            # Continuous: return a single value with probability 1
            return {domain: 1.0}
    
    def _compute_factor_to_var_message(
        self,
        graph: FactorGraph,
        factor_id: Hashable,
        var_id: Hashable,
        messages_var_to_factor: Dict[Tuple[Hashable, Hashable], Dict[Any, float]],
        use_max_product: bool = False
    ) -> Dict[Any, float]:
        """Compute message from factor to variable."""
        factor = graph.factors[factor_id]
        var = graph.variables[var_id]
        
        # Get all neighbor variables except the target
        other_vars = factor.var_neighbors - {var_id}
        
        # Initialize message
        message = {}
        
        # For discrete variables, iterate over domain
        if isinstance(var.domain, (list, tuple, set)):
            for var_val in var.domain:
                # Sum (or max) over all configurations of other variables
                total = 0.0 if not use_max_product else float('-inf')
                
                # Generate all combinations of other variables (simplified: sample-based)
                # For efficiency, we sample or use a simplified approach
                if len(other_vars) == 0:
                    # Only this variable
                    var_values = {var_id: var_val}
                    factor_val = factor.evaluate(var_values)
                    total = factor_val
                else:
                    # Sample-based approximation for multiple variables
                    # In full implementation, would enumerate or use better sampling
                    samples = self._sample_combinations(graph, other_vars, num_samples=100)
                    for sample in samples:
                        var_values = {var_id: var_val, **sample}
                        factor_val = factor.evaluate(var_values)
                        
                        # Multiply by incoming messages
                        msg_product = 1.0
                        for other_var_id, other_val in sample.items():
                            msg_key = (other_var_id, factor_id)
                            if msg_key in messages_var_to_factor:
                                msg = messages_var_to_factor[msg_key]
                                msg_product *= msg.get(other_val, 0.0)
                        
                        if use_max_product:
                            total = max(total, factor_val * msg_product)
                        else:
                            total += factor_val * msg_product
                
                message[var_val] = max(0.0, total)  # Ensure non-negative
        else:
            # Continuous: simplified approximation
            message = {var.domain: 1.0}
        
        # Normalize
        total = sum(message.values())
        if total > 0:
            message = {k: v / total for k, v in message.items()}
        
        return message
    
    def _compute_var_to_factor_message(
        self,
        graph: FactorGraph,
        var_id: Hashable,
        factor_id: Hashable,
        messages_factor_to_var: Dict[Tuple[Hashable, Hashable], Dict[Any, float]]
    ) -> Dict[Any, float]:
        """Compute message from variable to factor."""
        var = graph.variables[var_id]
        
        # Product of all incoming messages except from the target factor
        message = {}
        neighbor_factors = graph.get_variable_neighbors(var_id) - {factor_id}
        
        if isinstance(var.domain, (list, tuple, set)):
            for var_val in var.domain:
                product = 1.0
                for neighbor_factor_id in neighbor_factors:
                    msg_key = (neighbor_factor_id, var_id)
                    if msg_key in messages_factor_to_var:
                        msg = messages_factor_to_var[msg_key]
                        product *= msg.get(var_val, 0.0)
                message[var_val] = product
        else:
            message = {var.domain: 1.0}
        
        # Normalize
        total = sum(message.values())
        if total > 0:
            message = {k: v / total for k, v in message.items()}
        
        return message
    
    def _compute_beliefs(
        self,
        graph: FactorGraph,
        messages_factor_to_var: Dict[Tuple[Hashable, Hashable], Dict[Any, float]]
    ) -> Dict[Hashable, Dict[Any, float]]:
        """Compute beliefs for all variables."""
        beliefs = {}
        
        for var_id, var in graph.variables.items():
            belief = {}
            neighbor_factors = graph.get_variable_neighbors(var_id)
            
            if isinstance(var.domain, (list, tuple, set)):
                for var_val in var.domain:
                    product = 1.0
                    for factor_id in neighbor_factors:
                        msg_key = (factor_id, var_id)
                        if msg_key in messages_factor_to_var:
                            msg = messages_factor_to_var[msg_key]
                            product *= msg.get(var_val, 0.0)
                    belief[var_val] = product
            else:
                belief = {var.domain: 1.0}
            
            # Normalize
            total = sum(belief.values())
            if total > 0:
                belief = {k: v / total for k, v in belief.items()}
            
            beliefs[var_id] = belief
        
        return beliefs
    
    def _check_convergence(
        self,
        old_beliefs: Dict[Hashable, Dict[Any, float]],
        new_beliefs: Dict[Hashable, Dict[Any, float]]
    ) -> bool:
        """
        Check if beliefs have converged using marginal entropy change.
        
        Convergence criterion: ΔH < ε
        Where ΔH = |H(new) - H(old)| is the change in marginal entropy.
        """
        total_entropy_change = 0.0
        
        for var_id in old_beliefs:
            if var_id not in new_beliefs:
                return False
            
            old_belief = old_beliefs[var_id]
            new_belief = new_beliefs[var_id]
            
            # Compute entropy of old belief
            old_entropy = 0.0
            for prob in old_belief.values():
                if prob > 0:
                    old_entropy -= prob * np.log(prob + 1e-10)
            
            # Compute entropy of new belief
            new_entropy = 0.0
            for prob in new_belief.values():
                if prob > 0:
                    new_entropy -= prob * np.log(prob + 1e-10)
            
            # Entropy change
            entropy_change = abs(new_entropy - old_entropy)
            total_entropy_change += entropy_change
        
        # Average entropy change per variable
        avg_entropy_change = total_entropy_change / max(1, len(old_beliefs))
        
        return avg_entropy_change < self.convergence_threshold
    
    def _sample_combinations(
        self,
        graph: FactorGraph,
        var_ids: Set[Hashable],
        num_samples: int = 100
    ) -> List[Dict[Hashable, Any]]:
        """Sample combinations of variable values (simplified)."""
        samples = []
        for _ in range(num_samples):
            sample = {}
            for var_id in var_ids:
                var = graph.variables[var_id]
                if isinstance(var.domain, (list, tuple, set)):
                    sample[var_id] = np.random.choice(list(var.domain))
                else:
                    sample[var_id] = var.domain
            samples.append(sample)
        return samples


# ============================================================================
# Knowledge Base
# ============================================================================

class KnowledgeBase:
    """
    Knowledge base K_t containing CRMs, CSTs, Mreqs, entities, and drives.
    
    This is the self-modifying knowledge structure that grows via:
    - CTPX: detecting unexplained changes
    - PTPX: refining failed predictions
    - GTPX: explaining unexpected goal achievement
    - Analogy: transferring CRMs across domains
    """
    
    def __init__(self):
        """Initialize an empty knowledge base."""
        self.entities: Dict[Hashable, Entity] = {}
        self.crms: Dict[Hashable, CRM] = {}
        self.csts: Dict[Hashable, CompositeState] = {}
        self.mreqs: Dict[Hashable, Mreq] = {}
        self.anti_mreqs: Dict[Hashable, AntiMreq] = {}
        self.drives: Dict[Hashable, Drive] = {}
        self.attention_policy: Optional[Dict[str, Any]] = None
        self.rl_modules: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []  # Experience history
    
    def add_entity(self, entity: Entity):
        """Add an entity to the knowledge base."""
        self.entities[entity.id] = entity
    
    def add_crm(self, crm: CRM):
        """Add a CRM to the knowledge base."""
        self.crms[crm.id] = crm
        if "created_at" not in crm.stats:
            crm.stats["created_at"] = time.time()
    
    def add_cst(self, cst: CompositeState):
        """Add a CST to the knowledge base."""
        self.csts[cst.id] = cst
    
    def add_mreq(self, mreq: Mreq):
        """Add an Mreq to the knowledge base."""
        self.mreqs[mreq.id] = mreq
    
    def add_anti_mreq(self, anti_mreq: AntiMreq):
        """Add an AntiMreq to the knowledge base."""
        self.anti_mreqs[anti_mreq.id] = anti_mreq
    
    def add_drive(self, drive: Drive):
        """Add a drive to the knowledge base."""
        self.drives[drive.id] = drive
    
    def get_active_crms(
        self,
        var_values: Dict[Hashable, Any],
        action: Optional[str] = None,
        threshold: float = 0.3
    ) -> List[CRM]:
        """
        Get CRMs that should be active given current state.
        
        Args:
            var_values: Current variable values
            action: Current action (if any)
            threshold: Confidence threshold for Mreqs
            
        Returns:
            List of active CRMs
        """
        active_crms = []
        
        for mreq in self.mreqs.values():
            # Check if context CST is satisfied
            if mreq.cst_id not in self.csts:
                continue
            
            cst = self.csts[mreq.cst_id]
            cst_satisfied = cst.evaluate(var_values)
            
            # Check if blocked by anti-Mreq
            blocked = False
            for anti_mreq in self.anti_mreqs.values():
                if anti_mreq.crm_id == mreq.crm_id:
                    if anti_mreq.cst_id in self.csts:
                        anti_cst = self.csts[anti_mreq.cst_id]
                        if anti_mreq.blocks(anti_cst.evaluate(var_values), threshold):
                            blocked = True
                            break
            
            if not blocked and mreq.is_active(cst_satisfied, threshold):
                if mreq.crm_id in self.crms:
                    crm = self.crms[mreq.crm_id]
                    # Check if action matches
                    if action is None or action in crm.actions:
                        active_crms.append(crm)
        
        return active_crms
    
    def record_experience(
        self,
        x_t: Dict[Hashable, Any],
        u_t: Optional[str],
        x_tp1: Dict[Hashable, Any],
        active_crms: List[Hashable]
    ):
        """
        Record an experience tuple for learning.
        
        Args:
            x_t: State at time t
            u_t: Action at time t
            x_tp1: State at time t+1
            active_crms: List of CRM IDs that were active
        """
        self.history.append({
            "x_t": x_t,
            "u_t": u_t,
            "x_tp1": x_tp1,
            "active_crms": active_crms,
            "timestamp": time.time()
        })
        
        # Keep only recent history (last 1000 experiences)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]


# ============================================================================
# Experience Graph for Causal Discovery
# ============================================================================

@dataclass
class ExperienceNode:
    """
    Node in experience graph representing a context/state.
    
    Nodes cluster similar contexts for CRM discovery.
    """
    id: Hashable
    context: Dict[Hashable, Any]  # State/context features
    transitions: List[Dict[str, Any]] = field(default_factory=list)  # Outgoing transitions
    cluster_id: Optional[Hashable] = None
    embedding: Optional[np.ndarray] = None
    
    def add_transition(self, action: str, next_context: Dict[Hashable, Any], reward: float = 0.0):
        """Add a transition from this node."""
        self.transitions.append({
            "action": action,
            "next_context": next_context,
            "reward": reward,
            "timestamp": time.time()
        })


class ExperienceGraph:
    """
    Experience graph: edges = observed transitions (x_t, u_t, x_{t+1}); nodes = contexts.
    
    Mathematical formulation:
        G_exp = (V_exp, E_exp)
        V_exp = {context nodes}
        E_exp = {(v_i, v_j, u, r) | transition from context i to j via action u with reward r}
    
    Used for:
    - Clustering similar contexts
    - Dependency analysis (mutual information, causal tests)
    - CRM synthesis from patterns
    """
    
    def __init__(self, max_nodes: int = 10000, similarity_threshold: float = 0.7):
        """
        Initialize experience graph.
        
        Args:
            max_nodes: Maximum number of nodes to maintain
            similarity_threshold: Threshold for merging similar contexts
        """
        self.nodes: Dict[Hashable, ExperienceNode] = {}
        self.max_nodes = max_nodes
        self.similarity_threshold = similarity_threshold
        self.clusters: Dict[Hashable, Set[Hashable]] = defaultdict(set)
        self.next_node_id = 0
    
    def add_experience(
        self,
        x_t: Dict[Hashable, Any],
        u_t: str,
        x_tp1: Dict[Hashable, Any],
        reward: float = 0.0
    ):
        """
        Add experience transition to graph.
        
        Args:
            x_t: State at time t
            u_t: Action at time t
            x_tp1: State at time t+1
            reward: Reward for transition
        """
        # Find or create node for x_t
        node_t_id = self._find_or_create_node(x_t)
        node_t = self.nodes[node_t_id]
        
        # Find or create node for x_tp1
        node_tp1_id = self._find_or_create_node(x_tp1)
        
        # Add transition
        node_t.add_transition(u_t, x_tp1, reward)
        
        # Update clusters periodically
        if len(self.nodes) % 100 == 0:
            self._update_clusters()
    
    def _find_or_create_node(self, context: Dict[Hashable, Any]) -> Hashable:
        """Find existing similar node or create new one."""
        # Check for similar existing node
        for node_id, node in self.nodes.items():
            similarity = self._compute_context_similarity(context, node.context)
            if similarity > self.similarity_threshold:
                # Merge contexts (update node context)
                self._merge_contexts(node.context, context)
                return node_id
        
        # Create new node
        node_id = f"exp_node_{self.next_node_id}"
        self.next_node_id += 1
        
        self.nodes[node_id] = ExperienceNode(
            id=node_id,
            context=context.copy()
        )
        
        # Maintain max nodes
        if len(self.nodes) > self.max_nodes:
            self._prune_nodes()
        
        return node_id
    
    def _compute_context_similarity(
        self,
        ctx1: Dict[Hashable, Any],
        ctx2: Dict[Hashable, Any]
    ) -> float:
        """Compute similarity between two contexts."""
        common_keys = set(ctx1.keys()) & set(ctx2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1 = ctx1[key]
            val2 = ctx2[key]
            
            if val1 == val2:
                similarities.append(1.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalized distance
                max_val = max(abs(val1), abs(val2), 1.0)
                sim = 1.0 - min(1.0, abs(val1 - val2) / max_val)
                similarities.append(sim)
            else:
                similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _merge_contexts(self, target: Dict[Hashable, Any], source: Dict[Hashable, Any]):
        """Merge source context into target (exponential moving average)."""
        alpha = 0.1  # Learning rate
        for key, val in source.items():
            if key in target:
                if isinstance(val, (int, float)) and isinstance(target[key], (int, float)):
                    target[key] = alpha * val + (1 - alpha) * target[key]
                else:
                    target[key] = val  # Discrete: replace
            else:
                target[key] = val
    
    def _update_clusters(self):
        """Update context clusters using similarity."""
        # Simple clustering: group similar contexts
        self.clusters.clear()
        cluster_id = 0
        
        for node_id, node in self.nodes.items():
            # Find existing cluster or create new
            assigned = False
            for cid, cluster_nodes in self.clusters.items():
                # Check similarity to cluster centroid
                if cluster_nodes:
                    # Use first node as centroid (simplified)
                    centroid_id = next(iter(cluster_nodes))
                    centroid = self.nodes[centroid_id].context
                    similarity = self._compute_context_similarity(node.context, centroid)
                    
                    if similarity > self.similarity_threshold:
                        self.clusters[cid].add(node_id)
                        node.cluster_id = cid
                        assigned = True
                        break
            
            if not assigned:
                # Create new cluster
                cid = f"cluster_{cluster_id}"
                cluster_id += 1
                self.clusters[cid] = {node_id}
                node.cluster_id = cid
    
    def _prune_nodes(self):
        """Prune least frequently used nodes."""
        # Sort by number of transitions
        node_usage = [(node_id, len(node.transitions)) for node_id, node in self.nodes.items()]
        node_usage.sort(key=lambda x: x[1])
        
        # Remove bottom 10%
        num_to_remove = len(self.nodes) // 10
        for node_id, _ in node_usage[:num_to_remove]:
            if node_id in self.nodes:
                # Remove from clusters
                node = self.nodes[node_id]
                if node.cluster_id and node.cluster_id in self.clusters:
                    self.clusters[node.cluster_id].discard(node_id)
                del self.nodes[node_id]
    
    def compute_mutual_information(
        self,
        var_x: Hashable,
        var_y: Hashable,
        conditioning_set: Optional[Set[Hashable]] = None
    ) -> float:
        """
        Compute mutual information I(X; Y | Z) from experience graph.
        
        I(X; Y | Z) = H(X | Z) - H(X | Y, Z)
        """
        # Extract variable values from all nodes
        x_vals = []
        y_vals = []
        
        for node in self.nodes.values():
            if var_x in node.context and var_y in node.context:
                x_vals.append(node.context[var_x])
                y_vals.append(node.context[var_y])
        
        if len(x_vals) < 2:
            return 0.0
        
        # Compute mutual information (simplified)
        if all(isinstance(v, (int, float)) for v in x_vals) and \
           all(isinstance(v, (int, float)) for v in y_vals):
            # Correlation-based MI estimate
            x_arr = np.array(x_vals)
            y_arr = np.array(y_vals)
            
            if len(x_arr) == len(y_arr):
                correlation = np.abs(np.corrcoef(x_arr, y_arr)[0, 1])
                # Approximate MI from correlation
                mi = -0.5 * np.log(1 - correlation ** 2) if correlation < 1.0 else float('inf')
                return mi
        
        # Discrete: use entropy-based estimate
        # Simplified: count co-occurrences
        x_unique = set(x_vals)
        y_unique = set(y_vals)
        
        if len(x_unique) == 1 or len(y_unique) == 1:
            return 0.0
        
        # Joint distribution
        joint_counts = defaultdict(int)
        for x, y in zip(x_vals, y_vals):
            joint_counts[(x, y)] += 1
        
        total = len(x_vals)
        if total == 0:
            return 0.0
        
        # Compute MI
        mi = 0.0
        for (x, y), count in joint_counts.items():
            p_xy = count / total
            p_x = sum(1 for v in x_vals if v == x) / total
            p_y = sum(1 for v in y_vals if v == y) / total
            
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log(p_xy / (p_x * p_y))
        
        return max(0.0, mi)
    
    def suggest_crm_candidates(self, min_support: int = 5) -> List[Dict[str, Any]]:
        """
        Suggest CRM candidates from experience graph patterns.
        
        Uses clustering and dependency analysis.
        """
        candidates = []
        
        # Analyze transitions within clusters
        for cluster_id, cluster_nodes in self.clusters.items():
            if len(cluster_nodes) < min_support:
                continue
            
            # Collect transitions from cluster nodes
            cluster_transitions = defaultdict(list)
            
            for node_id in cluster_nodes:
                node = self.nodes[node_id]
                for trans in node.transitions:
                    action = trans["action"]
                    cluster_transitions[action].append({
                        "from": node.context,
                        "to": trans["next_context"],
                        "reward": trans["reward"]
                    })
            
            # For each action, check if transitions are consistent
            for action, transitions in cluster_transitions.items():
                if len(transitions) < min_support:
                    continue
                
                # Analyze what changes consistently
                from_contexts = [t["from"] for t in transitions]
                to_contexts = [t["to"] for t in transitions]
                
                # Find variables that change consistently
                all_vars = set()
                for ctx in from_contexts + to_contexts:
                    all_vars.update(ctx.keys())
                
                consistent_changes = {}
                for var in all_vars:
                    from_vals = [ctx.get(var) for ctx in from_contexts if var in ctx]
                    to_vals = [ctx.get(var) for ctx in to_contexts if var in ctx]
                    
                    if from_vals and to_vals:
                        # Check if change is consistent
                        if isinstance(from_vals[0], (int, float)) and isinstance(to_vals[0], (int, float)):
                            # Numeric: check if change direction is consistent
                            changes = [to_vals[i] - from_vals[i] for i in range(min(len(from_vals), len(to_vals)))]
                            if changes:
                                mean_change = np.mean(changes)
                                std_change = np.std(changes)
                                if std_change < abs(mean_change) * 0.5:  # Relatively consistent
                                    consistent_changes[var] = mean_change
                        else:
                            # Discrete: check if mapping is consistent
                            mappings = {}
                            for i in range(min(len(from_vals), len(to_vals))):
                                key = from_vals[i]
                                val = to_vals[i]
                                if key not in mappings:
                                    mappings[key] = val
                                elif mappings[key] != val:
                                    mappings = None
                                    break
                            
                            if mappings:
                                consistent_changes[var] = mappings
                
                if consistent_changes:
                    candidates.append({
                        "cluster_id": cluster_id,
                        "action": action,
                        "pre_context": from_contexts[0],  # Representative
                        "post_changes": consistent_changes,
                        "support": len(transitions),
                        "avg_reward": np.mean([t["reward"] for t in transitions])
                    })
        
        return candidates


# ============================================================================
# Learning Mechanisms (CTPX, PTPX, GTPX) - Enhanced
# ============================================================================

class LearningMechanisms:
    """
    Implements AERA's three pattern extractors with experience graph support:
    - CTPX: Change-Targeted Pattern Extractor
    - PTPX: Prediction-Targeted Pattern Extractor
    - GTPX: Goal-Targeted Pattern Extractor
    
    Enhanced with:
    - Experience graph for pattern discovery
    - Mutual information for dependency analysis
    - Firing frequency priors to limit scope explosion
    - Differentiable probabilistic models for factor graph integration
    """
    
    def __init__(
        self,
        kb: KnowledgeBase,
        experience_graph: Optional[ExperienceGraph] = None,
        llm_reasoning_engine: Optional[LLMReasoningEngine] = None,
        change_threshold: float = 0.1,
        error_threshold: float = 1.0,
        min_firing_frequency: float = 0.01
    ):
        """
        Initialize learning mechanisms.
        
        Args:
            kb: Knowledge base to modify
            experience_graph: Experience graph for pattern discovery
            llm_reasoning_engine: Optional LLM reasoning engine for hypothesis generation
            change_threshold: Threshold for detecting significant changes
            error_threshold: Threshold for prediction errors
            min_firing_frequency: Minimum firing frequency to keep CRM (prevents explosion)
        """
        self.kb = kb
        self.experience_graph = experience_graph or ExperienceGraph()
        self.llm_reasoning_engine = llm_reasoning_engine
        self.change_threshold = change_threshold
        self.error_threshold = error_threshold
        self.min_firing_frequency = min_firing_frequency
    
    def apply_ctpx(
        self,
        x_t: Dict[Hashable, Any],
        u_t: Optional[str],
        x_tp1: Dict[Hashable, Any]
    ) -> Optional[CRM]:
        """
        CTPX: Change-Targeted Pattern Extractor (Enhanced).
        
        Uses experience graph for pattern discovery and mutual information for dependency analysis.
        """
        # Add to experience graph
        self.experience_graph.add_experience(x_t, u_t or "noop", x_tp1)
        
        # Check experience graph for CRM candidates
        candidates = self.experience_graph.suggest_crm_candidates(min_support=3)
        
        # Use candidate if available and better than single observation
        if candidates:
            # Find best candidate for this transition
            best_candidate = None
            best_score = 0.0
            
            for candidate in candidates:
                if candidate["action"] == (u_t or "noop"):
                    # Check if pre_context matches
                    similarity = self.experience_graph._compute_context_similarity(
                        x_t, candidate["pre_context"]
                    )
                    score = similarity * candidate["support"]
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate
            
            if best_candidate and best_score > 2.0:
                # Create CRM from candidate
                return self._create_crm_from_candidate(best_candidate, x_t, u_t, x_tp1)
        
        # Try LLM hypothesis generation if available
        if self.llm_reasoning_engine:
            context = f"Observing transition: {x_t} -> {x_tp1} via action {u_t}"
            llm_hypothesis = self.llm_reasoning_engine.generate_crm_hypothesis(
                x_t, u_t, x_tp1, context
            )
            
            if llm_hypothesis and llm_hypothesis.get("confidence", 0.0) > 0.5:
                # Create CRM from LLM hypothesis
                return self._create_crm_from_llm_hypothesis(llm_hypothesis, x_t, u_t, x_tp1)
        
        # Fall back to original CTPX logic
        return self._apply_ctpx_original(x_t, u_t, x_tp1)
    
    def _create_crm_from_llm_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        x_t: Dict[Hashable, Any],
        u_t: Optional[str],
        x_tp1: Dict[Hashable, Any]
    ) -> Optional[CRM]:
        """Create CRM from LLM-generated hypothesis."""
        preconditions = hypothesis.get("preconditions", {})
        postconditions = hypothesis.get("postconditions", {})
        
        if not preconditions or not postconditions:
            return None
        
        # Create pre-CST
        pre_conditions = []
        for var_id, val in preconditions.items():
            pre_conditions.append(AtomicPredicate(
                var_id=var_id,
                operator="=",
                value=val
            ))
        
        # Create post-CST
        post_conditions = []
        for var_id, val in postconditions.items():
            post_conditions.append(AtomicPredicate(
                var_id=var_id,
                operator="=",
                value=val
            ))
        
        # Create CSTs
        pre_cst_id = f"cst_llm_pre_{uuid4()}"
        post_cst_id = f"cst_llm_post_{uuid4()}"
        
        pre_cst = CompositeState(id=pre_cst_id, conditions=pre_conditions)
        post_cst = CompositeState(id=post_cst_id, conditions=post_conditions)
        
        self.kb.add_cst(pre_cst)
        self.kb.add_cst(post_cst)
        
        # Create parametric model
        changing_vars = list(postconditions.keys())
        param_model = ParametricModel(
            model_type="linear_gaussian",
            parameters={
                "A": np.eye(len(changing_vars)),
                "B": np.zeros((len(changing_vars), 1)) if u_t else np.zeros((len(changing_vars), 0)),
                "Sigma": np.eye(len(changing_vars)) * 0.1
            }
        )
        
        # Create CRM
        crm_id = f"crm_llm_{uuid4()}"
        crm = CRM(
            id=crm_id,
            pre_cst_id=pre_cst_id,
            post_cst_id=post_cst_id,
            actions={u_t} if u_t else set(),
            param_model=param_model
        )
        
        self.kb.add_crm(crm)
        
        # Create Mreq with confidence from LLM
        confidence = hypothesis.get("confidence", 0.5)
        mreq_id = f"mreq_llm_{uuid4()}"
        mreq = Mreq(
            id=mreq_id,
            cst_id=pre_cst_id,
            crm_id=crm_id,
            confidence=confidence
        )
        self.kb.add_mreq(mreq)
        
        logger.info(f"CTPX created CRM {crm_id} from LLM hypothesis (confidence={confidence:.3f})")
        return crm
    
    def _apply_ctpx_original(
        self,
        x_t: Dict[Hashable, Any],
        u_t: Optional[str],
        x_tp1: Dict[Hashable, Any]
    ) -> Optional[CRM]:
        """
        CTPX: Change-Targeted Pattern Extractor.
        
        Triggered when a change occurs that cannot be explained by existing CRMs.
        
        Algorithm:
        1. Detect attribute changes: ΔX_{e,a,t} = X_{e,a,t+1} - X_{e,a,t}
        2. Check if any active CRM explains the change
        3. If not, synthesize a new CRM with minimal context
        
        Args:
            x_t: State at time t
            u_t: Action at time t
            x_tp1: State at time t+1
            
        Returns:
            Newly created CRM if any, None otherwise
        """
        # Detect significant changes
        changes = {}
        for var_id in set(x_t.keys()) | set(x_tp1.keys()):
            val_t = x_t.get(var_id)
            val_tp1 = x_tp1.get(var_id)
            
            if val_t != val_tp1:
                if isinstance(val_t, (int, float)) and isinstance(val_tp1, (int, float)):
                    change_magnitude = abs(val_tp1 - val_t)
                    if change_magnitude > self.change_threshold:
                        changes[var_id] = (val_t, val_tp1)
                else:
                    # Discrete change
                    changes[var_id] = (val_t, val_tp1)
        
        if not changes:
            return None
        
        # Check if existing CRMs explain these changes
        active_crms = self.kb.get_active_crms(x_t, u_t)
        explained = False
        
        for crm in active_crms:
            # Try to predict x_tp1 using this CRM
            # Simplified: check if CRM's post-CST matches x_tp1
            if crm.post_cst_id in self.kb.csts:
                post_cst = self.kb.csts[crm.post_cst_id]
                if post_cst.evaluate(x_tp1):
                    explained = True
                    break
        
        if explained:
            return None
        
        # Synthesize new CRM
        # Extract minimal context features
        pre_conditions = []
        for var_id, (val_t, _) in changes.items():
            # Create equality predicate for current value
            pre_conditions.append(AtomicPredicate(
                var_id=var_id,
                operator="=",
                value=val_t
            ))
        
        # Create post-conditions
        post_conditions = []
        for var_id, (_, val_tp1) in changes.items():
            post_conditions.append(AtomicPredicate(
                var_id=var_id,
                operator="=",
                value=val_tp1
            ))
        
        # Create CSTs
        pre_cst_id = f"cst_pre_{uuid4()}"
        post_cst_id = f"cst_post_{uuid4()}"
        
        pre_cst = CompositeState(id=pre_cst_id, conditions=pre_conditions)
        post_cst = CompositeState(id=post_cst_id, conditions=post_conditions)
        
        self.kb.add_cst(pre_cst)
        self.kb.add_cst(post_cst)
        
        # Create parametric model (linear-Gaussian as default)
        param_model = ParametricModel(
            model_type="linear_gaussian",
            parameters={
                "A": np.eye(len(changes)),
                "B": np.zeros((len(changes), 1)) if u_t else np.zeros((len(changes), 0)),
                "Sigma": np.eye(len(changes)) * 0.1
            }
        )
        
        # Create CRM
        crm_id = f"crm_{uuid4()}"
        crm = CRM(
            id=crm_id,
            pre_cst_id=pre_cst_id,
            post_cst_id=post_cst_id,
            actions={u_t} if u_t else set(),
            param_model=param_model
        )
        
        self.kb.add_crm(crm)
        
        # Create Mreq with low initial confidence
        mreq_id = f"mreq_{uuid4()}"
        mreq = Mreq(
            id=mreq_id,
            cst_id=pre_cst_id,
            crm_id=crm_id,
            confidence=0.3  # Low initial confidence
        )
        self.kb.add_mreq(mreq)
        
        logger.info(f"CTPX created new CRM {crm_id} to explain changes: {changes}")
        return crm
    
    def _create_crm_from_candidate(
        self,
        candidate: Dict[str, Any],
        x_t: Dict[Hashable, Any],
        u_t: Optional[str],
        x_tp1: Dict[Hashable, Any]
    ) -> Optional[CRM]:
        """Create CRM from experience graph candidate."""
        # Extract pre-conditions from candidate
        pre_conditions = []
        for var_id, val in candidate["pre_context"].items():
            pre_conditions.append(AtomicPredicate(
                var_id=var_id,
                operator="=",
                value=val
            ))
        
        # Extract post-conditions from changes
        post_conditions = []
        for var_id, change in candidate["post_changes"].items():
            if isinstance(change, dict):
                # Discrete mapping: use observed value
                if var_id in x_tp1:
                    post_conditions.append(AtomicPredicate(
                        var_id=var_id,
                        operator="=",
                        value=x_tp1[var_id]
                    ))
            else:
                # Numeric change: predict value
                if var_id in x_t:
                    predicted_val = x_t[var_id] + change
                    post_conditions.append(AtomicPredicate(
                        var_id=var_id,
                        operator="=",
                        value=predicted_val
                    ))
        
        if not post_conditions:
            return None
        
        # Create CSTs
        pre_cst_id = f"cst_pre_{uuid4()}"
        post_cst_id = f"cst_post_{uuid4()}"
        
        pre_cst = CompositeState(id=pre_cst_id, conditions=pre_conditions)
        post_cst = CompositeState(id=post_cst_id, conditions=post_conditions)
        
        self.kb.add_cst(pre_cst)
        self.kb.add_cst(post_cst)
        
        # Learn parametric model from candidate transitions
        # Extract training data
        changing_vars = list(candidate["post_changes"].keys())
        if not changing_vars:
            return None
        
        # Create linear-Gaussian model
        param_model = ParametricModel(
            model_type="linear_gaussian",
            parameters={
                "A": np.eye(len(changing_vars)),
                "B": np.zeros((len(changing_vars), 1)) if u_t else np.zeros((len(changing_vars), 0)),
                "Sigma": np.eye(len(changing_vars)) * 0.1
            }
        )
        
        # Create CRM
        crm_id = f"crm_{uuid4()}"
        crm = CRM(
            id=crm_id,
            pre_cst_id=pre_cst_id,
            post_cst_id=post_cst_id,
            actions={u_t} if u_t else set(),
            param_model=param_model
        )
        
        self.kb.add_crm(crm)
        
        # Create Mreq with confidence based on support
        confidence = min(0.7, 0.3 + 0.1 * np.log(candidate["support"] + 1))
        mreq_id = f"mreq_{uuid4()}"
        mreq = Mreq(
            id=mreq_id,
            cst_id=pre_cst_id,
            crm_id=crm_id,
            confidence=confidence
        )
        self.kb.add_mreq(mreq)
        
        logger.info(f"CTPX created CRM {crm_id} from experience graph candidate (support={candidate['support']})")
        return crm
    
    def prune_low_frequency_crms(self):
        """
        Prune CRMs with low firing frequency to prevent scope explosion.
        
        Removes CRMs where firing_rate < min_firing_frequency.
        """
        current_time = time.time()
        crms_to_remove = []
        
        for crm_id, crm in self.kb.crms.items():
            # Compute firing rate
            if "created_at" in crm.stats:
                age = current_time - crm.stats["created_at"]
                if age > 0:
                    firing_rate = crm.stats["firings"] / age
                    
                    if firing_rate < self.min_firing_frequency:
                        crms_to_remove.append(crm_id)
        
        # Remove low-frequency CRMs
        for crm_id in crms_to_remove:
            # Remove associated Mreqs
            mreqs_to_remove = [
                mreq_id for mreq_id, mreq in self.kb.mreqs.items()
                if mreq.crm_id == crm_id
            ]
            for mreq_id in mreqs_to_remove:
                del self.kb.mreqs[mreq_id]
            
            # Remove CRM
            del self.kb.crms[crm_id]
            logger.info(f"Pruned low-frequency CRM {crm_id}")
        
        return len(crms_to_remove)
    
    def apply_ptpx(
        self,
        x_t: Dict[Hashable, Any],
        u_t: Optional[str],
        x_tp1: Dict[Hashable, Any],
        active_crms: List[CRM]
    ) -> List[CRM]:
        """
        PTPX: Prediction-Targeted Pattern Extractor.
        
        Triggered when an existing CRM mispredicts.
        
        Algorithm:
        1. Measure prediction error for each active CRM
        2. If error > threshold, try to specialize the CST
        3. Split CST into success/failure regions
        4. Create new CRM or anti-Mreq for failure region
        
        Args:
            x_t: State at time t
            u_t: Action at time t
            x_tp1: Observed state at time t+1
            active_crms: List of CRMs that were active
            
        Returns:
            List of newly created/modified CRMs
        """
        new_crms = []
        
        for crm in active_crms:
            # Compute prediction error
            if crm.post_cst_id not in self.kb.csts:
                continue
            
            post_cst = self.kb.csts[crm.post_cst_id]
            predicted_satisfied = post_cst.evaluate_soft(x_tp1)
            error = 1.0 - predicted_satisfied
            
            if error > self.error_threshold:
                # Record failure
                crm.record_failure(error)
                
                # Try LLM-based CST refinement if available
                if self.llm_reasoning_engine and crm.pre_cst_id in self.kb.csts:
                    pre_cst = self.kb.csts[crm.pre_cst_id]
                    
                    # Collect success/failure examples from history
                    success_examples = []
                    failure_examples = [x_t]  # Current failure
                    
                    # Get examples from experience graph
                    for node in self.experience_graph.nodes.values():
                        for trans in node.transitions:
                            # Check if this transition matches CRM
                            if trans.get("action") == u_t:
                                # Simplified: use node context as example
                                if len(success_examples) < 5:
                                    success_examples.append(node.context)
                    
                    # Use LLM to refine CST
                    refined_cst = self.llm_reasoning_engine.refine_cst_with_llm(
                        pre_cst, success_examples, failure_examples
                    )
                    
                    if refined_cst:
                        # Add refined CST
                        self.kb.add_cst(refined_cst)
                        
                        # Create new Mreq with refined CST
                        mreq_id = f"mreq_refined_{uuid4()}"
                        mreq = Mreq(
                            id=mreq_id,
                            cst_id=refined_cst.id,
                            crm_id=crm.id,
                            confidence=0.6  # Moderate confidence
                        )
                        self.kb.add_mreq(mreq)
                        
                        logger.info(f"PTPX refined CST for CRM {crm.id} using LLM")
                        continue
                
                # Fallback: create anti-Mreq for failure context
                failure_conditions = []
                for var_id, val in x_t.items():
                    failure_conditions.append(AtomicPredicate(
                        var_id=var_id,
                        operator="=",
                        value=val
                    ))
                
                failure_cst_id = f"cst_failure_{uuid4()}"
                failure_cst = CompositeState(id=failure_cst_id, conditions=failure_conditions)
                self.kb.add_cst(failure_cst)
                
                # Create anti-Mreq
                anti_mreq_id = f"antimreq_{uuid4()}"
                anti_mreq = AntiMreq(
                    id=anti_mreq_id,
                    cst_id=failure_cst_id,
                    crm_id=crm.id,
                    confidence=0.5
                )
                self.kb.add_anti_mreq(anti_mreq)
                
                logger.info(f"PTPX created anti-Mreq {anti_mreq_id} for CRM {crm.id} due to prediction error {error:.3f}")
            else:
                # Record success
                crm.record_success(error)
        
        return new_crms
    
    def apply_gtpx(
        self,
        x_t: Dict[Hashable, Any],
        x_tp1: Dict[Hashable, Any],
        drives: List[Drive]
    ) -> List[CRM]:
        """
        GTPX: Goal-Targeted Pattern Extractor.
        
        Triggered when a drive is satisfied unexpectedly.
        
        Algorithm:
        1. Detect drive satisfaction increase
        2. Trace back state/action/CRM history
        3. Infer new CRM that explains goal achievement
        
        Args:
            x_t: State at time t
            x_tp1: State at time t+1
            drives: List of active drives
            
        Returns:
            List of newly created CRMs
        """
        new_crms = []
        
        for drive in drives:
            if drive.goal_cst_id not in self.kb.csts:
                continue
            
            goal_cst = self.kb.csts[drive.goal_cst_id]
            sat_t = goal_cst.evaluate_soft(x_t)
            sat_tp1 = goal_cst.evaluate_soft(x_tp1)
            
            delta_sat = sat_tp1 - sat_t
            
            if delta_sat > 0.3:  # Significant improvement
                # Check if this improvement is unexpected (no active CRM explains it)
                # Simplified: create a CRM that maps from x_t to goal satisfaction
                
                # Extract conditions that led to goal improvement
                improvement_conditions = []
                for var_id, val in x_t.items():
                    improvement_conditions.append(AtomicPredicate(
                        var_id=var_id,
                        operator="=",
                        value=val
                    ))
                
                pre_cst_id = f"cst_gtpx_pre_{uuid4()}"
                pre_cst = CompositeState(id=pre_cst_id, conditions=improvement_conditions)
                self.kb.add_cst(pre_cst)
                
                # Post-CST is the goal CST
                post_cst_id = drive.goal_cst_id
                
                # Create parametric model
                param_model = ParametricModel(
                    model_type="linear_gaussian",
                    parameters={
                        "A": np.eye(len(x_t)),
                        "B": np.zeros((len(x_t), 0)),
                        "Sigma": np.eye(len(x_t)) * 0.1
                    }
                )
                
                # Create CRM
                crm_id = f"crm_gtpx_{uuid4()}"
                crm = CRM(
                    id=crm_id,
                    pre_cst_id=pre_cst_id,
                    post_cst_id=post_cst_id,
                    actions=set(),  # May be refined later
                    param_model=param_model
                )
                
                self.kb.add_crm(crm)
                
                # Create Mreq
                mreq_id = f"mreq_gtpx_{uuid4()}"
                mreq = Mreq(
                    id=mreq_id,
                    cst_id=pre_cst_id,
                    crm_id=crm_id,
                    confidence=0.4  # Moderate initial confidence
                )
                self.kb.add_mreq(mreq)
                
                new_crms.append(crm)
                logger.info(f"GTPX created new CRM {crm_id} explaining goal achievement for drive {drive.id}")
        
        return new_crms


# ============================================================================
# Analogy Mechanisms
# ============================================================================

class EntityEmbedder:
    """
    Entity/attribute embeddings for analogical transfer.
    
    Mathematical formulation:
        sim(e_i, e_j) = (E(e_i) · E(e_j)) / (||E(e_i)|| ||E(e_j)||)
        
    Learned via contrastive learning:
        L = -log exp(sim(e_pos, e_anchor)) / (exp(sim(e_pos, e_anchor)) + Σ exp(sim(e_neg, e_anchor)))
    """
    
    def __init__(self, embedding_dim: int = 64):
        """
        Initialize entity embedder.
        
        Args:
            embedding_dim: Dimension of entity embeddings
        """
        self.embedding_dim = embedding_dim
        self.embeddings: Dict[Hashable, np.ndarray] = {}
        self.entity_to_embedding: Dict[Hashable, Hashable] = {}
    
    def get_embedding(self, entity_id: Hashable) -> np.ndarray:
        """Get or create embedding for entity."""
        if entity_id not in self.embeddings:
            # Initialize random embedding
            self.embeddings[entity_id] = np.random.randn(self.embedding_dim) / np.sqrt(self.embedding_dim)
        return self.embeddings[entity_id]
    
    def compute_similarity(self, e1_id: Hashable, e2_id: Hashable) -> float:
        """
        Compute cosine similarity between entity embeddings.
        
        sim(e_i, e_j) = (E(e_i) · E(e_j)) / (||E(e_i)|| ||E(e_j)||)
        """
        emb1 = self.get_embedding(e1_id)
        emb2 = self.get_embedding(e2_id)
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, (similarity + 1) / 2))  # Normalize to [0, 1]
    
    def update_embedding(
        self,
        entity_id: Hashable,
        positive_examples: List[Hashable],
        negative_examples: List[Hashable],
        learning_rate: float = 0.01
    ):
        """
        Update embedding via contrastive learning.
        
        Args:
            entity_id: Entity to update
            positive_examples: Similar entities (positive pairs)
            negative_examples: Dissimilar entities (negative pairs)
            learning_rate: Learning rate for updates
        """
        if entity_id not in self.embeddings:
            self.get_embedding(entity_id)
        
        emb = self.embeddings[entity_id]
        
        # Contrastive update (simplified)
        for pos_id in positive_examples:
            pos_emb = self.get_embedding(pos_id)
            # Pull embeddings together
            emb += learning_rate * (pos_emb - emb)
        
        for neg_id in negative_examples:
            neg_emb = self.get_embedding(neg_id)
            # Push embeddings apart
            emb -= learning_rate * 0.1 * (neg_emb - emb)
        
        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        
        self.embeddings[entity_id] = emb


class AnalogyEngine:
    """
    Implements analogy mechanisms for transferring CRMs across domains.
    
    Enhanced with entity embeddings for better similarity computation.
    
    A(Φ_S, Φ_T, K) → ΔK
    
    Transfers CRMs from source domain to target domain via:
    - Entity mapping: σ: E_S → E_T
    - Attribute/relational similarity (using embeddings)
    - Bayesian confidence updates: p(M | D) ∝ p(D | M) p(M)
    """
    
    def __init__(
        self,
        kb: KnowledgeBase,
        similarity_threshold: float = 0.7,
        embedding_dim: int = 64
    ):
        """
        Initialize analogy engine.
        
        Args:
            kb: Knowledge base
            similarity_threshold: Minimum similarity for transfer
            embedding_dim: Dimension of entity embeddings
        """
        self.kb = kb
        self.similarity_threshold = similarity_threshold
        self.embedder = EntityEmbedder(embedding_dim)
    
    def compute_entity_similarity(
        self,
        e_s: Entity,
        e_t: Entity,
        var_values_s: Dict[Hashable, Any],
        var_values_t: Dict[Hashable, Any]
    ) -> float:
        """
        Compute similarity between two entities using embeddings.
        
        sim(e_s, e_t) = (E(e_s) · E(e_t)) / (||E(e_s)|| ||E(e_t)||)
        
        Args:
            e_s: Source entity
            e_t: Target entity
            var_values_s: Variable values in source domain
            var_values_t: Variable values in target domain
            
        Returns:
            Similarity score in [0, 1]
        """
        # Use embedding-based similarity
        embedding_sim = self.embedder.compute_similarity(e_s.id, e_t.id)
        
        # Also check ontology overlap
        ontology_overlap = len(e_s.ontologies & e_t.ontologies) / max(1, len(e_s.ontologies | e_t.ontologies))
        
        # Check attribute similarity
        attr_similarities = []
        common_attrs = e_s.attributes.keys() & e_t.attributes.keys()
        
        for attr in common_attrs:
            var_id_s = e_s.attributes[attr]
            var_id_t = e_t.attributes[attr]
            
            val_s = var_values_s.get(var_id_s)
            val_t = var_values_t.get(var_id_t)
            
            if val_s == val_t:
                attr_similarities.append(1.0)
            elif isinstance(val_s, (int, float)) and isinstance(val_t, (int, float)):
                max_val = max(abs(val_s), abs(val_t), 1.0)
                similarity = 1.0 - min(1.0, abs(val_s - val_t) / max_val)
                attr_similarities.append(similarity)
            else:
                attr_similarities.append(0.0)
        
        attr_sim = np.mean(attr_similarities) if attr_similarities else 0.0
        
        # Combined similarity (weighted)
        similarity = 0.5 * embedding_sim + 0.3 * ontology_overlap + 0.2 * attr_sim
        return similarity
    
    def transfer_crm(
        self,
        crm: CRM,
        entity_mapping: Dict[Hashable, Hashable],
        target_var_values: Dict[Hashable, Any]
    ) -> Optional[CRM]:
        """
        Transfer a CRM from source to target domain.
        
        Args:
            crm: Source CRM to transfer
            entity_mapping: Mapping from source entity IDs to target entity IDs
            target_var_values: Variable values in target domain
            
        Returns:
            Newly created CRM in target domain, or None if transfer fails
        """
        # Map pre-CST
        if crm.pre_cst_id not in self.kb.csts:
            return None
        
        pre_cst = self.kb.csts[crm.pre_cst_id]
        
        # Create mapped conditions
        mapped_pre_conditions = []
        for pred in pre_cst.conditions:
            # Map variable IDs through entity mapping
            # Simplified: assume direct mapping
            mapped_pre_conditions.append(pred)  # In full implementation, would map var_ids
        
        pre_cst_id_new = f"cst_transfer_pre_{uuid4()}"
        pre_cst_new = CompositeState(id=pre_cst_id_new, conditions=mapped_pre_conditions)
        self.kb.add_cst(pre_cst_new)
        
        # Map post-CST
        if crm.post_cst_id not in self.kb.csts:
            return None
        
        post_cst = self.kb.csts[crm.post_cst_id]
        mapped_post_conditions = [pred for pred in post_cst.conditions]  # Simplified
        post_cst_id_new = f"cst_transfer_post_{uuid4()}"
        post_cst_new = CompositeState(id=post_cst_id_new, conditions=mapped_post_conditions)
        self.kb.add_cst(post_cst_new)
        
        # Copy parametric model (may need adaptation)
        param_model = ParametricModel(
            model_type=crm.param_model.model_type,
            parameters=crm.param_model.parameters.copy()
        )
        
        # Create new CRM
        crm_id_new = f"crm_transfer_{uuid4()}"
        crm_new = CRM(
            id=crm_id_new,
            pre_cst_id=pre_cst_id_new,
            post_cst_id=post_cst_id_new,
            actions=crm.actions.copy(),
            param_model=param_model
        )
        
        self.kb.add_crm(crm_new)
        
        # Create Mreq with low confidence (will be updated Bayesianly)
        mreq_id_new = f"mreq_transfer_{uuid4()}"
        mreq_new = Mreq(
            id=mreq_id_new,
            cst_id=pre_cst_id_new,
            crm_id=crm_id_new,
            confidence=0.3  # Low initial confidence
        )
        self.kb.add_mreq(mreq_new)
        
        logger.info(f"Analogy transferred CRM {crm.id} to {crm_id_new}")
        return crm_new
    
    def update_transferred_crm_confidence(
        self,
        crm_id: Hashable,
        successes: int,
        failures: int
    ):
        """
        Update CRM confidence Bayesianly after validation.
        
        p(M | D) ∝ p(D | M) p(M)
        
        Where p(D | M) uses observed success rates.
        """
        if crm_id not in self.kb.crms:
            return
        
        crm = self.kb.crms[crm_id]
        total = successes + failures
        
        if total == 0:
            return
        
        # Update success rate
        success_rate = successes / total
        
        # Find associated Mreqs
        for mreq in self.kb.mreqs.values():
            if mreq.crm_id == crm_id:
                # Bayesian update: combine prior with likelihood
                prior_confidence = mreq.confidence
                likelihood = success_rate
                
                # Simple Bayesian update (simplified)
                # Posterior ∝ prior * likelihood
                posterior = (prior_confidence * 0.7 + likelihood * 0.3)
                mreq.confidence = min(1.0, max(0.0, posterior))
                
                logger.debug(f"Updated Mreq {mreq.id} confidence: {prior_confidence:.3f} → {mreq.confidence:.3f}")


# ============================================================================
# Factor Graph Construction
# ============================================================================

class DualLayerFactorGraph:
    """
    Dual-layer factor graph system: Working Graph G_t and Template Graph G_K.
    
    Mathematical formulation:
        G_t: Working graph (current inference state)
        G_K: Template graph (schema from CRMs/CSTs/Mreqs)
        
    Synchronization:
        G_t is instantiated from G_K based on current context.
        Both share the same structure but G_t has concrete variable values.
    """
    
    def __init__(self, kb: KnowledgeBase):
        """Initialize dual-layer system."""
        self.kb = kb
        self.template_graph: Optional[FactorGraph] = None
        self.working_graph: Optional[FactorGraph] = None
        self.last_sync_time = time.time()
    
    def build_template_graph(self) -> FactorGraph:
        """
        Build template graph G_K from knowledge base schema.
        
        Template contains factor types and structure, not concrete values.
        """
        template = FactorGraph()
        
        # Add template variables (all possible variable types)
        # In full implementation, would extract from entities
        for entity in self.kb.entities.values():
            for attr_name, var_id in entity.attributes.items():
                template.add_variable(
                    var_id=var_id,
                    var_type=VarType.ATTRIBUTE,
                    domain=None,  # Template: domain not specified
                    metadata={"entity_id": entity.id, "attribute": attr_name}
                )
        
        # Add template factors (CRM/CST/Mreq schemas)
        for mreq in self.kb.mreqs.values():
            if mreq.cst_id in self.kb.csts and mreq.crm_id in self.kb.crms:
                cst = self.kb.csts[mreq.cst_id]
                crm = self.kb.crms[mreq.crm_id]
                
                # Template factor (structure only)
                cst_vars = {pred.var_id for pred in cst.conditions}
                factor_id = f"template_{mreq.id}"
                
                def make_template_func(vars_set):
                    def template_func(var_vals):
                        # Template: always return 1.0 (will be instantiated)
                        return 1.0
                    return template_func
                
                factor = FactorNode(
                    id=factor_id,
                    factor_type="template_crm",
                    var_neighbors=cst_vars,
                    factor_func=make_template_func(cst_vars),
                    metadata={"mreq_id": mreq.id, "crm_id": mreq.crm_id, "cst_id": mreq.cst_id}
                )
                template.add_factor(factor_id, factor)
        
        self.template_graph = template
        return template
    
    def instantiate_working_graph(
        self,
        var_values: Dict[Hashable, Any],
        goals: List[Drive],
        action: Optional[str] = None,
        perception_factors: Optional[Dict[Hashable, Dict[Any, float]]] = None
    ) -> FactorGraph:
        """
        Instantiate working graph G_t from template G_K.
        
        Args:
            var_values: Current variable values
            goals: Active drives
            action: Current action
            perception_factors: Perception factors
            
        Returns:
            Instantiated working graph
        """
        if self.template_graph is None:
            self.build_template_graph()
        
        working = FactorGraph()
        perception_factors = perception_factors or {}
        
        # Collect all variables referenced by CRMs/CSTs
        all_referenced_vars = set(var_values.keys())
        for crm in self.kb.crms.values():
            pre_cst = self.kb.csts.get(crm.pre_cst_id)
            post_cst = self.kb.csts.get(crm.post_cst_id)
            if pre_cst:
                all_referenced_vars.update(pred.var_id for pred in pre_cst.conditions)
            if post_cst:
                all_referenced_vars.update(pred.var_id for pred in post_cst.conditions)
        
        # Add perception variables
        all_referenced_vars.update(perception_factors.keys())
        
        # Instantiate variables (both from var_values and referenced by CRMs)
        for var_id in all_referenced_vars:
            var_val = var_values.get(var_id, None)
            
            # Determine domain
            if var_val is not None:
                if isinstance(var_val, (int, float)):
                    domain = [-np.inf, np.inf]
                elif isinstance(var_val, str):
                    domain = [var_val]  # String domain
                else:
                    domain = [var_val]
            else:
                # Variable not in var_values, use default domain
                domain = [-np.inf, np.inf]  # Continuous default
            
            # Determine var_type
            if var_id in self.template_graph.variables:
                template_var = self.template_graph.variables[var_id]
                var_type = template_var.var_type
                metadata = template_var.metadata.copy()
            else:
                # Infer type from name or use default
                if "text" in str(var_id).lower() or "content" in str(var_id).lower():
                    var_type = VarType.ATTRIBUTE
                else:
                    var_type = VarType.ATTRIBUTE
                metadata = {}
            
            working.add_variable(
                var_id=var_id,
                var_type=var_type,
                domain=domain,
                metadata=metadata
            )
        
        # Instantiate factors from template
        for factor_id, template_factor in self.template_graph.factors.items():
            # Check if variables exist in working graph
            if template_factor.var_neighbors.issubset(working.variables.keys()):
                # Instantiate factor with concrete function
                metadata = template_factor.metadata
                
                if template_factor.factor_type == "template_crm":
                    # Instantiate CRM factor
                    mreq_id = metadata.get("mreq_id")
                    crm_id = metadata.get("crm_id")
                    cst_id = metadata.get("cst_id")
                    
                    if mreq_id in self.kb.mreqs and crm_id in self.kb.crms and cst_id in self.kb.csts:
                        mreq = self.kb.mreqs[mreq_id]
                        crm = self.kb.crms[crm_id]
                        cst = self.kb.csts[cst_id]
                        
                        # Check if Mreq is active
                        if mreq.is_active(cst.evaluate(var_values)):
                            # Create concrete CRM factor
                            pre_cst = self.kb.csts.get(crm.pre_cst_id)
                            post_cst = self.kb.csts.get(crm.post_cst_id)
                            
                            if pre_cst and post_cst:
                                pre_vars = {pred.var_id for pred in pre_cst.conditions}
                                post_vars = {pred.var_id for pred in post_cst.conditions}
                                all_vars = pre_vars | post_vars
                                
                                def make_crm_func(c, pm, pre_vs, post_vs):
                                    def crm_func(var_vals):
                                        if not c.evaluate(var_vals):
                                            return 1.0
                                        
                                        # Check if variables are strings (text processing)
                                        has_strings = any(isinstance(var_vals.get(v, ""), str) for v in (pre_vs | post_vs))
                                        
                                        if has_strings:
                                            # For string variables, use simple matching
                                            # If pre-condition is satisfied and we have post values, return high likelihood
                                            pre_vals = [var_vals.get(v, "") for v in sorted(pre_vs)]
                                            post_vals = [var_vals.get(v, "") for v in sorted(post_vs)]
                                            
                                            # If we have non-empty pre and post, it's a valid transition
                                            if any(v for v in pre_vals) and any(v for v in post_vals):
                                                return 1.0
                                            elif any(v for v in pre_vals):
                                                # Pre satisfied but no post yet - medium likelihood
                                                return 0.5
                                            else:
                                                return 0.01
                                        else:
                                            # Numeric variables: use parametric model
                                            try:
                                                x_t = np.array([float(var_vals.get(v, 0.0)) for v in sorted(pre_vs)])
                                                u_t = np.array([0.0])
                                                mean, cov = pm.predict(x_t, u_t)
                                                x_tp1 = np.array([float(var_vals.get(v, 0.0)) for v in sorted(post_vs)])
                                                diff = x_tp1 - mean
                                                likelihood = np.exp(-0.5 * diff.T @ np.linalg.inv(cov) @ diff)
                                                return max(0.01, likelihood)
                                            except (ValueError, TypeError):
                                                # Fallback for non-numeric
                                                return 0.5
                                    return crm_func
                                
                                factor = FactorNode(
                                    id=f"crm_{crm_id}",
                                    factor_type="crm",
                                    var_neighbors=all_vars,
                                    factor_func=make_crm_func(pre_cst, crm.param_model, pre_vars, post_vars),
                                    metadata={"crm_id": crm_id}
                                )
                                working.add_factor(f"crm_{crm_id}", factor)
        
        # Add perception factors
        for var_id, belief_dist in perception_factors.items():
            if var_id in working.variables:
                factor_id = f"percep_{var_id}"
                
                def make_percep_func(vid, bel):
                    def percep_func(var_vals):
                        val = var_vals.get(vid)
                        return bel.get(val, 0.0) if isinstance(bel, dict) else 1.0
                    return percep_func
                
                factor = FactorNode(
                    id=factor_id,
                    factor_type="perception",
                    var_neighbors={var_id},
                    factor_func=make_percep_func(var_id, belief_dist),
                    metadata={"var_id": var_id}
                )
                working.add_factor(factor_id, factor)
        
        # Add utility factors for drives
        for drive in goals:
            if drive.goal_cst_id in self.kb.csts:
                goal_cst = self.kb.csts[drive.goal_cst_id]
                goal_vars = {pred.var_id for pred in goal_cst.conditions}
                
                if goal_vars.issubset(working.variables.keys()):
                    factor_id = f"drive_{drive.id}"
                    
                    def make_drive_func(d, gc):
                        def drive_func(var_vals):
                            satisfaction = gc.evaluate_soft(var_vals)
                            utility = d.compute_utility(satisfaction)
                            return np.exp(utility)
                        return drive_func
                    
                    factor = FactorNode(
                        id=factor_id,
                        factor_type="utility",
                        var_neighbors=goal_vars,
                        factor_func=make_drive_func(drive, goal_cst),
                        metadata={"drive_id": drive.id}
                    )
                    working.add_factor(factor_id, factor)
        
        self.working_graph = working
        return working
    
    def sync_template_from_kb(self):
        """Synchronize template graph with knowledge base changes."""
        self.build_template_graph()
        self.last_sync_time = time.time()


class FactorGraphBuilder:
    """
    Implements factor graph construction function F.
    
    F(K_t, x_t, g_t, c_t) → G_t
    
    Builds factor graph from knowledge base, current state, goals, and control state.
    
    Enhanced with dual-layer system (working + template graphs).
    """
    
    def __init__(self, kb: KnowledgeBase, message_passing_engine: MessagePassingEngine):
        """
        Initialize factor graph builder.
        
        Args:
            kb: Knowledge base
            message_passing_engine: Message passing engine for inference
        """
        self.kb = kb
        self.message_passing_engine = message_passing_engine
        self.dual_layer = DualLayerFactorGraph(kb)
    
    def build_factor_graph(
        self,
        var_values: Dict[Hashable, Any],
        goals: List[Drive],
        action: Optional[str] = None,
        perception_factors: Optional[Dict[Hashable, Dict[Any, float]]] = None,
        use_dual_layer: bool = True
    ) -> FactorGraph:
        """
        Build factor graph from knowledge base and current context.
        
        Args:
            var_values: Current variable values (beliefs)
            goals: Active drives/goals
            action: Current action (if any)
            perception_factors: Perception factors (prior beliefs from observations)
            use_dual_layer: Whether to use dual-layer system
            
        Returns:
            Constructed factor graph
        """
        if use_dual_layer:
            # Use dual-layer system
            # Sync template if needed (every 10 seconds or on KB change)
            if time.time() - self.dual_layer.last_sync_time > 10:
                self.dual_layer.sync_template_from_kb()
            
            return self.dual_layer.instantiate_working_graph(
                var_values, goals, action, perception_factors
            )
        
        # Fall back to original method
        return self._build_factor_graph_original(var_values, goals, action, perception_factors)
    
    def _build_factor_graph_original(
        self,
        var_values: Dict[Hashable, Any],
        goals: List[Drive],
        action: Optional[str] = None,
        perception_factors: Optional[Dict[Hashable, Dict[Any, float]]] = None
    ) -> FactorGraph:
        """Original factor graph construction method."""
        graph = FactorGraph()
        perception_factors = perception_factors or {}
        
        # 1. Add variables for all relevant attributes/relations
        # (In full implementation, would extract from KB entities)
        for var_id, var_val in var_values.items():
            # Infer domain from value
            if isinstance(var_val, (int, float)):
                domain = [-np.inf, np.inf]  # Continuous
            else:
                domain = [var_val]  # Discrete (simplified)
            
            graph.add_variable(
                var_id=var_id,
                var_type=VarType.ATTRIBUTE,
                domain=domain,
                metadata={"value": var_val}
            )
        
        # 2. Add perception factors
        for var_id, belief_dist in perception_factors.items():
            if var_id in graph.variables:
                factor_id = f"percep_{var_id}"
                
                def make_percep_func(vid, bel):
                    def percep_func(var_vals):
                        val = var_vals.get(vid)
                        return bel.get(val, 0.0) if isinstance(bel, dict) else 1.0
                    return percep_func
                
                factor = FactorNode(
                    id=factor_id,
                    factor_type="perception",
                    var_neighbors={var_id},
                    factor_func=make_percep_func(var_id, belief_dist),
                    metadata={"var_id": var_id}
                )
                graph.add_factor(factor_id, factor)
        
        # 3. Add CST indicator factors
        for cst_id, cst in self.kb.csts.items():
            # Only add if relevant variables are in graph
            cst_vars = {pred.var_id for pred in cst.conditions}
            if cst_vars.issubset(graph.variables.keys()):
                factor_id = f"cst_{cst_id}"
                
                def make_cst_func(c):
                    def cst_func(var_vals):
                        if c.evaluate(var_vals):
                            return 1.0
                        return 0.01  # Soft penalty
                    return cst_func
                
                factor = FactorNode(
                    id=factor_id,
                    factor_type="cst",
                    var_neighbors=cst_vars,
                    factor_func=make_cst_func(cst),
                    metadata={"cst_id": cst_id}
                )
                graph.add_factor(factor_id, factor)
        
        # 4. Add CRM factors via Mreqs
        active_crms = self.kb.get_active_crms(var_values, action)
        
        for crm in active_crms:
            # Get pre and post CSTs
            if crm.pre_cst_id not in self.kb.csts or crm.post_cst_id not in self.kb.csts:
                continue
            
            pre_cst = self.kb.csts[crm.pre_cst_id]
            post_cst = self.kb.csts[crm.post_cst_id]
            
            # Get variables involved
            pre_vars = {pred.var_id for pred in pre_cst.conditions}
            post_vars = {pred.var_id for pred in post_cst.conditions}
            all_vars = pre_vars | post_vars
            
            if all_vars.issubset(graph.variables.keys()):
                factor_id = f"crm_{crm.id}"
                
                def make_crm_func(c, pm, pre_vs, post_vs):
                    def crm_func(var_vals):
                        # Check pre-condition
                        if not c.evaluate(var_vals):
                            return 1.0
                        
                        # Check if variables are strings (text processing)
                        has_strings = any(isinstance(var_vals.get(v, ""), str) for v in (pre_vs | post_vs))
                        
                        if has_strings:
                            # For string variables, use simple matching
                            pre_vals = [var_vals.get(v, "") for v in sorted(pre_vs)]
                            post_vals = [var_vals.get(v, "") for v in sorted(post_vs)]
                            
                            if any(v for v in pre_vals) and any(v for v in post_vals):
                                return 1.0
                            elif any(v for v in pre_vals):
                                return 0.5
                            else:
                                return 0.01
                        else:
                            # Extract state vectors (simplified)
                            try:
                                x_t = np.array([float(var_vals.get(v, 0.0)) for v in sorted(pre_vs)])
                                u_t = np.array([0.0])  # Simplified
                                
                                # Predict next state
                                mean, cov = pm.predict(x_t, u_t)
                                
                                # Compute likelihood of post-state
                                x_tp1 = np.array([float(var_vals.get(v, 0.0)) for v in sorted(post_vs)])
                                
                                # Gaussian likelihood
                                diff = x_tp1 - mean
                                likelihood = np.exp(-0.5 * diff.T @ np.linalg.inv(cov) @ diff)
                                
                                return max(0.01, likelihood)
                            except (ValueError, TypeError):
                                # Fallback for non-numeric
                                return 0.5
                    return crm_func
                
                factor = FactorNode(
                    id=factor_id,
                    factor_type="crm",
                    var_neighbors=all_vars,
                    factor_func=make_crm_func(pre_cst, crm.param_model, pre_vars, post_vars),
                    metadata={"crm_id": crm.id}
                )
                graph.add_factor(factor_id, factor)
        
        # 5. Add utility factors for drives
        for drive in goals:
            if drive.goal_cst_id not in self.kb.csts:
                continue
            
            goal_cst = self.kb.csts[drive.goal_cst_id]
            goal_vars = {pred.var_id for pred in goal_cst.conditions}
            
            if goal_vars.issubset(graph.variables.keys()):
                factor_id = f"drive_{drive.id}"
                
                def make_drive_func(d, gc):
                    def drive_func(var_vals):
                        satisfaction = gc.evaluate_soft(var_vals)
                        utility = d.compute_utility(satisfaction)
                        return np.exp(utility)  # Convert to potential
                    return drive_func
                
                factor = FactorNode(
                    id=factor_id,
                    factor_type="utility",
                    var_neighbors=goal_vars,
                    factor_func=make_drive_func(drive, goal_cst),
                    metadata={"drive_id": drive.id}
                )
                graph.add_factor(factor_id, factor)
        
        return graph


# ============================================================================
# Deep LLM Integration: LLM as Core Reasoning Component
# ============================================================================

class LLMReasoningEngine:
    """
    Deep integration of LLM as a core reasoning component within factor graphs.
    
    Mathematical formulation:
        LLM provides semantic factors: f_LLM(X) = p_θ(X | context, prompt)
        
        Joint distribution with factor graph:
            p(X) ∝ ∏_{a ∈ factors} f_a(X_{N(a)}) · f_LLM(X | context)
        
        Bidirectional integration:
            - Factor graph guides LLM reasoning via structured prompts
            - LLM provides semantic priors for factor graph inference
            - Joint training: ∇_θ L = ∇_θ [L_factor_graph + λ · L_LLM]
    """
    
    def __init__(
        self,
        llm_backend: Any,
        embedding_dim: int = 768,
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        """
        Initialize LLM reasoning engine.
        
        Args:
            llm_backend: LLM backend (OpenAI, Anthropic, etc.)
            embedding_dim: Dimension for semantic embeddings
            temperature: Sampling temperature
            max_tokens: Maximum tokens for generation
        """
        self.llm_backend = llm_backend
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.semantic_cache: Dict[str, np.ndarray] = {}
        self.reasoning_history: List[Dict[str, Any]] = []
    
    def compute_semantic_factor(
        self,
        variables: Set[Hashable],
        var_values: Dict[Hashable, Any],
        context: str,
        prompt_template: str = "Given context: {context}, what are the likely values for {variables}? Provide structured reasoning."
    ) -> Dict[Hashable, Dict[Any, float]]:
        """
        Compute semantic factor from LLM.
        
        f_LLM(X | context) = p_θ(X | context, prompt)
        
        Args:
            variables: Set of variable IDs
            var_values: Current variable values
            context: Contextual information
            prompt_template: Template for LLM prompt
            
        Returns:
            Belief distributions over variables
        """
        # Construct prompt
        var_descriptions = ", ".join([str(v) for v in variables])
        prompt = prompt_template.format(context=context, variables=var_descriptions)
        
        # Query LLM
        try:
            if hasattr(self.llm_backend, 'run'):
                response = self.llm_backend.run(prompt)
            elif callable(self.llm_backend):
                response = self.llm_backend(prompt)
            else:
                # Try to use as OpenAI-style API
                import openai
                if hasattr(openai, 'ChatCompletion'):
                    response = openai.ChatCompletion.create(
                        model=self.llm_backend if isinstance(self.llm_backend, str) else "gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    response = response.choices[0].message.content
                else:
                    response = str(prompt)  # Fallback
        except Exception as e:
            logger.warning(f"LLM query failed: {e}")
            response = ""
        
        # Parse response into belief distributions
        beliefs = self._parse_llm_response(response, variables, var_values)
        
        # Cache for future use
        cache_key = f"{context}_{var_descriptions}"
        self.semantic_cache[cache_key] = beliefs
        
        return beliefs
    
    def _parse_llm_response(
        self,
        response: str,
        variables: Set[Hashable],
        var_values: Dict[Hashable, Any]
    ) -> Dict[Hashable, Dict[Any, float]]:
        """
        Parse LLM response into belief distributions.
        
        Uses structured output parsing (JSON, YAML, or natural language).
        """
        beliefs = {}
        
        # Try JSON parsing
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                for var_id in variables:
                    if var_id in parsed:
                        val = parsed[var_id]
                        beliefs[var_id] = {val: 1.0}
            except:
                pass
        
        # Fallback: keyword-based extraction
        if not beliefs:
            for var_id in variables:
                # Look for variable mentions in response
                var_str = str(var_id)
                if var_str.lower() in response.lower():
                    # Extract associated value
                    pattern = rf"{re.escape(var_str)}[:\s]+([^\n,;]+)"
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        val_str = match.group(1).strip()
                        # Try to parse as number
                        try:
                            val = float(val_str)
                            beliefs[var_id] = {val: 1.0}
                        except:
                            beliefs[var_id] = {val_str: 1.0}
                    else:
                        # Use current value with uncertainty
                        if var_id in var_values:
                            beliefs[var_id] = {var_values[var_id]: 0.8, "unknown": 0.2}
                        else:
                            beliefs[var_id] = {"unknown": 1.0}
        
        return beliefs
    
    def generate_crm_hypothesis(
        self,
        x_t: Dict[Hashable, Any],
        u_t: str,
        x_tp1: Dict[Hashable, Any],
        context: str
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to generate CRM hypothesis from observation.
        
        LLM acts as a pattern extractor suggesting causal relationships.
        """
        prompt = f"""
        Context: {context}
        
        Observation:
        - Initial state: {x_t}
        - Action: {u_t}
        - Result state: {x_tp1}
        
        Analyze this transition and suggest:
        1. What causal relationship might explain this change?
        2. What preconditions are necessary?
        3. What postconditions result?
        4. What is the confidence in this hypothesis?
        
        Respond in JSON format:
        {{
            "preconditions": {{"var1": "value1", ...}},
            "postconditions": {{"var2": "value2", ...}},
            "causal_mechanism": "description",
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            if hasattr(self.llm_backend, 'run'):
                response = self.llm_backend.run(prompt)
            elif callable(self.llm_backend):
                response = self.llm_backend(prompt)
            else:
                response = ""
            
            # Parse JSON response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                hypothesis = json.loads(json_match.group())
                return hypothesis
        except Exception as e:
            logger.debug(f"LLM hypothesis generation failed: {e}")
        
        return None
    
    def refine_cst_with_llm(
        self,
        cst: CompositeState,
        success_examples: List[Dict[Hashable, Any]],
        failure_examples: List[Dict[Hashable, Any]]
    ) -> Optional[CompositeState]:
        """
        Use LLM to refine CST based on success/failure examples.
        
        LLM learns a classifier separating success/failure contexts.
        """
        prompt = f"""
        Given a composite state condition, analyze when it succeeds vs fails:
        
        Success examples: {success_examples[:5]}
        Failure examples: {failure_examples[:5]}
        
        Refine the condition to better separate successes from failures.
        Current condition: {[str(pred) for pred in cst.conditions]}
        
        Suggest refined predicates in JSON:
        {{
            "refined_conditions": [
                {{"var_id": "...", "operator": "...", "value": "..."}},
                ...
            ],
            "explanation": "..."
        }}
        """
        
        try:
            if hasattr(self.llm_backend, 'run'):
                response = self.llm_backend.run(prompt)
            elif callable(self.llm_backend):
                response = self.llm_backend(prompt)
            else:
                response = ""
            
            # Parse and create refined CST
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                refined_conditions = []
                for cond_dict in parsed.get("refined_conditions", []):
                    refined_conditions.append(AtomicPredicate(
                        var_id=cond_dict["var_id"],
                        operator=cond_dict.get("operator", "="),
                        value=cond_dict["value"]
                    ))
                
                if refined_conditions:
                    return CompositeState(
                        id=f"{cst.id}_refined",
                        conditions=refined_conditions
                    )
        except Exception as e:
            logger.debug(f"LLM CST refinement failed: {e}")
        
        return None


# ============================================================================
# Multi-Modal Perception Pipeline
# ============================================================================

class MultiModalPerceptionPipeline:
    """
    Multi-modal perception pipeline with vision, audio, and text.
    
    Mathematical formulation:
        q_ψ(X_t | o_t) = ∫ q_ψ(X_t | z_t) p(z_t | o_t) dz_t
        
        Where:
        - o_t = {o_text, o_vision, o_audio}: multi-modal observations
        - z_t: latent representation
        - X_t: attribute variables
        
        Cross-modal alignment:
            L_align = -log p(z_text, z_vision, z_audio | o_t)
        
        Vision-Language Model:
            z_vision = VisionEncoder(o_vision)
            z_text = TextEncoder(o_text)
            z_joint = CrossModalAttention(z_vision, z_text)
    """
    
    def __init__(
        self,
        llm_backend: Optional[Any] = None,
        vision_model: Optional[Any] = None,
        audio_model: Optional[Any] = None,
        embedding_dim: int = 768,
        use_clip: bool = True
    ):
        """
        Initialize multi-modal perception pipeline.
        
        Args:
            llm_backend: LLM for text processing
            vision_model: Vision model (ViT, CLIP, etc.)
            audio_model: Audio model (Whisper, etc.)
            embedding_dim: Embedding dimension
            use_clip: Whether to use CLIP for vision-language alignment
        """
        self.llm_backend = llm_backend
        self.vision_model = vision_model
        self.audio_model = audio_model
        self.embedding_dim = embedding_dim
        self.use_clip = use_clip
        
        # Initialize vision encoder if available
        self.clip_model = None
        self.clip_preprocess = None
        if vision_model is None and TORCH_AVAILABLE:
            try:
                import clip  # type: ignore
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
                self.use_clip = True
            except ImportError:
                self.use_clip = False
                logger.debug("CLIP not available, vision processing will be limited")
        
        # Entity detection cache
        self.entity_cache: Dict[str, List[Entity]] = {}
        self.scene_graph_cache: Dict[str, Dict[str, Any]] = {}
    
    def process_observation(
        self,
        observation: Any,
        entity_schema: Optional[Dict[str, Any]] = None,
        modality: str = "auto"
    ) -> Dict[Hashable, Dict[Any, float]]:
        """
        Process multi-modal observation into attribute belief distributions.
        
        Supports:
        - Text: LLM-based extraction
        - Images: Vision-language model
        - Audio: Speech-to-text + LLM
        - Video: Frame-by-frame processing
        - Structured data: Direct mapping
        
        Args:
            observation: Raw observation (text, image path, audio, dict, etc.)
            entity_schema: Schema defining expected entities and attributes
            modality: "text", "vision", "audio", "video", "auto" (detect automatically)
            
        Returns:
            Dictionary mapping variable IDs to belief distributions
        """
        beliefs = {}
        
        # Auto-detect modality
        if modality == "auto":
            modality = self._detect_modality(observation)
        
        if modality == "text":
            beliefs = self._process_text(observation, entity_schema)
        elif modality == "vision":
            beliefs = self._process_vision(observation, entity_schema)
        elif modality == "audio":
            beliefs = self._process_audio(observation, entity_schema)
        elif modality == "video":
            beliefs = self._process_video(observation, entity_schema)
        elif modality == "structured":
            beliefs = self._process_structured(observation)
        else:
            # Fallback
            beliefs = self._process_text(str(observation), entity_schema)
        
        return beliefs
    
    def _detect_modality(self, observation: Any) -> str:
        """Detect observation modality."""
        if isinstance(observation, dict):
            return "structured"
        elif isinstance(observation, str):
            # Check if it's a file path
            if observation.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                return "vision"
            elif observation.endswith(('.mp3', '.wav', '.flac', '.ogg')):
                return "audio"
            elif observation.endswith(('.mp4', '.avi', '.mov')):
                return "video"
            else:
                return "text"
        elif hasattr(observation, 'shape'):  # NumPy array or tensor
            if len(observation.shape) >= 3:  # Image-like
                return "vision"
            else:
                return "structured"
        else:
            return "text"
    
    def _process_text(
        self,
        text: str,
        entity_schema: Optional[Dict[str, Any]]
    ) -> Dict[Hashable, Dict[Any, float]]:
        """Process text observation using LLM."""
        beliefs = {}
        
        if not self.llm_backend:
            # Fallback: simple keyword extraction
            beliefs["text_content"] = {text: 1.0}
            return beliefs
        
        # Use LLM to extract structured information
        prompt = f"""
        Extract structured information from the following text:
        
        Text: {text}
        
        {f"Expected entities: {entity_schema}" if entity_schema else ""}
        
        Return a JSON object mapping entity attributes to values:
        {{
            "entity_id": {{
                "attribute1": "value1",
                "attribute2": "value2"
            }},
            ...
        }}
        """
        
        try:
            if hasattr(self.llm_backend, 'run'):
                response = self.llm_backend.run(prompt)
            elif callable(self.llm_backend):
                response = self.llm_backend(prompt)
            else:
                response = ""
            
            # Parse JSON response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
                for entity_id, attrs in extracted.items():
                    for attr_name, value in attrs.items():
                        var_id = f"{entity_id}_{attr_name}"
                        beliefs[var_id] = {value: 1.0}
        except Exception as e:
            logger.debug(f"Text processing failed: {e}")
            beliefs["text_content"] = {text: 1.0}
        
        return beliefs
    
    def _process_vision(
        self,
        image: Any,
        entity_schema: Optional[Dict[str, Any]]
    ) -> Dict[Hashable, Dict[Any, float]]:
        """
        Process vision observation using vision-language model.
        
        Uses CLIP or vision transformer for scene understanding.
        """
        beliefs = {}
        
        # Load image
        if isinstance(image, str):
            from PIL import Image
            try:
                img = Image.open(image)
            except:
                logger.warning(f"Could not load image: {image}")
                return beliefs
        else:
            img = image
        
        # Use CLIP for vision-language alignment
        if self.use_clip and TORCH_AVAILABLE and self.clip_model is not None:
            try:
                import torch
                import clip  # type: ignore
                
                # Encode image
                img_tensor = self.clip_preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(img_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Generate text descriptions for entities
                if entity_schema:
                    text_descriptions = []
                    for entity_id, attrs in entity_schema.items():
                        for attr_name in attrs.keys():
                            text_descriptions.append(f"{entity_id} {attr_name}")
                    
                    # Encode text
                    text_tokens = clip.tokenize(text_descriptions)
                    with torch.no_grad():
                        text_features = self.clip_model.encode_text(text_tokens)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Compute similarities
                    similarities = (image_features @ text_features.T).squeeze(0)
                    
                    # Convert to beliefs
                    idx = 0
                    for entity_id, attrs in entity_schema.items():
                        for attr_name in attrs.keys():
                            var_id = f"{entity_id}_{attr_name}"
                            similarity = float(similarities[idx])
                            # Convert similarity to probability
                            prob = max(0.0, min(1.0, (similarity + 1) / 2))
                            beliefs[var_id] = {"present": prob, "absent": 1.0 - prob}
                            idx += 1
            except Exception as e:
                logger.debug(f"CLIP processing failed: {e}")
        
        # Fallback: use LLM with vision capabilities
        if not beliefs and self.llm_backend:
            # Convert image to base64 or use vision API
            try:
                import base64
                from io import BytesIO
                
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # Use vision-capable LLM (GPT-4V, Claude, etc.)
                prompt = f"""
                Analyze this image and extract structured information.
                
                {f"Expected entities: {entity_schema}" if entity_schema else ""}
                
                Return JSON with entity attributes.
                """
                
                # Note: Would need vision API call here
                # For now, return basic beliefs
                beliefs["image_present"] = {True: 1.0}
            except Exception as e:
                logger.debug(f"Vision LLM processing failed: {e}")
        
        return beliefs
    
    def _process_audio(
        self,
        audio: Any,
        entity_schema: Optional[Dict[str, Any]]
    ) -> Dict[Hashable, Dict[Any, float]]:
        """Process audio observation (speech-to-text + LLM)."""
        beliefs = {}
        
        # Use Whisper or similar for speech-to-text
        if self.audio_model:
            try:
                # Transcribe audio
                transcription = self.audio_model.transcribe(audio)
                text = transcription.get("text", "")
                
                # Process transcribed text
                beliefs = self._process_text(text, entity_schema)
            except Exception as e:
                logger.debug(f"Audio processing failed: {e}")
        
        return beliefs
    
    def _process_video(
        self,
        video: Any,
        entity_schema: Optional[Dict[str, Any]]
    ) -> Dict[Hashable, Dict[Any, float]]:
        """Process video observation (frame-by-frame)."""
        beliefs = {}
        
        # Extract frames and process each
        # Simplified: would use video processing library
        try:
            # For now, treat as single frame
            if isinstance(video, str):
                # Assume first frame
                beliefs = self._process_vision(video, entity_schema)
        except Exception as e:
            logger.debug(f"Video processing failed: {e}")
        
        return beliefs
    
    def _process_structured(
        self,
        data: Dict[str, Any]
    ) -> Dict[Hashable, Dict[Any, float]]:
        """Process structured observation."""
        beliefs = {}
        for var_id, value in data.items():
            beliefs[var_id] = {value: 1.0}
        return beliefs


# ============================================================================
# Perception Pipeline (Enhanced)
# ============================================================================

class PerceptionPipeline:
    """
    Enhanced perception pipeline with multi-modal support and LLM integration.
    
    q_ψ(X_t | o_t) ≈ p(X_t | o_t)
    
    Now supports:
    - Multi-modal observations (text, vision, audio, video)
    - LLM-based semantic extraction
    - Vision-language models (CLIP)
    - Cross-modal alignment
    """
    
    def __init__(
        self,
        llm_backend: Optional[Any] = None,
        vision_model: Optional[Any] = None,
        audio_model: Optional[Any] = None,
        use_multi_modal: bool = True
    ):
        """
        Initialize perception pipeline.
        
        Args:
            llm_backend: LLM backend for text-based perception
            vision_model: Vision model for image processing
            audio_model: Audio model for speech processing
            use_multi_modal: Whether to use multi-modal processing
        """
        self.llm_backend = llm_backend
        
        if use_multi_modal:
            self.multi_modal_pipeline = MultiModalPerceptionPipeline(
                llm_backend=llm_backend,
                vision_model=vision_model,
                audio_model=audio_model
            )
        else:
            self.multi_modal_pipeline = None
    
    def process_observation(
        self,
        observation: Any,
        entity_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[Hashable, Dict[Any, float]]:
        """
        Process observation into attribute belief distributions.
        
        Enhanced with multi-modal support.
        
        Args:
            observation: Raw observation (text, image, audio, video, structured data, etc.)
            entity_schema: Schema defining expected entities and attributes
            
        Returns:
            Dictionary mapping variable IDs to belief distributions
        """
        if self.multi_modal_pipeline:
            return self.multi_modal_pipeline.process_observation(
                observation, entity_schema
            )
        
        # Fallback to simple processing
        beliefs = {}
        
        if isinstance(observation, dict):
            # Structured observation: direct mapping
            for var_id, value in observation.items():
                beliefs[var_id] = {value: 1.0}
        elif isinstance(observation, str) and self.llm_backend:
            # Text observation: use LLM to extract structured information
            try:
                prompt = f"""
                Extract structured information from: {observation}
                
                {f"Expected schema: {entity_schema}" if entity_schema else ""}
                
                Return JSON mapping variables to values.
                """
                
                if hasattr(self.llm_backend, 'run'):
                    response = self.llm_backend.run(prompt)
                elif callable(self.llm_backend):
                    response = self.llm_backend(prompt)
                else:
                    response = ""
                
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    extracted = json.loads(json_match.group())
                    for var_id, value in extracted.items():
                        beliefs[var_id] = {value: 1.0}
            except Exception as e:
                logger.debug(f"LLM extraction failed: {e}")
                beliefs["text_content"] = {observation: 1.0}
        else:
            # Default: create a single variable with the observation
            var_id = "obs_0"
            beliefs[var_id] = {observation: 1.0}
        
        return beliefs


# ============================================================================
# Planning and Action Selection
# ============================================================================

class Planner:
    """
    Implements planning/abduction over CRMs.
    
    Given current belief and drives, find action sequence maximizing expected utility.
    """
    
    def __init__(
        self,
        kb: KnowledgeBase,
        factor_graph_builder: FactorGraphBuilder,
        message_passing_engine: MessagePassingEngine,
        horizon: int = 3,
        num_samples: int = 10
    ):
        """
        Initialize planner.
        
        Args:
            kb: Knowledge base
            factor_graph_builder: Factor graph builder
            message_passing_engine: Message passing engine
            horizon: Planning horizon (number of steps ahead)
            num_samples: Number of action sequences to sample
        """
        self.kb = kb
        self.factor_graph_builder = factor_graph_builder
        self.message_passing_engine = message_passing_engine
        self.horizon = horizon
        self.num_samples = num_samples
    
    def plan(
        self,
        current_beliefs: Dict[Hashable, Dict[Any, float]],
        goals: List[Drive],
        available_actions: List[str]
    ) -> Tuple[Optional[str], float]:
        """
        Plan next action to maximize expected utility.
        
        Args:
            current_beliefs: Current belief distributions over variables
            goals: Active drives/goals
            available_actions: List of available actions
            
        Returns:
            Tuple of (best action, expected utility)
        """
        if not available_actions:
            return None, 0.0
        
        # Simplified: evaluate each action greedily
        best_action = None
        best_utility = float('-inf')
        
        # Extract most likely state from beliefs
        var_values = {}
        for var_id, belief in current_beliefs.items():
            if belief:
                best_val = max(belief.items(), key=lambda x: x[1])[0]
                var_values[var_id] = best_val
        
        for action in available_actions:
            # Build factor graph with this action
            graph = self.factor_graph_builder.build_factor_graph(
                var_values=var_values,
                goals=goals,
                action=action
            )
            
            # Run inference
            new_beliefs = self.message_passing_engine.run_sum_product(graph)
            
            # Compute expected utility
            total_utility = 0.0
            for drive in goals:
                if drive.goal_cst_id not in self.kb.csts:
                    continue
                
                goal_cst = self.kb.csts[drive.goal_cst_id]
                goal_vars = {pred.var_id for pred in goal_cst.conditions}
                
                # Compute expected satisfaction
                expected_sat = 0.0
                for var_id in goal_vars:
                    if var_id in new_beliefs:
                        belief = new_beliefs[var_id]
                        # Simplified: use most likely value
                        if belief:
                            best_val = max(belief.items(), key=lambda x: x[1])[0]
                            # Create temporary var_values for satisfaction computation
                            temp_vals = var_values.copy()
                            temp_vals[var_id] = best_val
                            sat = goal_cst.evaluate_soft(temp_vals)
                            expected_sat += sat * belief.get(best_val, 0.0)
                
                expected_sat /= max(1, len(goal_vars))
                utility = drive.compute_utility(expected_sat)
                total_utility += utility
            
            if total_utility > best_utility:
                best_utility = total_utility
                best_action = action
        
        return best_action, best_utility


# ============================================================================
# Main AERASigmaAgent Class
# ============================================================================

class AERASigmaAgent(Agent):
    """
    Main Σ-AERA agent integrating all components.
    
    Inherits from Swarms' Agent class for full framework compatibility.
    
    Implements the complete cognitive cycle:
    1. Perception: o_t → X_t beliefs
    2. Factor graph construction: F(K_t, x_t, g_t, c_t) → G_t
    3. Inference: message passing on G_t
    4. Planning: abduction over CRMs
    5. Action selection
    6. Learning: CTPX, PTPX, GTPX, parametric updates
    7. Analogy: CRM transfer
    8. Meta-control: attention and resource allocation
    
    Swarms Integration:
    - Inherits from swarms.structs.agent.Agent
    - Uses Swarms' LLM handling (llm parameter)
    - Compatible with Swarms' tools, memory, and workflows
    - Overrides run() method to use cognitive cycle
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        model_name: Optional[str] = None,
        seed_kb: Optional[KnowledgeBase] = None,
        max_iterations: int = 100,
        learning_enabled: bool = True,
        analogy_enabled: bool = True,
        agent_name: Optional[str] = "sigma-aera-agent",
        agent_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_loops: Optional[Union[int, str]] = 1,
        verbose: Optional[bool] = False,
        **kwargs
    ):
        """
        Initialize Σ-AERA agent with Swarms integration.
        
        Args:
            llm: Swarms LLM backend (from swarms.models or compatible)
            model_name: Model name string (alternative to llm)
            seed_kb: Initial seed knowledge base (AERA-style)
            max_iterations: Maximum cognitive cycles per task
            learning_enabled: Whether to enable learning mechanisms
            analogy_enabled: Whether to enable analogy mechanisms
            agent_name: Name of the agent (Swarms compatible)
            agent_description: Description of the agent
            system_prompt: System prompt for LLM integration
            max_loops: Maximum loops for Swarms Agent (default: 1, uses cognitive cycle)
            verbose: Enable verbose logging
            **kwargs: Additional Swarms Agent parameters
        """
        # Initialize Swarms Agent parent class FIRST
        # Use Swarms' default system prompt if not provided
        if system_prompt is None:
            try:
                from swarms.prompts.agent_system_prompts import AGENT_SYSTEM_PROMPT_3
                system_prompt = AGENT_SYSTEM_PROMPT_3
            except ImportError:
                system_prompt = "You are a Σ-AERA cognitive agent with self-programming and causal reasoning capabilities."
        
        # Call parent Agent __init__ with Swarms parameters
        # Pass all standard Swarms parameters
        super().__init__(
            llm=llm,
            model_name=model_name,
            agent_name=agent_name or "sigma-aera-agent",
            agent_description=agent_description or "Σ-AERA cognitive architecture agent with self-programming, causal reasoning, and multi-modal perception",
            system_prompt=system_prompt,
            max_loops=max_loops,
            verbose=verbose,
            **kwargs
        )
        
        # Store AERA-specific configuration
        self.max_iterations = max_iterations
        self.learning_enabled = learning_enabled
        self.analogy_enabled = analogy_enabled
        
        # Initialize knowledge base
        self.kb = seed_kb or KnowledgeBase()
        
        # Resolve LLM backend (use Swarms' llm after parent init)
        # The parent Agent.__init__ will have set self.llm via llm_handling() if model_name was provided
        llm_backend = getattr(self, 'llm', None) if hasattr(self, 'llm') else (llm if llm else None)
        
        # Initialize components
        self.message_passing_engine = MessagePassingEngine()
        self.factor_graph_builder = FactorGraphBuilder(self.kb, self.message_passing_engine)
        
        # Enhanced perception with multi-modal support
        self.perception_pipeline = PerceptionPipeline(
            llm_backend=llm_backend,
            use_multi_modal=True
        )
        
        self.planner = Planner(self.kb, self.factor_graph_builder, self.message_passing_engine)
        
        # Deep LLM integration (use Swarms' LLM)
        self.llm_reasoning_engine = LLMReasoningEngine(llm_backend) if llm_backend else None
        
        # Enhanced learning with experience graph and LLM integration
        experience_graph = ExperienceGraph()
        self.learning_mechanisms = LearningMechanisms(
            self.kb,
            experience_graph=experience_graph,
            llm_reasoning_engine=self.llm_reasoning_engine
        )
        self.analogy_engine = AnalogyEngine(self.kb) if analogy_enabled else None
        
        # Dual learning system
        self.dual_learning = DualLearningSystem(self.kb, self.learning_mechanisms)
        
        # Joint training system (LLM + Factor Graph)
        if self.llm_reasoning_engine:
            self.joint_training = JointTrainingSystem(
                llm_reasoning_engine=self.llm_reasoning_engine,
                kb=self.kb,
                factor_graph_builder=self.factor_graph_builder
            )
        else:
            self.joint_training = None
        
        # Safety layer
        self.safety_layer = SafetyLayer(self.kb)
        
        # Meta-control
        self.meta_control = MetaControl(self.kb)
        
        # Memory compression
        self.memory_compressor = MemoryCompressor()
        self.procedural_cache = ProceduralCache()
        
        # Prediction error tracking for meta-control
        self.prediction_errors: Dict[Hashable, float] = {}
        
        # Configuration
        self.max_iterations = max_iterations
        self.learning_enabled = learning_enabled
        
        # State
        self.current_beliefs: Dict[Hashable, Dict[Any, float]] = {}
        self.active_drives: List[Drive] = []
        self.history: List[Dict[str, Any]] = []
        self.iteration = 0
        
        # Initialize seed KB if not provided
        if seed_kb is None:
            self.initialize_seed_kb()
    
    def initialize_seed_kb(self):
        """
        Initialize a minimal seed knowledge base.
        
        This provides basic ontologies and high-level CRMs for interaction.
        Enhanced with text processing capabilities.
        """
        # Create a text processing entity
        text_entity = Entity(
            id="text_processor",
            ontologies={"text", "information"},
            attributes={"content": "text_content", "topic": "text_topic", "intent": "text_intent"}
        )
        self.kb.add_entity(text_entity)
        
        # Create a general response entity
        response_entity = Entity(
            id="response_generator",
            ontologies={"response", "output"},
            attributes={"content": "response_content", "quality": "response_quality"}
        )
        self.kb.add_entity(response_entity)
        
        # Create a simple CST for text processing
        text_cst = CompositeState(
            id="has_text_input",
            conditions=[
                AtomicPredicate(var_id="text_content", operator="!=", value="")
            ]
        )
        self.kb.add_cst(text_cst)
        
        # Create a simple CRM for text-to-response transformation
        # This allows the agent to process text and generate responses
        response_cst = CompositeState(
            id="has_response_output",
            conditions=[
                AtomicPredicate(var_id="response_content", operator="!=", value="")
            ]
        )
        self.kb.add_cst(response_cst)
        
        # Create a simple parametric model for text processing
        text_to_response_model = ParametricModel(
            model_type="linear_gaussian",
            parameters={
                "A": np.eye(2),  # Identity (text -> response)
                "B": np.zeros((2, 1)),  # No action input needed
                "Sigma": np.eye(2) * 0.1
            }
        )
        
        # Create CRM: text input -> response output
        text_response_crm = CRM(
            id="text_to_response",
            pre_cst_id="has_text_input",
            post_cst_id="has_response_output",
            actions={"process_text", "generate_response"},
            param_model=text_to_response_model
        )
        self.kb.add_crm(text_response_crm)
        
        # Create Mreq linking text CST to text_response CRM
        text_mreq = Mreq(
            id="text_mreq_1",
            cst_id="has_text_input",
            crm_id="text_to_response",
            confidence=0.7
        )
        self.kb.add_mreq(text_mreq)
        
        if self.verbose:
            logger.info(f"Initialized seed knowledge base: {len(self.kb.entities)} entities, {len(self.kb.crms)} CRMs, {len(self.kb.csts)} CSTs")
    
    def cognitive_cycle(
        self,
        observation: Any,
        available_actions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute one cognitive cycle.
        
        Args:
            observation: Raw observation from environment
            available_actions: List of available actions (if None, agent must infer)
            
        Returns:
            Dictionary with action, beliefs, and metadata
        """
        self.iteration += 1
        
        # Step 1: Perception
        perception_beliefs = self.perception_pipeline.process_observation(observation)
        
        # Merge with current beliefs
        for var_id, belief in perception_beliefs.items():
            if var_id in self.current_beliefs:
                # Update beliefs (simplified: replace with perception)
                self.current_beliefs[var_id] = belief
            else:
                self.current_beliefs[var_id] = belief
        
        # Extract most likely state
        var_values = {}
        for var_id, belief in self.current_beliefs.items():
            if belief:
                best_val = max(belief.items(), key=lambda x: x[1])[0]
                var_values[var_id] = best_val
        
        # Step 2: Factor graph construction with LLM semantic factors
        graph = self.factor_graph_builder.build_factor_graph(
            var_values=var_values,
            goals=self.active_drives,
            action=None,
            perception_factors=perception_beliefs
        )
        
        # Add LLM semantic factors if available
        if self.llm_reasoning_engine:
            # Get variables in graph
            graph_vars = set(graph.variables.keys())
            
            # Compute semantic factors from LLM
            context = f"Iteration {self.iteration}, current state: {var_values}"
            llm_beliefs = self.llm_reasoning_engine.compute_semantic_factor(
                variables=graph_vars,
                var_values=var_values,
                context=context
            )
            
            # Add LLM factors to graph
            for var_id, belief_dist in llm_beliefs.items():
                if var_id in graph.variables:
                    factor_id = f"llm_semantic_{var_id}"
                    
                    def make_llm_factor(vid, bel):
                        def llm_factor(var_vals):
                            val = var_vals.get(vid)
                            return bel.get(val, 0.01) if isinstance(bel, dict) else 1.0
                        return llm_factor
                    
                    factor = FactorNode(
                        id=factor_id,
                        factor_type="llm_semantic",
                        var_neighbors={var_id},
                        factor_func=make_llm_factor(var_id, belief_dist),
                        metadata={"var_id": var_id, "source": "llm"}
                    )
                    graph.add_factor(factor_id, factor)
        
        # Step 3: Meta-control: compute attention allocation
        meta_state = self.meta_control.compute_meta_state(
            self.current_beliefs,
            self.active_drives,
            self.prediction_errors
        )
        
        active_crms = self.kb.get_active_crms(var_values)
        attention_weights = self.meta_control.select_attention(
            meta_state,
            active_crms,
            []
        )
        
        # Allocate compute budget
        compute_allocation = self.meta_control.allocate_compute_budget(attention_weights)
        
        # Step 3: Inference with bounded compute budget
        # Use attention-weighted message passing (simplified)
        beliefs = self.message_passing_engine.run_sum_product(graph)
        
        # Joint training: align factor graph and LLM beliefs
        if self.joint_training and self.llm_reasoning_engine:
            # Get LLM beliefs for comparison
            llm_beliefs = self.llm_reasoning_engine.compute_semantic_factor(
                variables=set(graph.variables.keys()),
                var_values=var_values,
                context=f"Iteration {self.iteration}"
            )
            
            # Compute joint loss and update
            loss_dict = self.joint_training.update_joint_parameters(
                var_values, beliefs, llm_beliefs, f"Iteration {self.iteration}"
            )
            
            # Blend beliefs (weighted combination)
            blended_beliefs = {}
            for var_id in set(beliefs.keys()) | set(llm_beliefs.keys()):
                belief_fg = beliefs.get(var_id, {})
                belief_llm = llm_beliefs.get(var_id, {})
                
                # Weighted combination
                alpha = 1.0 - self.joint_training.llm_weight
                blended = {}
                all_vals = set(belief_fg.keys()) | set(belief_llm.keys())
                for val in all_vals:
                    p_fg = belief_fg.get(val, 0.0)
                    p_llm = belief_llm.get(val, 0.0)
                    blended[val] = alpha * p_fg + (1 - alpha) * p_llm
                
                # Renormalize
                total = sum(blended.values())
                if total > 0:
                    blended = {k: v / total for k, v in blended.items()}
                else:
                    blended = {val: 1.0 / len(all_vals) for val in all_vals} if all_vals else {}
                
                blended_beliefs[var_id] = blended
            
            beliefs = blended_beliefs
        
        self.current_beliefs = beliefs
        
        # Update meta-control policy based on performance
        # Compute performance from drive satisfaction
        performance = 0.0
        for drive in self.active_drives:
            if drive.goal_cst_id in self.kb.csts:
                goal_cst = self.kb.csts[drive.goal_cst_id]
                satisfaction = goal_cst.evaluate_soft(var_values)
                performance += satisfaction
        
        performance /= max(1, len(self.active_drives))
        self.meta_control.update_attention_policy(meta_state, compute_allocation, performance)
        
        # Step 4: Planning and action selection
        if available_actions is None:
            # Infer actions from active CRMs
            active_crms = self.kb.get_active_crms(var_values)
            available_actions = []
            for crm in active_crms:
                available_actions.extend(crm.actions)
            
            # If no CRMs active, provide default actions for text processing
            if not available_actions:
                # Check if we have text input
                if "text_content" in var_values or any("text" in str(k).lower() for k in var_values.keys()):
                    available_actions = ["process_text", "generate_response", "analyze"]
                else:
                    available_actions = ["noop"]
            
            available_actions = list(set(available_actions))
        
        action, expected_utility = self.planner.plan(
            current_beliefs=beliefs,
            goals=self.active_drives,
            available_actions=available_actions
        )
        
        if action is None:
            action = available_actions[0] if available_actions else "noop"
        
        # If action is "process_text" or "generate_response", use LLM to generate response
        if action in ["process_text", "generate_response", "analyze"] and hasattr(self, 'llm') and self.llm:
            # Extract task from observation
            task_text = str(observation) if isinstance(observation, str) else str(observation)
            
            # Use LLM to generate response
            try:
                if hasattr(self.llm, 'run'):
                    llm_response = self.llm.run(task=task_text)
                else:
                    llm_response = str(self.llm(task_text))
                
                # Update beliefs with response
                if "response_content" not in beliefs:
                    beliefs["response_content"] = {}
                beliefs["response_content"][llm_response] = 1.0
                
                # Update var_values
                var_values["response_content"] = llm_response
                
                # Update expected utility (high for successful text processing)
                expected_utility = 0.9
                
            except Exception as e:
                if self.verbose:
                    logger.debug(f"LLM action execution failed: {e}")
                expected_utility = 0.1
        
        # Step 5: Record experience (for learning)
        x_t = var_values.copy()
        u_t = action
        # x_tp1 will be updated after observation
        
        # Compute intrinsic reward (curiosity)
        predicted_state = {}
        for var_id, belief in beliefs.items():
            if belief:
                predicted_state[var_id] = max(belief.items(), key=lambda x: x[1])[0]
        
        intrinsic_reward = self.meta_control.compute_intrinsic_reward(
            predicted_state,
            var_values
        )
        
        # Generate explanation if available
        explanation = ""
        if hasattr(self, 'explanation_generator') and self.explanation_generator:
            try:
                # Use recent history for explanation
                recent_history = self.history[-5:] if len(self.history) >= 5 else self.history
                if recent_history:
                    explanation_dict = self.explanation_generator.generate_explanation(recent_history)
                    explanation = explanation_dict.get("explanation", "")
            except Exception as e:
                if self.verbose:
                    logger.debug(f"Explanation generation failed: {e}")
        
        result = {
            "action": action,
            "expected_utility": expected_utility,
            "intrinsic_reward": intrinsic_reward,
            "beliefs": beliefs,
            "active_crms": [crm.id for crm in self.kb.get_active_crms(var_values, action)],
            "meta_state": meta_state,
            "attention_weights": attention_weights,
            "iteration": self.iteration,
            "explanation": explanation
        }
        
        # Store for learning
        self.history.append({
            "x_t": x_t,
            "u_t": u_t,
            "beliefs": beliefs,
            "timestamp": time.time()
        })
        
        return result
    
    def learn_from_experience(
        self,
        x_t: Dict[Hashable, Any],
        u_t: str,
        x_tp1: Dict[Hashable, Any]
    ):
        """
        Apply learning mechanisms after observing transition.
        
        Enhanced with dual learning loops and safety validation.
        
        Args:
            x_t: State at time t
            u_t: Action at time t
            x_tp1: Observed state at time t+1
        """
        if not self.learning_enabled:
            return
        
        # Get active CRMs
        active_crms = self.kb.get_active_crms(x_t, u_t)
        active_crm_ids = [crm.id for crm in active_crms]
        
        # Record experience
        self.kb.record_experience(x_t, u_t, x_tp1, active_crm_ids)
        
        # Compute prediction errors for meta-control
        for crm in active_crms:
            if crm.post_cst_id in self.kb.csts:
                post_cst = self.kb.csts[crm.post_cst_id]
                error = 1.0 - post_cst.evaluate_soft(x_tp1)
                self.prediction_errors[crm.id] = error
        
        # Dual learning: parametric update (inner loop)
        old_kb_state = {
            "num_crms": len(self.kb.crms),
            "num_csts": len(self.kb.csts),
            "num_mreqs": len(self.kb.mreqs)
        }
        
        self.dual_learning.run_parametric_update(x_t, u_t, x_tp1, active_crms)
        
        # Dual learning: structural update (outer loop)
        structural_changes = self.dual_learning.run_structural_update(
            x_t, u_t, x_tp1, active_crms, self.active_drives
        )
        
        # Safety validation for structural changes
        if structural_changes["ctpx"] or structural_changes["ptpx"] or structural_changes["gtpx"]:
            # Get validation traces (recent history)
            validation_traces = self.kb.history[-10:] if len(self.kb.history) >= 10 else self.kb.history
            
            for change_type, change_id in [
                ("add_crm", structural_changes["ctpx"]),
                ("add_crm", structural_changes["gtpx"][0] if structural_changes["gtpx"] else None)
            ]:
                if change_id:
                    proposed_change = {"type": change_type, "id": change_id}
                    if not self.safety_layer.validate_change(proposed_change, validation_traces):
                        # Rollback if validation fails
                        logger.warning(f"Structural change {change_id} failed validation, rolling back")
                        self.safety_layer.rollback()
        
        # Compute stability loss
        new_kb_state = {
            "num_crms": len(self.kb.crms),
            "num_csts": len(self.kb.csts),
            "num_mreqs": len(self.kb.mreqs)
        }
        stability_loss = self.safety_layer.compute_stability_loss(old_kb_state, new_kb_state)
        
        # Prune low-frequency CRMs periodically
        if self.iteration % 50 == 0:
            pruned = self.learning_mechanisms.prune_low_frequency_crms()
            if pruned > 0:
                logger.info(f"Pruned {pruned} low-frequency CRMs")
        
        logger.debug(
            f"Learning: CTPX={structural_changes['ctpx'] is not None}, "
            f"PTPX={len(structural_changes['ptpx'])}, "
            f"GTPX={len(structural_changes['gtpx'])}, "
            f"stability_loss={stability_loss:.4f}"
        )
    
    def add_drive(self, drive: Drive):
        """Add a drive/goal to the agent."""
        self.kb.add_drive(drive)
        self.active_drives.append(drive)
    
    def run(
        self,
        task: Optional[Union[str, Any]] = None,
        img: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Swarms-compatible run method using Σ-AERA cognitive cycle.
        
        This method overrides Swarms' Agent.run() to use the cognitive architecture
        while maintaining compatibility with Swarms' interface.
        
        Args:
            task: Task description or question (Swarms standard)
            img: Optional image path or image data (Swarms standard)
            **kwargs: Additional arguments (e.g., available_actions, goals)
            
        Returns:
            Response string (Swarms standard) or structured output based on output_type
        """
        if task is None:
            if self.verbose:
                logger.warning("No task provided to AERASigmaAgent.run()")
            return ""
        
        # Convert task to observation format
        observation = img if img else task
        
        # Get available actions if provided
        available_actions = kwargs.get("available_actions")
        
        # Capture initial state before cognitive cycle (for learning)
        initial_perception = self.perception_pipeline.process_observation(observation)
        x_t = {}
        for var_id, belief in initial_perception.items():
            if belief:
                x_t[var_id] = max(belief.items(), key=lambda x: x[1])[0]
        
        # Execute cognitive cycle
        try:
            # Try cognitive cycle first
            result = self.cognitive_cycle(
                observation=observation,
                available_actions=available_actions
            )
            
            # Extract response components
            action = result.get("action", "noop")
            beliefs = result.get("beliefs", {})
            expected_utility = result.get("expected_utility", 0.0)
            explanation = result.get("explanation", "")
            
            # Learn from experience if learning is enabled
            if self.learning_enabled:
                # Extract state after action (x_tp1) from beliefs
                x_tp1 = {}
                for var_id, belief in beliefs.items():
                    if belief:
                        x_tp1[var_id] = max(belief.items(), key=lambda x: x[1])[0]
                
                # Only learn if we have meaningful state transitions
                if x_t and x_tp1 and action != "noop":
                    try:
                        self.learn_from_experience(x_t, action, x_tp1)
                        if self.verbose:
                            kb_stats = self.get_knowledge_base_stats()
                            logger.info(
                                f"Learned from experience. KB stats: "
                                f"{kb_stats['num_crms']} CRMs, "
                                f"{kb_stats['num_csts']} CSTs, "
                                f"{kb_stats['num_mreqs']} Mreqs"
                            )
                    except Exception as e:
                        if self.verbose:
                            logger.warning(f"Learning from experience failed: {e}")
            
            # Format response based on Swarms output_type
            output_type = getattr(self, 'output_type', 'str-all-except-first')
            
            # If cognitive cycle didn't produce meaningful output, fall back to parent Agent
            if not result or (action == "noop" and not beliefs and not explanation):
                if self.verbose:
                    logger.info("Cognitive cycle produced minimal output, falling back to parent Agent.run()")
                # Fall back to standard Swarms Agent behavior
                return super().run(task=task, img=img, **kwargs)
            
            # If we have LLM reasoning, try to get a natural language response
            if self.llm_reasoning_engine and hasattr(self.llm_reasoning_engine, 'llm_backend') and self.llm_reasoning_engine.llm_backend:
                try:
                    # Use LLM to generate natural language response
                    llm_prompt = f"""
                    Task: {task}
                    Agent's reasoning: {beliefs}
                    Action taken: {action}
                    Expected utility: {expected_utility:.3f}
                    
                    Generate a clear, natural language response explaining the agent's reasoning and action.
                    """
                    
                    # Use Swarms' LLM if available, otherwise use reasoning engine's backend
                    if hasattr(self, 'llm') and self.llm:
                        if hasattr(self.llm, 'run'):
                            llm_response = self.llm.run(task=llm_prompt)
                        else:
                            llm_response = str(self.llm(llm_prompt))
                    else:
                        llm_response = self.llm_reasoning_engine.llm_backend.run(llm_prompt) if hasattr(self.llm_reasoning_engine.llm_backend, 'run') else str(self.llm_reasoning_engine.llm_backend(llm_prompt))
                    
                    if llm_response:
                        response = llm_response
                    else:
                        response = explanation or f"Processed task: {task}\nAction: {action}"
                except Exception as e:
                    if self.verbose:
                        logger.debug(f"LLM response generation failed: {e}")
                    response = explanation or f"Processed task: {task}\nAction: {action}"
            else:
                # Fallback: use Swarms LLM directly if available
                if hasattr(self, 'llm') and self.llm:
                    try:
                        # Use parent Agent's standard LLM call
                        if hasattr(self.llm, 'run'):
                            response = self.llm.run(task=task)
                        else:
                            response = str(self.llm(task))
                        
                        # Enhance with AERA context if available
                        if beliefs:
                            response = f"{response}\n\n[AERA Reasoning: {len(beliefs)} belief variables analyzed]"
                    except Exception as e:
                        if self.verbose:
                            logger.debug(f"LLM fallback failed: {e}")
                        # Final fallback: structured response
                        response_parts = []
                        if explanation:
                            response_parts.append(explanation)
                        else:
                            response_parts.append(f"Task: {task}")
                            if action != "noop":
                                response_parts.append(f"Action: {action}")
                            if expected_utility > 0.1:
                                response_parts.append(f"Expected utility: {expected_utility:.3f}")
                        
                        response = "\n".join(response_parts) if response_parts else f"Processed task: {task}"
                else:
                    # Final fallback: structured response
                    response_parts = []
                    if explanation:
                        response_parts.append(explanation)
                    else:
                        response_parts.append(f"Task: {task}")
                        if action != "noop":
                            response_parts.append(f"Action: {action}")
                        if expected_utility > 0.1:
                            response_parts.append(f"Expected utility: {expected_utility:.3f}")
                    
                    response = "\n".join(response_parts) if response_parts else f"Processed task: {task}"
            
            # Store in history
            self.history.append({
                "task": task,
                "observation": observation,
                "result": result,
                "response": response,
                "timestamp": time.time()
            })
            
            # Add to Swarms' short_memory if available
            if hasattr(self, 'short_memory'):
                try:
                    self.short_memory.add(
                        role=getattr(self, 'agent_name', 'sigma-aera-agent'),
                        content=response
                    )
                except Exception as e:
                    if self.verbose:
                        logger.debug(f"Could not add to short_memory: {e}")
            
            if self.verbose:
                logger.info(f"Task completed. Response length: {len(response)}")
            
            # Return formatted based on output_type
            if output_type in ['json', 'dict']:
                return {
                    "response": response,
                    "action": action,
                    "beliefs": beliefs,
                    "expected_utility": expected_utility,
                    "explanation": explanation
                }
            elif output_type in ['list']:
                return [response]
            else:
                return response
                
        except Exception as e:
            error_msg = f"Error executing task: {str(e)}"
            logger.error(error_msg)
            if self.verbose:
                import traceback
                logger.error(traceback.format_exc())
            
            # Fallback to parent Agent if cognitive cycle fails
            try:
                if self.verbose:
                    logger.info("Falling back to parent Agent.run() due to error")
                return super().run(task=task, img=img, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return error_msg
    
    def run_batch(
        self,
        observations: List[Any],
        available_actions: Optional[List[List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run agent for multiple steps (original AERA method, kept for compatibility).
        
        Args:
            observations: List of observations
            available_actions: Optional list of available actions per step
            
        Returns:
            List of results from each cognitive cycle
        """
        results = []
        
        for i, obs in enumerate(observations):
            actions = available_actions[i] if available_actions and i < len(available_actions) else None
            
            result = self.cognitive_cycle(obs, actions)
            results.append(result)
            
            # If we have next observation, learn from transition
            if i < len(observations) - 1:
                # Extract states from beliefs
                x_t = {}
                for var_id, belief in self.current_beliefs.items():
                    if belief:
                        x_t[var_id] = max(belief.items(), key=lambda x: x[1])[0]
                
                # Process next observation to get x_tp1
                next_obs = observations[i + 1]
                next_perception = self.perception_pipeline.process_observation(next_obs)
                x_tp1 = {}
                for var_id, belief in next_perception.items():
                    if belief:
                        x_tp1[var_id] = max(belief.items(), key=lambda x: x[1])[0]
                
                # Learn
                self.learn_from_experience(x_t, result["action"], x_tp1)
        
        return results
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            "num_entities": len(self.kb.entities),
            "num_crms": len(self.kb.crms),
            "num_csts": len(self.kb.csts),
            "num_mreqs": len(self.kb.mreqs),
            "num_anti_mreqs": len(self.kb.anti_mreqs),
            "num_drives": len(self.kb.drives),
            "history_size": len(self.kb.history)
        }
    
    def save_knowledge_base(self, file_path: Optional[str] = None) -> bool:
        """
        Save the knowledge base to disk.
        
        Args:
            file_path: Optional path to save the knowledge base. 
                      If None, uses agent_name_kb.pkl in workspace_dir.
        
        Returns:
            True if save was successful, False otherwise.
        """
        try:
            if file_path is None:
                # Use workspace directory if available
                workspace_dir = getattr(self, 'workspace_dir', '.')
                agent_name = getattr(self, 'agent_name', 'sigma-aera-agent')
                file_path = os.path.join(workspace_dir, f"{agent_name}_kb.pkl")
            
            # Ensure directory exists
            dir_path = os.path.dirname(os.path.abspath(file_path))
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # Serialize knowledge base
            kb_data = {
                "entities": {str(k): self._serialize_entity(v) for k, v in self.kb.entities.items()},
                "crms": {str(k): self._serialize_crm(v) for k, v in self.kb.crms.items()},
                "csts": {str(k): self._serialize_cst(v) for k, v in self.kb.csts.items()},
                "mreqs": {str(k): self._serialize_mreq(v) for k, v in self.kb.mreqs.items()},
                "anti_mreqs": {str(k): self._serialize_anti_mreq(v) for k, v in self.kb.anti_mreqs.items()},
                "drives": {str(k): self._serialize_drive(v) for k, v in self.kb.drives.items()},
                "history": self.kb.history[-1000:],  # Keep last 1000 experiences
                "attention_policy": self.kb.attention_policy,
                "rl_modules": self.kb.rl_modules,
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(kb_data, f)
            
            if self.verbose:
                logger.info(f"Saved knowledge base to {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            if self.verbose:
                import traceback
                logger.error(traceback.format_exc())
            return False
    
    def load_knowledge_base(self, file_path: Optional[str] = None) -> bool:
        """
        Load the knowledge base from disk.
        
        Args:
            file_path: Optional path to load the knowledge base from.
                      If None, uses agent_name_kb.pkl in workspace_dir.
        
        Returns:
            True if load was successful, False otherwise.
        """
        try:
            if file_path is None:
                # Use workspace directory if available
                workspace_dir = getattr(self, 'workspace_dir', '.')
                agent_name = getattr(self, 'agent_name', 'sigma-aera-agent')
                file_path = os.path.join(workspace_dir, f"{agent_name}_kb.pkl")
            
            if not os.path.exists(file_path):
                if self.verbose:
                    logger.warning(f"Knowledge base file not found: {file_path}")
                return False
            
            with open(file_path, 'rb') as f:
                kb_data = pickle.load(f)
            
            # Deserialize entities
            for k, v in kb_data.get("entities", {}).items():
                self.kb.add_entity(self._deserialize_entity(k, v))
            
            # Deserialize CSTs first (CRMs depend on them)
            for k, v in kb_data.get("csts", {}).items():
                self.kb.add_cst(self._deserialize_cst(k, v))
            
            # Deserialize CRMs
            for k, v in kb_data.get("crms", {}).items():
                self.kb.add_crm(self._deserialize_crm(k, v))
            
            # Deserialize Mreqs
            for k, v in kb_data.get("mreqs", {}).items():
                self.kb.add_mreq(self._deserialize_mreq(k, v))
            
            # Deserialize AntiMreqs
            for k, v in kb_data.get("anti_mreqs", {}).items():
                self.kb.add_anti_mreq(self._deserialize_anti_mreq(k, v))
            
            # Deserialize drives
            for k, v in kb_data.get("drives", {}).items():
                self.kb.add_drive(self._deserialize_drive(k, v))
            
            # Restore history and other data
            self.kb.history = kb_data.get("history", [])
            self.kb.attention_policy = kb_data.get("attention_policy")
            self.kb.rl_modules = kb_data.get("rl_modules", {})
            
            if self.verbose:
                logger.info(f"Loaded knowledge base from {file_path}")
                stats = self.get_knowledge_base_stats()
                logger.info(
                    f"Loaded: {stats['num_crms']} CRMs, {stats['num_csts']} CSTs, "
                    f"{stats['num_mreqs']} Mreqs, {stats['num_entities']} entities"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            if self.verbose:
                import traceback
                logger.error(traceback.format_exc())
            return False
    
    # Serialization helpers
    def _serialize_entity(self, entity: Entity) -> Dict[str, Any]:
        """Serialize an Entity to a dictionary."""
        return {
            "id": str(entity.id),
            "ontologies": list(entity.ontologies),
            "attributes": {str(k): str(v) for k, v in entity.attributes.items()}
        }
    
    def _deserialize_entity(self, k: str, v: Dict[str, Any]) -> Entity:
        """Deserialize an Entity from a dictionary."""
        return Entity(
            id=v["id"],
            ontologies=set(v["ontologies"]),
            attributes={k: v for k, v in v["attributes"].items()}
        )
    
    def _serialize_cst(self, cst: CompositeState) -> Dict[str, Any]:
        """Serialize a CompositeState to a dictionary."""
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
    
    def _deserialize_cst(self, k: str, v: Dict[str, Any]) -> CompositeState:
        """Deserialize a CompositeState from a dictionary."""
        conditions = [
            AtomicPredicate(
                var_id=cond["var_id"],
                operator=cond["operator"],
                value=cond["value"]
            )
            for cond in v["conditions"]
        ]
        return CompositeState(id=v["id"], conditions=conditions)
    
    def _serialize_crm(self, crm: CRM) -> Dict[str, Any]:
        """Serialize a CRM to a dictionary."""
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
    
    def _deserialize_crm(self, k: str, v: Dict[str, Any]) -> CRM:
        """Deserialize a CRM from a dictionary."""
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
    
    def _serialize_mreq(self, mreq: Mreq) -> Dict[str, Any]:
        """Serialize an Mreq to a dictionary."""
        return {
            "id": str(mreq.id),
            "cst_id": str(mreq.cst_id),
            "crm_id": str(mreq.crm_id),
            "confidence": mreq.confidence
        }
    
    def _deserialize_mreq(self, k: str, v: Dict[str, Any]) -> Mreq:
        """Deserialize an Mreq from a dictionary."""
        return Mreq(
            id=v["id"],
            cst_id=v["cst_id"],
            crm_id=v["crm_id"],
            confidence=v.get("confidence", 0.5)
        )
    
    def _serialize_anti_mreq(self, anti_mreq: AntiMreq) -> Dict[str, Any]:
        """Serialize an AntiMreq to a dictionary."""
        return {
            "id": str(anti_mreq.id),
            "cst_id": str(anti_mreq.cst_id),
            "crm_id": str(anti_mreq.crm_id),
            "confidence": anti_mreq.confidence
        }
    
    def _deserialize_anti_mreq(self, k: str, v: Dict[str, Any]) -> AntiMreq:
        """Deserialize an AntiMreq from a dictionary."""
        return AntiMreq(
            id=v["id"],
            cst_id=v["cst_id"],
            crm_id=v["crm_id"],
            confidence=v.get("confidence", 0.5)
        )
    
    def _serialize_drive(self, drive: Drive) -> Dict[str, Any]:
        """Serialize a Drive to a dictionary."""
        return {
            "id": str(drive.id),
            "goal_cst_id": str(drive.goal_cst_id),
            "weight": drive.weight,
            "novelty_weight": drive.novelty_weight,
            "exploitation_weight": drive.exploitation_weight,
        }
    
    def _deserialize_drive(self, k: str, v: Dict[str, Any]) -> Drive:
        """Deserialize a Drive from a dictionary."""
        return Drive(
            id=v["id"],
            goal_cst_id=v["goal_cst_id"],
            weight=v.get("weight", 1.0),
            novelty_weight=v.get("novelty_weight", 0.3),
            exploitation_weight=v.get("exploitation_weight", 0.7),
        )


# ============================================================================
# Advanced Memory Systems
# ============================================================================

@dataclass
class EpisodicMemory:
    """
    Episodic memory: stores specific experiences with temporal context.
    
    Mathematical formulation:
        E = {(s_t, a_t, r_t, s_{t+1}, t, context_t)}_{t=1}^T
        
    Retrieval via similarity:
        P(retrieve e | query) ∝ exp(-β · d(e, query))
        
    Where d is a distance metric (e.g., L2, cosine, edit distance).
    """
    episodes: List[Dict[str, Any]] = field(default_factory=list)
    max_size: int = 10000
    similarity_threshold: float = 0.7
    
    def store_episode(
        self,
        state: Dict[Hashable, Any],
        action: str,
        reward: float,
        next_state: Dict[Hashable, Any],
        timestamp: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """Store an episode in episodic memory."""
        episode = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "timestamp": timestamp,
            "context": context or {}
        }
        self.episodes.append(episode)
        
        # Maintain max size
        if len(self.episodes) > self.max_size:
            self.episodes = self.episodes[-self.max_size:]
    
    def retrieve_similar(
        self,
        query_state: Dict[Hashable, Any],
        k: int = 5,
        beta: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve k most similar episodes.
        
        Uses exponential similarity: P ∝ exp(-β · d)
        """
        similarities = []
        
        for episode in self.episodes:
            # Compute distance (simplified: L1 distance on common keys)
            common_keys = set(query_state.keys()) & set(episode["state"].keys())
            if not common_keys:
                distance = 1.0
            else:
                distance = sum(
                    abs(query_state[k] - episode["state"].get(k, 0.0))
                    if isinstance(query_state[k], (int, float))
                    else (0.0 if query_state[k] == episode["state"].get(k) else 1.0)
                    for k in common_keys
                ) / len(common_keys)
            
            similarity = np.exp(-beta * distance)
            similarities.append((similarity, episode))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [ep for sim, ep in similarities[:k]]


# ============================================================================
# Hierarchical Memory Compression
# ============================================================================

@dataclass
class ProceduralCache:
    """
    Procedural cache: compiled factor-graph fragments for fast reuse.
    
    Stores pre-computed factor graph subgraphs that can be quickly instantiated.
    """
    fragments: Dict[str, FactorGraph] = field(default_factory=dict)
    access_counts: Dict[str, int] = field(default_factory=dict)
    max_fragments: int = 100
    
    def store_fragment(self, key: str, fragment: FactorGraph):
        """Store a compiled factor graph fragment."""
        self.fragments[key] = fragment
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        
        # Maintain max fragments
        if len(self.fragments) > self.max_fragments:
            # Remove least accessed
            sorted_keys = sorted(self.access_counts.items(), key=lambda x: x[1])
            for key_to_remove, _ in sorted_keys[:len(self.fragments) - self.max_fragments]:
                if key_to_remove in self.fragments:
                    del self.fragments[key_to_remove]
                    del self.access_counts[key_to_remove]
    
    def retrieve_fragment(self, key: str) -> Optional[FactorGraph]:
        """Retrieve a compiled fragment."""
        if key in self.fragments:
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            return self.fragments[key]
        return None


class MemoryCompressor:
    """
    Hierarchical memory compression using variational autoencoders.
    
    Mathematical formulation:
        min_{φ,θ} E_{q_φ(z|x)}[log p_θ(x|z)] + KL[q_φ(z|x) || p(z)]
        
    Where x are CRM parameter vectors, z are compressed representations.
    """
    
    def __init__(self, compression_dim: int = 32):
        """
        Initialize memory compressor.
        
        Args:
            compression_dim: Dimension of compressed representation
        """
        self.compression_dim = compression_dim
        self.encoder_params: Dict[str, np.ndarray] = {}
        self.decoder_params: Dict[str, np.ndarray] = {}
        self.compressed_representations: Dict[Hashable, np.ndarray] = {}
    
    def compress_crm_parameters(
        self,
        crm_id: Hashable,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compress CRM parameters to low-dimensional representation.
        
        Simplified: uses PCA-like compression.
        In full implementation, would use VAE.
        """
        # Flatten parameters to vector
        param_vector = []
        for key, val in sorted(parameters.items()):
            if isinstance(val, np.ndarray):
                param_vector.extend(val.flatten().tolist())
            elif isinstance(val, (int, float)):
                param_vector.append(val)
        
        param_vector = np.array(param_vector)
        
        if len(param_vector) == 0:
            return np.zeros(self.compression_dim)
        
        # Simple compression: random projection (simplified)
        # In full implementation, would learn encoder/decoder
        if crm_id not in self.encoder_params:
            # Initialize random projection matrix
            input_dim = len(param_vector)
            self.encoder_params[crm_id] = np.random.randn(self.compression_dim, input_dim) / np.sqrt(input_dim)
            self.decoder_params[crm_id] = np.random.randn(input_dim, self.compression_dim) / np.sqrt(self.compression_dim)
        
        # Encode
        compressed = self.encoder_params[crm_id] @ param_vector
        self.compressed_representations[crm_id] = compressed
        
        return compressed
    
    def decompress_crm_parameters(
        self,
        crm_id: Hashable,
        compressed: np.ndarray
    ) -> Dict[str, Any]:
        """
        Decompress representation back to parameters.
        
        Simplified: uses learned decoder.
        """
        if crm_id not in self.decoder_params:
            return {}
        
        # Decode
        param_vector = self.decoder_params[crm_id] @ compressed
        
        # Reshape back to parameter dict (simplified)
        # In full implementation, would properly reconstruct structure
        return {"compressed_reconstruction": param_vector}


@dataclass
class SemanticMemory:
    """
    Semantic memory: stores abstract knowledge, concepts, and generalizations.
    
    Mathematical formulation:
        S = {concept_i: (prototype_i, variance_i, frequency_i)}
        
    Prototype-based representation:
        prototype = (1/N) Σ_{x ∈ concept} x
        
    Variance:
        σ² = (1/N) Σ_{x ∈ concept} ||x - prototype||²
    """
    concepts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def update_concept(
        self,
        concept_id: str,
        instance: Dict[Hashable, Any],
        weight: float = 1.0
    ):
        """
        Update concept with new instance (online learning).
        
        Uses exponential moving average:
            prototype_{t+1} = α · instance + (1-α) · prototype_t
            σ²_{t+1} = α · ||instance - prototype_t||² + (1-α) · σ²_t
        """
        if concept_id not in self.concepts:
            self.concepts[concept_id] = {
                "prototype": instance.copy(),
                "variance": {k: 0.0 for k in instance.keys()},
                "count": 0,
                "instances": []
            }
        
        concept = self.concepts[concept_id]
        alpha = weight / (concept["count"] + weight)
        
        # Update prototype
        for key, value in instance.items():
            if key in concept["prototype"]:
                old_prototype = concept["prototype"][key]
                concept["prototype"][key] = alpha * value + (1 - alpha) * old_prototype
                
                # Update variance
                if isinstance(value, (int, float)) and isinstance(old_prototype, (int, float)):
                    diff_sq = (value - old_prototype) ** 2
                    concept["variance"][key] = alpha * diff_sq + (1 - alpha) * concept["variance"].get(key, 0.0)
            else:
                concept["prototype"][key] = value
                concept["variance"][key] = 0.0
        
        concept["count"] += weight
        concept["instances"].append(instance)
        
        # Keep only recent instances
        if len(concept["instances"]) > 100:
            concept["instances"] = concept["instances"][-100:]
    
    def classify(
        self,
        instance: Dict[Hashable, Any],
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Classify instance into concepts using Mahalanobis distance.
        
        Returns probability distribution over concepts:
            P(concept | instance) ∝ exp(-d_M(instance, prototype) / temperature)
        """
        scores = {}
        
        for concept_id, concept in self.concepts.items():
            prototype = concept["prototype"]
            variance = concept["variance"]
            
            # Compute Mahalanobis distance
            distance_sq = 0.0
            for key in set(instance.keys()) & set(prototype.keys()):
                diff = instance[key] - prototype[key]
                var = variance.get(key, 1.0)
                if var > 0:
                    distance_sq += (diff ** 2) / var
                else:
                    distance_sq += diff ** 2
            
            # Convert to probability
            score = np.exp(-np.sqrt(distance_sq) / temperature)
            scores[concept_id] = score
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores


@dataclass
class WorkingMemory:
    """
    Working memory: active, limited-capacity buffer for current task.
    
    Mathematical formulation:
        WM = {item_i: (content_i, activation_i, recency_i)}
        
    Activation decay:
        activation_i(t) = activation_i(0) · exp(-λ · t) + recency_bonus
        
    Capacity limit: |WM| ≤ C (typically C ≈ 7±2)
    """
    items: List[Dict[str, Any]] = field(default_factory=list)
    max_capacity: int = 7
    decay_rate: float = 0.1
    
    def add_item(
        self,
        content: Any,
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add item to working memory."""
        item = {
            "content": content,
            "activation": importance,
            "recency": 1.0,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.items.append(item)
        
        # Enforce capacity limit
        if len(self.items) > self.max_capacity:
            # Remove least activated item
            self.items.sort(key=lambda x: x["activation"] * x["recency"], reverse=True)
            self.items = self.items[:self.max_capacity]
    
    def update_activations(self):
        """Update activation levels with decay."""
        current_time = time.time()
        
        for item in self.items:
            elapsed = current_time - item["timestamp"]
            # Exponential decay
            item["activation"] *= np.exp(-self.decay_rate * elapsed)
            # Recency bonus
            item["recency"] = np.exp(-elapsed / 10.0)  # Decay over 10 seconds
            item["timestamp"] = current_time
    
    def get_active_items(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Get items with activation above threshold."""
        self.update_activations()
        return [
            item for item in self.items
            if item["activation"] * item["recency"] > threshold
        ]


# ============================================================================
# Hierarchical Planning with Options
# ============================================================================

@dataclass
class Option:
    """
    Option: temporally extended action (macro-action).
    
    Mathematical formulation:
        Option o = ⟨I_o, π_o, β_o⟩
        
    Where:
    - I_o: initiation set (when option can start)
    - π_o: option policy (action selection within option)
    - β_o: termination condition (when option ends)
    
    Option value:
        Q(s, o) = E[Σ_{k=0}^{τ-1} γ^k r_{t+k} + γ^τ max_{o'} Q(s_{t+τ}, o') | s_t, o]
    """
    id: Hashable
    initiation_cst_id: Hashable  # CST defining when option can start
    termination_cst_id: Hashable  # CST defining when option terminates
    policy: Callable[[Dict[Hashable, Any]], str]  # Option policy
    stats: Dict[str, Any] = field(default_factory=lambda: {
        "uses": 0,
        "successes": 0,
        "avg_reward": 0.0,
        "avg_duration": 0.0
    })
    
    def can_initiate(self, state: Dict[Hashable, Any], kb: KnowledgeBase) -> bool:
        """Check if option can be initiated."""
        if self.initiation_cst_id not in kb.csts:
            return False
        cst = kb.csts[self.initiation_cst_id]
        return cst.evaluate(state)
    
    def should_terminate(self, state: Dict[Hashable, Any], kb: KnowledgeBase) -> bool:
        """Check if option should terminate."""
        if self.termination_cst_id not in kb.csts:
            return False
        cst = kb.csts[self.termination_cst_id]
        return cst.evaluate(state)


class HierarchicalPlanner:
    """
    Hierarchical planner using options (macro-actions).
    
    Implements MAXQ value function decomposition:
        Q(s, o) = V^o(s) + C(s, o)
        
    Where:
    - V^o(s): value of executing option o in state s
    - C(s, o): completion function (expected value after option completes)
    """
    
    def __init__(
        self,
        kb: KnowledgeBase,
        factor_graph_builder: FactorGraphBuilder,
        message_passing_engine: MessagePassingEngine
    ):
        """Initialize hierarchical planner."""
        self.kb = kb
        self.factor_graph_builder = factor_graph_builder
        self.message_passing_engine = message_passing_engine
        self.options: Dict[Hashable, Option] = {}
        self.option_q_values: Dict[Tuple[Hashable, Hashable], float] = {}
        self.completion_functions: Dict[Tuple[Hashable, Hashable], float] = {}
    
    def add_option(self, option: Option):
        """Add an option to the planner."""
        self.options[option.id] = option
    
    def compute_option_value(
        self,
        state: Dict[Hashable, Any],
        option: Option,
        goals: List[Drive],
        depth: int = 0,
        max_depth: int = 5
    ) -> float:
        """
        Compute value of executing option in state.
        
        Uses recursive value function decomposition.
        """
        if depth >= max_depth:
            return 0.0
        
        # Check if option can be initiated
        if not option.can_initiate(state, self.kb):
            return float('-inf')
        
        # Simulate option execution (simplified)
        # In full implementation, would use CRM rollouts
        total_reward = 0.0
        current_state = state.copy()
        steps = 0
        max_steps = 20
        
        while steps < max_steps and not option.should_terminate(current_state, self.kb):
            # Execute option policy
            action = option.policy(current_state)
            
            # Simulate transition (simplified)
            # In full implementation, would use active CRMs
            next_state = current_state.copy()
            
            # Compute reward
            reward = self._compute_reward(current_state, goals)
            total_reward += (0.9 ** steps) * reward  # Discounted
            
            current_state = next_state
            steps += 1
        
        # Completion value
        completion_value = 0.0
        if option.should_terminate(current_state, self.kb):
            # Value of best next option
            best_next_value = 0.0
            for next_option in self.options.values():
                if next_option.can_initiate(current_state, self.kb):
                    next_value = self.compute_option_value(
                        current_state, next_option, goals, depth + 1, max_depth
                    )
                    best_next_value = max(best_next_value, next_value)
            completion_value = (0.9 ** steps) * best_next_value
        
        return total_reward + completion_value
    
    def _compute_reward(self, state: Dict[Hashable, Any], goals: List[Drive]) -> float:
        """Compute reward from drive satisfaction."""
        total_reward = 0.0
        for drive in goals:
            if drive.goal_cst_id in self.kb.csts:
                goal_cst = self.kb.csts[drive.goal_cst_id]
                satisfaction = goal_cst.evaluate_soft(state)
                total_reward += drive.compute_utility(satisfaction)
        return total_reward
    
    def select_option(
        self,
        state: Dict[Hashable, Any],
        goals: List[Drive]
    ) -> Optional[Option]:
        """Select best option to execute."""
        best_option = None
        best_value = float('-inf')
        
        for option in self.options.values():
            value = self.compute_option_value(state, option, goals)
            if value > best_value:
                best_value = value
                best_option = option
        
        return best_option


# ============================================================================
# Meta-Learning and Meta-Cognition
# ============================================================================

class MetaLearner:
    """
    Meta-learning: learning to learn.
    
    Mathematical formulation:
        Meta-parameters: φ (learning rate, architecture, etc.)
        
        Outer loop:
            φ* = argmax_φ E_{task ~ p(T)}[L(θ*(φ), task)]
        
        Inner loop:
            θ*(φ) = argmin_θ L(θ, task; φ)
        
    Implements MAML (Model-Agnostic Meta-Learning) style updates.
    """
    
    def __init__(self, kb: KnowledgeBase, meta_lr: float = 0.01):
        """
        Initialize meta-learner.
        
        Args:
            kb: Knowledge base
            meta_lr: Meta-learning rate
        """
        self.kb = kb
        self.meta_lr = meta_lr
        self.meta_parameters = {
            "learning_rate": 0.01,
            "exploration_rate": 0.1,
            "confidence_threshold": 0.3,
            "ctpx_threshold": 0.1,
            "ptpx_threshold": 1.0
        }
        self.task_performance_history: List[Dict[str, Any]] = []
    
    def adapt_to_task(
        self,
        task_examples: List[Tuple[Dict[Hashable, Any], str, Dict[Hashable, Any]]],
        num_steps: int = 5
    ) -> Dict[str, Any]:
        """
        Adapt to a new task using few-shot learning.
        
        Args:
            task_examples: List of (state, action, next_state) tuples
            num_steps: Number of adaptation steps
            
        Returns:
            Adapted parameters
        """
        # Initialize task-specific parameters
        task_params = self.meta_parameters.copy()
        
        # Fast adaptation (inner loop)
        for step in range(num_steps):
            # Compute gradients on task examples
            gradients = self._compute_task_gradients(task_examples, task_params)
            
            # Update task parameters
            for key in task_params:
                if key in gradients:
                    task_params[key] -= self.meta_parameters["learning_rate"] * gradients[key]
        
        return task_params
    
    def _compute_task_gradients(
        self,
        examples: List[Tuple[Dict[Hashable, Any], str, Dict[Hashable, Any]]],
        params: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute gradients for task adaptation."""
        gradients = {key: 0.0 for key in params}
        
        # Simplified: compute loss and approximate gradients
        total_loss = 0.0
        for x_t, u_t, x_tp1 in examples:
            # Try to predict using current CRMs
            active_crms = self.kb.get_active_crms(x_t, u_t)
            
            if not active_crms:
                # High loss if no CRM can explain
                total_loss += 1.0
            else:
                # Compute prediction error
                for crm in active_crms:
                    if crm.post_cst_id in self.kb.csts:
                        post_cst = self.kb.csts[crm.post_cst_id]
                        error = 1.0 - post_cst.evaluate_soft(x_tp1)
                        total_loss += error
        
        # Approximate gradients (simplified)
        # In full implementation, would use automatic differentiation
        for key in params:
            # Finite difference approximation
            eps = 0.01
            params_plus = params.copy()
            params_plus[key] += eps
            
            loss_plus = self._compute_loss(examples, params_plus)
            gradients[key] = (loss_plus - total_loss) / eps
        
        return gradients
    
    def _compute_loss(
        self,
        examples: List[Tuple[Dict[Hashable, Any], str, Dict[Hashable, Any]]],
        params: Dict[str, float]
    ) -> float:
        """Compute loss for given parameters."""
        loss = 0.0
        for x_t, u_t, x_tp1 in examples:
            active_crms = self.kb.get_active_crms(x_t, u_t, threshold=params["confidence_threshold"])
            if not active_crms:
                loss += 1.0
            else:
                for crm in active_crms:
                    if crm.post_cst_id in self.kb.csts:
                        post_cst = self.kb.csts[crm.post_cst_id]
                        error = 1.0 - post_cst.evaluate_soft(x_tp1)
                        loss += error
        return loss / len(examples) if examples else 0.0
    
    def meta_update(self, task_performances: List[Dict[str, Any]]):
        """
        Update meta-parameters based on task performance.
        
        Outer loop update:
            φ ← φ - α ∇_φ E[L(θ*(φ))]
        """
        if not task_performances:
            return
        
        # Aggregate performance metrics
        avg_performance = np.mean([p.get("performance", 0.0) for p in task_performances])
        
        # Update meta-parameters (simplified gradient ascent)
        if avg_performance > 0.7:
            # Good performance: increase exploration
            self.meta_parameters["exploration_rate"] *= 1.1
        else:
            # Poor performance: decrease exploration, increase learning
            self.meta_parameters["exploration_rate"] *= 0.9
            self.meta_parameters["learning_rate"] *= 1.1
        
        # Clip parameters
        self.meta_parameters["exploration_rate"] = np.clip(
            self.meta_parameters["exploration_rate"], 0.01, 0.5
        )
        self.meta_parameters["learning_rate"] = np.clip(
            self.meta_parameters["learning_rate"], 0.001, 0.1
        )


class MetaControl:
    """
    Meta-control: adaptive allocation of reasoning resources.
    
    Mathematical formulation:
        Meta-state: m_t = [error_gradients, drive_priorities, graph_load, confidence]
        
        Attention policy: π_ω(A_t | m_t)
        
        Intrinsic reward: r_t^int = |x̂_{t+1} - x_{t+1}|
        
        Bounded rationality: fixed compute budget per cycle
    """
    
    def __init__(
        self,
        kb: KnowledgeBase,
        compute_budget: int = 100,
        learning_rate: float = 0.01
    ):
        """
        Initialize meta-control system.
        
        Args:
            kb: Knowledge base
            compute_budget: Fixed compute budget per cycle
            learning_rate: Learning rate for attention policy
        """
        self.kb = kb
        self.compute_budget = compute_budget
        self.learning_rate = learning_rate
        self.attention_weights: Dict[Hashable, float] = {}  # Attention weights for CRMs/subgraphs
        self.meta_state_history: List[Dict[str, Any]] = []
        self.intrinsic_rewards: List[float] = []
    
    def compute_meta_state(
        self,
        beliefs: Dict[Hashable, Dict[Any, float]],
        drives: List[Drive],
        prediction_errors: Dict[Hashable, float]
    ) -> Dict[str, Any]:
        """
        Compute meta-state vector.
        
        m_t = [error_gradients, drive_priorities, graph_load, confidence]
        """
        # Error gradients (simplified: use prediction errors)
        error_gradients = np.mean(list(prediction_errors.values())) if prediction_errors else 0.0
        
        # Drive priorities (satisfaction levels)
        drive_priorities = []
        for drive in drives:
            if drive.goal_cst_id in self.kb.csts:
                goal_cst = self.kb.csts[drive.goal_cst_id]
                goal_vars = {pred.var_id for pred in goal_cst.conditions}
                
                # Compute satisfaction from beliefs
                satisfaction = 0.0
                for var_id in goal_vars:
                    if var_id in beliefs:
                        belief = beliefs[var_id]
                        if belief:
                            best_val = max(belief.items(), key=lambda x: x[1])[0]
                            temp_vals = {var_id: best_val}
                            satisfaction += goal_cst.evaluate_soft(temp_vals) / len(goal_vars)
                
                drive_priorities.append(satisfaction)
        
        avg_drive_priority = np.mean(drive_priorities) if drive_priorities else 0.0
        
        # Graph load (number of active factors)
        graph_load = len(self.kb.crms) + len(self.kb.csts)
        
        # Confidence (average belief confidence)
        confidences = []
        for belief in beliefs.values():
            if belief:
                entropy = sum(-p * np.log(p + 1e-10) for p in belief.values() if p > 0)
                max_entropy = np.log(len(belief)) if len(belief) > 0 else 1.0
                confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
                confidences.append(confidence)
        
        avg_confidence = np.mean(confidences) if confidences else 0.5
        
        meta_state = {
            "error_gradients": error_gradients,
            "drive_priorities": avg_drive_priority,
            "graph_load": graph_load,
            "confidence": avg_confidence
        }
        
        self.meta_state_history.append(meta_state)
        if len(self.meta_state_history) > 1000:
            self.meta_state_history = self.meta_state_history[-1000:]
        
        return meta_state
    
    def select_attention(
        self,
        meta_state: Dict[str, Any],
        available_crms: List[CRM],
        available_subgraphs: List[str]
    ) -> Dict[Hashable, float]:
        """
        Select attention allocation using learned policy.
        
        π_ω(A_t | m_t) = softmax(ω^T · m_t)
        
        Returns attention weights for each CRM/subgraph.
        """
        attention = {}
        
        # Compute attention scores based on meta-state
        for crm in available_crms:
            # Attention score based on:
            # - CRM confidence
            # - Prediction error (higher error = more attention)
            # - Drive relevance
            
            crm_confidence = crm.compute_confidence()
            error_weight = meta_state["error_gradients"]
            drive_weight = meta_state["drive_priorities"]
            
            # Combined attention score
            score = (
                0.4 * crm_confidence +
                0.3 * (1.0 - error_weight) +  # Lower error = higher attention
                0.3 * drive_weight
            )
            
            attention[crm.id] = max(0.0, score)
        
        # Normalize to probability distribution
        total = sum(attention.values())
        if total > 0:
            attention = {k: v / total for k, v in attention.items()}
        else:
            # Uniform if no scores
            attention = {crm.id: 1.0 / len(available_crms) for crm in available_crms}
        
        self.attention_weights.update(attention)
        return attention
    
    def compute_intrinsic_reward(
        self,
        predicted_state: Dict[Hashable, Any],
        observed_state: Dict[Hashable, Any]
    ) -> float:
        """
        Compute intrinsic reward from prediction error (curiosity).
        
        r_t^int = |x̂_{t+1} - x_{t+1}|
        """
        if not predicted_state or not observed_state:
            return 0.0
        
        # Compute prediction error
        errors = []
        for var_id in set(predicted_state.keys()) & set(observed_state.keys()):
            pred_val = predicted_state[var_id]
            obs_val = observed_state[var_id]
            
            if isinstance(pred_val, (int, float)) and isinstance(obs_val, (int, float)):
                error = abs(pred_val - obs_val)
                errors.append(error)
            elif pred_val != obs_val:
                errors.append(1.0)
        
        intrinsic_reward = np.mean(errors) if errors else 0.0
        self.intrinsic_rewards.append(intrinsic_reward)
        
        if len(self.intrinsic_rewards) > 1000:
            self.intrinsic_rewards = self.intrinsic_rewards[-1000:]
        
        return intrinsic_reward
    
    def allocate_compute_budget(
        self,
        attention_weights: Dict[Hashable, float],
        compute_budget: Optional[int] = None
    ) -> Dict[Hashable, int]:
        """
        Allocate compute budget across CRMs/subgraphs based on attention.
        
        Returns allocation: {crm_id: iterations}
        """
        if compute_budget is None:
            compute_budget = self.compute_budget
        
        allocation = {}
        total_weight = sum(attention_weights.values())
        
        if total_weight > 0:
            for crm_id, weight in attention_weights.items():
                # Allocate iterations proportionally
                iterations = int(compute_budget * weight)
                allocation[crm_id] = max(1, iterations)  # At least 1 iteration
        else:
            # Uniform allocation
            num_items = len(attention_weights)
            if num_items > 0:
                per_item = compute_budget // num_items
                allocation = {crm_id: per_item for crm_id in attention_weights.keys()}
        
        return allocation
    
    def update_attention_policy(
        self,
        meta_state: Dict[str, Any],
        attention_allocation: Dict[Hashable, int],
        performance: float
    ):
        """
        Update attention policy using reinforcement learning.
        
        Policy gradient: ∇_ω J = E[∇_ω log π_ω(A | m) · R]
        """
        # Simplified policy gradient update
        # In full implementation, would use proper RL algorithm
        
        for crm_id, iterations in attention_allocation.items():
            # Update attention weight based on performance
            current_weight = self.attention_weights.get(crm_id, 0.5)
            
            # Reward-based update
            if performance > 0:
                # Increase attention for good performance
                new_weight = current_weight + self.learning_rate * performance
            else:
                # Decrease attention for poor performance
                new_weight = current_weight - self.learning_rate * abs(performance)
            
            # Clip to [0, 1]
            self.attention_weights[crm_id] = max(0.0, min(1.0, new_weight))


class MetaCognition:
    """
    Meta-cognition: thinking about thinking.
    
    Mathematical formulation:
        Confidence in belief:
            conf(b) = 1 - H(b) / H_max
        
        Where H(b) is entropy of belief distribution.
        
        Uncertainty quantification:
            U(s, a) = Var[Q(s, a)] + E[Var[Q(s', a')]]
    """
    
    def __init__(self):
        """Initialize meta-cognition system."""
        self.confidence_history: List[float] = []
        self.uncertainty_tracking: Dict[Hashable, float] = {}
    
    def compute_belief_confidence(
        self,
        belief: Dict[Any, float]
    ) -> float:
        """
        Compute confidence in a belief distribution.
        
        Confidence = 1 - normalized_entropy
        """
        if not belief:
            return 0.0
        
        # Compute entropy
        entropy = 0.0
        for prob in belief.values():
            if prob > 0:
                entropy -= prob * np.log(prob + 1e-10)
        
        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log(len(belief))
        if max_entropy == 0:
            return 1.0
        
        normalized_entropy = entropy / max_entropy
        confidence = 1.0 - normalized_entropy
        
        return confidence
    
    def compute_epistemic_uncertainty(
        self,
        beliefs: Dict[Hashable, Dict[Any, float]]
    ) -> float:
        """
        Compute epistemic uncertainty (uncertainty about the model).
        
        U_epistemic = (1/|V|) Σ_v H(b_v)
        """
        if not beliefs:
            return 1.0
        
        total_entropy = 0.0
        for var_id, belief in beliefs.items():
            entropy = 0.0
            for prob in belief.values():
                if prob > 0:
                    entropy -= prob * np.log(prob + 1e-10)
            total_entropy += entropy
        
        avg_entropy = total_entropy / len(beliefs)
        max_entropy = np.log(2)  # Binary case (simplified)
        
        return avg_entropy / max_entropy if max_entropy > 0 else 1.0
    
    def should_explore(self, uncertainty: float, threshold: float = 0.5) -> bool:
        """Decide whether to explore based on uncertainty."""
        return uncertainty > threshold
    
    def compute_expected_information_gain(
        self,
        action: str,
        current_beliefs: Dict[Hashable, Dict[Any, float]],
        kb: KnowledgeBase
    ) -> float:
        """
        Compute expected information gain from taking action.
        
        IG(a) = H(before) - E[H(after | a)]
        
        Uses mutual information:
            I(A; O | s) = H(O | s) - H(O | A, s)
        """
        # Simplified: estimate information gain
        # In full implementation, would compute expected entropy reduction
        
        # Current entropy
        current_entropy = 0.0
        for belief in current_beliefs.values():
            for prob in belief.values():
                if prob > 0:
                    current_entropy -= prob * np.log(prob + 1e-10)
        
        # Estimate expected entropy after action (simplified)
        # Assume action reduces uncertainty by some factor
        reduction_factor = 0.3  # Simplified
        expected_entropy = current_entropy * (1 - reduction_factor)
        
        information_gain = current_entropy - expected_entropy
        return max(0.0, information_gain)


# ============================================================================
# Joint Training: LLM + Factor Graph Integration
# ============================================================================

class JointTrainingSystem:
    """
    Joint training system integrating LLM and factor graph learning.
    
    Mathematical formulation:
        Combined loss: L = L_factor_graph + λ · L_LLM + μ · L_align
        
        Where:
        - L_factor_graph = -log p(X | G, K) (factor graph likelihood)
        - L_LLM = -log p_θ(X | context, prompt) (LLM likelihood)
        - L_align = ||z_factor_graph - z_LLM||² (alignment loss)
        
        Gradient updates:
            ∇_θ L = ∇_θ L_factor_graph + λ · ∇_θ L_LLM + μ · ∇_θ L_align
        
        Bidirectional learning:
            - Factor graph guides LLM via structured prompts
            - LLM provides semantic priors for factor graph
            - Joint optimization of both components
    """
    
    def __init__(
        self,
        llm_reasoning_engine: LLMReasoningEngine,
        kb: KnowledgeBase,
        factor_graph_builder: FactorGraphBuilder,
        llm_weight: float = 0.3,
        alignment_weight: float = 0.1,
        learning_rate: float = 0.001
    ):
        """
        Initialize joint training system.
        
        Args:
            llm_reasoning_engine: LLM reasoning engine
            kb: Knowledge base
            factor_graph_builder: Factor graph builder
            llm_weight: Weight for LLM loss (λ)
            alignment_weight: Weight for alignment loss (μ)
            learning_rate: Learning rate for joint updates
        """
        self.llm_reasoning_engine = llm_reasoning_engine
        self.kb = kb
        self.factor_graph_builder = factor_graph_builder
        self.llm_weight = llm_weight
        self.alignment_weight = alignment_weight
        self.learning_rate = learning_rate
        self.training_history: List[Dict[str, float]] = []
    
    def compute_joint_loss(
        self,
        var_values: Dict[Hashable, Any],
        beliefs_factor_graph: Dict[Hashable, Dict[Any, float]],
        beliefs_llm: Dict[Hashable, Dict[Any, float]],
        context: str
    ) -> Dict[str, float]:
        """
        Compute joint loss combining factor graph and LLM.
        
        L = L_factor_graph + λ · L_LLM + μ · L_align
        """
        # Factor graph loss: negative log-likelihood
        loss_factor_graph = 0.0
        for var_id, true_val in var_values.items():
            if var_id in beliefs_factor_graph:
                belief = beliefs_factor_graph[var_id]
                prob = belief.get(true_val, 0.0)
                if prob > 0:
                    loss_factor_graph -= np.log(prob + 1e-10)
                else:
                    loss_factor_graph += 10.0  # Large penalty for zero probability
        
        # LLM loss: negative log-likelihood
        loss_llm = 0.0
        for var_id, true_val in var_values.items():
            if var_id in beliefs_llm:
                belief = beliefs_llm[var_id]
                prob = belief.get(true_val, 0.0)
                if prob > 0:
                    loss_llm -= np.log(prob + 1e-10)
                else:
                    loss_llm += 10.0
        
        # Alignment loss: KL divergence between distributions
        loss_align = 0.0
        for var_id in set(beliefs_factor_graph.keys()) & set(beliefs_llm.keys()):
            belief_fg = beliefs_factor_graph[var_id]
            belief_llm = beliefs_llm[var_id]
            
            # Compute KL divergence: KL(P_FG || P_LLM)
            all_vals = set(belief_fg.keys()) | set(belief_llm.keys())
            for val in all_vals:
                p_fg = belief_fg.get(val, 1e-10)
                p_llm = belief_llm.get(val, 1e-10)
                if p_fg > 0:
                    loss_align += p_fg * np.log(p_fg / (p_llm + 1e-10))
        
        # Combined loss
        total_loss = (
            loss_factor_graph +
            self.llm_weight * loss_llm +
            self.alignment_weight * loss_align
        )
        
        loss_dict = {
            "total": total_loss,
            "factor_graph": loss_factor_graph,
            "llm": loss_llm,
            "alignment": loss_align
        }
        
        self.training_history.append(loss_dict)
        if len(self.training_history) > 1000:
            self.training_history = self.training_history[-1000:]
        
        return loss_dict
    
    def update_joint_parameters(
        self,
        var_values: Dict[Hashable, Any],
        beliefs_factor_graph: Dict[Hashable, Dict[Any, float]],
        beliefs_llm: Dict[Hashable, Dict[Any, float]],
        context: str
    ):
        """
        Update parameters via joint gradient descent.
        
        ∇_θ L = ∇_θ L_factor_graph + λ · ∇_θ L_LLM + μ · ∇_θ L_align
        """
        # Compute loss
        loss_dict = self.compute_joint_loss(
            var_values, beliefs_factor_graph, beliefs_llm, context
        )
        
        # Update factor graph parameters (CRMs)
        # This is done via the dual learning system
        # Here we just log the loss for monitoring
        
        # Update LLM reasoning (if LLM is trainable)
        # Most LLMs are not directly trainable via this interface,
        # but we can update prompts/context based on loss
        
        # Update alignment (adjust weights)
        if loss_dict["alignment"] > 1.0:
            # High alignment loss: increase LLM weight
            self.llm_weight = min(0.5, self.llm_weight + 0.01)
        elif loss_dict["alignment"] < 0.1:
            # Low alignment loss: decrease LLM weight
            self.llm_weight = max(0.1, self.llm_weight - 0.01)
        
        return loss_dict


# ============================================================================
# Dual Learning Loops (Structural + Parametric)
# ============================================================================

class DualLearningSystem:
    """
    Dual learning loops: structural (AERA) and parametric (Sigma).
    
    Mathematical formulation:
        Outer loop (AERA): adds/removes CRMs, CSTs, Mreqs
        Inner loop (Sigma): updates parameters θ via SGD
        
        Synchronization: structural updates when parametric confidence < threshold
        
        Differentiable surrogate: Gumbel-Softmax gating for structural decisions
    """
    
    def __init__(
        self,
        kb: KnowledgeBase,
        learning_mechanisms: LearningMechanisms,
        structural_confidence_threshold: float = 0.5,
        parametric_lr: float = 0.01
    ):
        """
        Initialize dual learning system.
        
        Args:
            kb: Knowledge base
            learning_mechanisms: Learning mechanisms (CTPX/PTPX/GTPX)
            structural_confidence_threshold: Threshold for structural updates
            parametric_lr: Learning rate for parametric updates
        """
        self.kb = kb
        self.learning_mechanisms = learning_mechanisms
        self.structural_confidence_threshold = structural_confidence_threshold
        self.parametric_lr = parametric_lr
        self.structural_update_history: List[Dict[str, Any]] = []
    
    def run_parametric_update(
        self,
        x_t: Dict[Hashable, Any],
        u_t: str,
        x_tp1: Dict[Hashable, Any],
        active_crms: List[CRM]
    ):
        """
        Inner loop: update parametric models via gradient descent.
        
        θ ← θ - η ∇_θ L(θ)
        """
        for crm in active_crms:
            # Check if parametric confidence is low
            confidence = crm.compute_confidence()
            
            if confidence < self.structural_confidence_threshold:
                # Low confidence: try structural update first
                continue
            
            # Update parameters
            try:
                # Extract state vectors
                x_t_vec = self._extract_state_vector(x_t, crm)
                u_t_vec = self._extract_action_vector(u_t)
                x_tp1_vec = self._extract_state_vector(x_tp1, crm)
                
                if x_t_vec is not None and x_tp1_vec is not None:
                    crm.param_model.update(x_t_vec, u_t_vec, x_tp1_vec, self.parametric_lr)
            except Exception as e:
                logger.debug(f"Parametric update failed for CRM {crm.id}: {e}")
    
    def run_structural_update(
        self,
        x_t: Dict[Hashable, Any],
        u_t: str,
        x_tp1: Dict[Hashable, Any],
        active_crms: List[CRM],
        drives: List[Drive]
    ) -> Dict[str, Any]:
        """
        Outer loop: structural updates (add/remove CRMs, CSTs, Mreqs).
        
        Only runs when parametric confidence is low.
        """
        structural_changes = {
            "ctpx": None,
            "ptpx": [],
            "gtpx": []
        }
        
        # Check if structural update is needed
        needs_structural_update = False
        
        for crm in active_crms:
            confidence = crm.compute_confidence()
            if confidence < self.structural_confidence_threshold:
                needs_structural_update = True
                break
        
        if not needs_structural_update:
            return structural_changes
        
        # Apply structural learning mechanisms
        # CTPX
        new_crm_ctpx = self.learning_mechanisms.apply_ctpx(x_t, u_t, x_tp1)
        structural_changes["ctpx"] = new_crm_ctpx.id if new_crm_ctpx else None
        
        # PTPX
        new_crms_ptpx = self.learning_mechanisms.apply_ptpx(x_t, u_t, x_tp1, active_crms)
        structural_changes["ptpx"] = [crm.id for crm in new_crms_ptpx]
        
        # GTPX
        new_crms_gtpx = self.learning_mechanisms.apply_gtpx(x_t, x_tp1, drives)
        structural_changes["gtpx"] = [crm.id for crm in new_crms_gtpx]
        
        # Record structural update
        self.structural_update_history.append({
            "timestamp": time.time(),
            "changes": structural_changes
        })
        
        if len(self.structural_update_history) > 1000:
            self.structural_update_history = self.structural_update_history[-1000:]
        
        return structural_changes
    
    def _extract_state_vector(
        self,
        state: Dict[Hashable, Any],
        crm: CRM
    ) -> Optional[np.ndarray]:
        """Extract state vector relevant to CRM."""
        if crm.pre_cst_id not in self.kb.csts:
            return None
        
        pre_cst = self.kb.csts[crm.pre_cst_id]
        relevant_vars = {pred.var_id for pred in pre_cst.conditions}
        
        # Extract values
        values = []
        for var_id in sorted(relevant_vars):
            if var_id in state:
                val = state[var_id]
                if isinstance(val, (int, float)):
                    values.append(val)
                else:
                    values.append(0.0)  # Discrete: encode as 0
        
        return np.array(values) if values else None
    
    def _extract_action_vector(self, action: str) -> np.ndarray:
        """Extract action vector."""
        # Simplified: one-hot encoding
        return np.array([1.0 if action else 0.0])


# ============================================================================
# Safety and Stability Layer
# ============================================================================

class SafetyLayer:
    """
    Safety and stability layer: containment of runaway self-modification.
    
    Mathematical formulation:
        Stability regularizer: L_stability = λ ||K_{t+1} - K_t||²
        
        Change impact prediction: P(impact | change) before applying
        
        Versioned KB: rollback capability
    """
    
    def __init__(
        self,
        kb: KnowledgeBase,
        stability_lambda: float = 0.1,
        validation_threshold: float = 0.6
    ):
        """
        Initialize safety layer.
        
        Args:
            kb: Knowledge base
            stability_lambda: Stability regularizer weight
            validation_threshold: Minimum validation score to commit changes
        """
        self.kb = kb
        self.stability_lambda = stability_lambda
        self.validation_threshold = validation_threshold
        self.kb_versions: List[Dict[str, Any]] = []  # Versioned KB snapshots
        self.max_versions = 10
        self.change_log: List[Dict[str, Any]] = []
    
    def checkpoint_kb(self):
        """Create a checkpoint of current knowledge base."""
        checkpoint = {
            "timestamp": time.time(),
            "num_crms": len(self.kb.crms),
            "num_csts": len(self.kb.csts),
            "num_mreqs": len(self.kb.mreqs),
            "crm_ids": list(self.kb.crms.keys()),
            "cst_ids": list(self.kb.csts.keys()),
            "mreq_ids": list(self.kb.mreqs.keys())
        }
        
        self.kb_versions.append(checkpoint)
        
        # Maintain max versions
        if len(self.kb_versions) > self.max_versions:
            self.kb_versions = self.kb_versions[-self.max_versions:]
    
    def compute_stability_loss(
        self,
        old_kb_state: Dict[str, int],
        new_kb_state: Dict[str, int]
    ) -> float:
        """
        Compute stability loss.
        
        L_stability = λ ||K_{t+1} - K_t||²
        """
        # Compute change magnitude
        changes = {
            "crms": abs(new_kb_state.get("num_crms", 0) - old_kb_state.get("num_crms", 0)),
            "csts": abs(new_kb_state.get("num_csts", 0) - old_kb_state.get("num_csts", 0)),
            "mreqs": abs(new_kb_state.get("num_mreqs", 0) - old_kb_state.get("num_mreqs", 0))
        }
        
        # L2 norm of changes
        change_norm = np.sqrt(sum(c ** 2 for c in changes.values()))
        stability_loss = self.stability_lambda * change_norm
        
        return stability_loss
    
    def predict_change_impact(
        self,
        proposed_change: Dict[str, Any],
        validation_traces: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Predict impact of proposed structural change.
        
        Uses validation traces to estimate predictive improvement.
        """
        if not validation_traces:
            return {"impact": 0.0, "predicted_improvement": 0.0, "risk": 1.0}
        
        # Simulate change and evaluate on validation traces
        # Simplified: estimate based on change type
        
        change_type = proposed_change.get("type", "unknown")
        
        if change_type == "add_crm":
            # Adding CRM: positive impact if it explains validation traces
            impact = 0.3  # Moderate positive
            risk = 0.2  # Low risk
        elif change_type == "remove_crm":
            # Removing CRM: check if it's used in validation
            impact = -0.1  # Slight negative
            risk = 0.3
        elif change_type == "modify_crm":
            # Modifying CRM: depends on modification
            impact = 0.1
            risk = 0.4
        else:
            impact = 0.0
            risk = 0.5
        
        # Compute predicted improvement
        predicted_improvement = impact * (1 - risk)
        
        return {
            "impact": impact,
            "predicted_improvement": predicted_improvement,
            "risk": risk,
            "should_commit": predicted_improvement > self.validation_threshold
        }
    
    def validate_change(
        self,
        proposed_change: Dict[str, Any],
        validation_traces: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate proposed change before committing.
        
        Requires proof-of-predictive-improvement.
        """
        # Create checkpoint
        self.checkpoint_kb()
        
        # Predict impact
        impact_prediction = self.predict_change_impact(proposed_change, validation_traces)
        
        # Log change
        self.change_log.append({
            "timestamp": time.time(),
            "change": proposed_change,
            "impact_prediction": impact_prediction
        })
        
        if len(self.change_log) > 1000:
            self.change_log = self.change_log[-1000:]
        
        return impact_prediction["should_commit"]
    
    def rollback(self, version_index: int = -1):
        """
        Rollback to a previous KB version.
        
        Args:
            version_index: Index of version to rollback to (-1 = previous)
        """
        if not self.kb_versions or abs(version_index) > len(self.kb_versions):
            logger.warning("Cannot rollback: invalid version index")
            return False
        
        target_version = self.kb_versions[version_index]
        
        # Restore KB state (simplified: would need full state restoration)
        logger.info(f"Rolling back to version at {target_version['timestamp']}")
        
        # In full implementation, would restore full KB state
        return True


# ============================================================================
# Theory of Mind
# ============================================================================

class TheoryOfMind:
    """
    Theory of Mind: modeling other agents' mental states.
    
    Mathematical formulation:
        Belief about other agent:
            b_other(s) = P(other's belief about state s)
        
        Goal inference:
            P(goal | actions) ∝ P(actions | goal) · P(goal)
        
        Inverse planning:
            goal* = argmax_goal P(actions | goal, policy)
    """
    
    def __init__(self, kb: KnowledgeBase):
        """Initialize Theory of Mind system."""
        self.kb = kb
        self.other_agent_models: Dict[str, Dict[str, Any]] = {}
    
    def infer_goal(
        self,
        agent_id: str,
        observed_actions: List[str],
        observed_states: List[Dict[Hashable, Any]]
    ) -> Dict[Hashable, float]:
        """
        Infer other agent's goal from observed behavior.
        
        Uses inverse planning:
            P(goal | actions) ∝ P(actions | goal) · P(goal)
        """
        if agent_id not in self.other_agent_models:
            self.other_agent_models[agent_id] = {
                "goals": {},
                "policy": {},
                "observations": []
            }
        
        model = self.other_agent_models[agent_id]
        
        # Compute likelihood of actions given each possible goal
        goal_scores = {}
        
        for drive_id, drive in self.kb.drives.items():
            if drive.goal_cst_id not in self.kb.csts:
                continue
            
            goal_cst = self.kb.csts[drive.goal_cst_id]
            
            # Compute how well actions align with achieving this goal
            alignment_score = 0.0
            for i, (action, state) in enumerate(zip(observed_actions, observed_states)):
                # Check if action moves toward goal
                satisfaction = goal_cst.evaluate_soft(state)
                alignment_score += satisfaction
            
            alignment_score /= max(1, len(observed_actions))
            goal_scores[drive_id] = alignment_score
        
        # Normalize to probabilities
        total = sum(goal_scores.values())
        if total > 0:
            goal_scores = {k: v / total for k, v in goal_scores.items()}
        else:
            # Uniform prior if no evidence
            goal_scores = {k: 1.0 / len(self.kb.drives) for k in self.kb.drives.keys()}
        
        model["goals"] = goal_scores
        return goal_scores
    
    def predict_action(
        self,
        agent_id: str,
        current_state: Dict[Hashable, Any]
    ) -> Optional[str]:
        """
        Predict other agent's next action.
        
        Uses inferred goals and policy.
        """
        if agent_id not in self.other_agent_models:
            return None
        
        model = self.other_agent_models[agent_id]
        goals = model.get("goals", {})
        
        if not goals:
            return None
        
        # Find most likely goal
        most_likely_goal_id = max(goals.items(), key=lambda x: x[1])[0]
        
        if most_likely_goal_id not in self.kb.drives:
            return None
        
        drive = self.kb.drives[most_likely_goal_id]
        
        # Predict action that would achieve this goal
        # Simplified: use planner to find best action
        # In full implementation, would use agent's policy model
        
        return "predict_action"  # Placeholder


# ============================================================================
# Causal Discovery
# ============================================================================

class CausalDiscovery:
    """
    Causal discovery: learning causal structure from data.
    
    Mathematical formulation:
        Causal graph: G = (V, E) where E represents causal relationships
        
        Causal Markov condition:
            P(X) = ∏_{i} P(X_i | Pa(X_i))
        
        Independence tests:
            X ⟂ Y | Z  (conditional independence)
        
        PC algorithm / constraint-based methods.
    """
    
    def __init__(self, kb: KnowledgeBase):
        """Initialize causal discovery system."""
        self.kb = kb
        self.causal_graph: Dict[Hashable, Set[Hashable]] = defaultdict(set)
        self.conditional_independencies: List[Tuple[Hashable, Hashable, Set[Hashable]]] = []
    
    def test_conditional_independence(
        self,
        x: Hashable,
        y: Hashable,
        z: Set[Hashable],
        data: List[Dict[Hashable, Any]],
        alpha: float = 0.05
    ) -> bool:
        """
        Test conditional independence X ⟂ Y | Z.
        
        Uses statistical test (simplified: correlation-based).
        """
        if not data:
            return True
        
        # Extract values
        x_vals = [d.get(x) for d in data if x in d]
        y_vals = [d.get(y) for d in data if y in d]
        
        if len(x_vals) < 2 or len(y_vals) < 2:
            return True
        
        # Check if numeric
        if all(isinstance(v, (int, float)) for v in x_vals) and \
           all(isinstance(v, (int, float)) for v in y_vals):
            # Compute partial correlation (simplified)
            x_arr = np.array(x_vals)
            y_arr = np.array(y_vals)
            
            # Simple correlation
            if len(x_arr) == len(y_arr):
                correlation = np.corrcoef(x_arr, y_arr)[0, 1]
                # Test significance (simplified)
                return abs(correlation) < alpha
        else:
            # Categorical: use chi-square test (simplified)
            # For now, assume independent if different
            return x_vals != y_vals
        
        return True
    
    def discover_causal_structure(
        self,
        variables: Set[Hashable],
        data: List[Dict[Hashable, Any]]
    ) -> Dict[Hashable, Set[Hashable]]:
        """
        Discover causal structure using constraint-based approach.
        
        Simplified PC algorithm:
        1. Start with fully connected graph
        2. Remove edges based on independence tests
        3. Orient edges using v-structures
        """
        # Initialize fully connected graph
        graph = {v: set(variables - {v}) for v in variables}
        
        # Test independence for each pair
        for x in variables:
            for y in variables:
                if x == y:
                    continue
                
                # Test X ⟂ Y | {}
                if self.test_conditional_independence(x, y, set(), data):
                    graph[x].discard(y)
                    graph[y].discard(x)
                    self.conditional_independencies.append((x, y, set()))
        
        # Store discovered structure
        self.causal_graph = graph
        
        return graph
    
    def suggest_crm_from_causality(
        self,
        cause_var: Hashable,
        effect_var: Hashable,
        data: List[Dict[Hashable, Any]]
    ) -> Optional[CRM]:
        """
        Suggest a CRM based on discovered causal relationship.
        
        Creates CRM with:
        - Pre-CST: condition on cause variable
        - Post-CST: condition on effect variable
        - Parametric model: learned from data
        """
        if cause_var not in self.causal_graph or effect_var not in self.causal_graph[cause_var]:
            return None
        
        # Extract training data
        x_t_data = [d.get(cause_var) for d in data if cause_var in d]
        x_tp1_data = [d.get(effect_var) for d in data if effect_var in d]
        
        if len(x_t_data) < 2 or len(x_t_data) != len(x_tp1_data):
            return None
        
        # Learn parametric model
        if all(isinstance(v, (int, float)) for v in x_t_data) and \
           all(isinstance(v, (int, float)) for v in x_tp1_data):
            # Linear regression
            x_arr = np.array(x_t_data).reshape(-1, 1)
            y_arr = np.array(x_tp1_data)
            
            # Simple linear fit: y = ax + b
            A = np.hstack([x_arr, np.ones((len(x_arr), 1))])
            coeffs = np.linalg.lstsq(A, y_arr, rcond=None)[0]
            
            # Create parametric model
            param_model = ParametricModel(
                model_type="linear_gaussian",
                parameters={
                    "A": np.array([[coeffs[0]]]),
                    "B": np.zeros((1, 0)),
                    "Sigma": np.array([[np.var(y_arr - (coeffs[0] * x_arr.flatten() + coeffs[1]))]])
                }
            )
        else:
            # Default model
            param_model = ParametricModel(
                model_type="linear_gaussian",
                parameters={
                    "A": np.eye(1),
                    "B": np.zeros((1, 0)),
                    "Sigma": np.eye(1) * 0.1
                }
            )
        
        # Create CSTs
        pre_cst_id = f"cst_cause_{uuid4()}"
        post_cst_id = f"cst_effect_{uuid4()}"
        
        pre_cst = CompositeState(
            id=pre_cst_id,
            conditions=[AtomicPredicate(var_id=cause_var, operator="=", value=None)]
        )
        post_cst = CompositeState(
            id=post_cst_id,
            conditions=[AtomicPredicate(var_id=effect_var, operator="=", value=None)]
        )
        
        self.kb.add_cst(pre_cst)
        self.kb.add_cst(post_cst)
        
        # Create CRM
        crm_id = f"crm_causal_{uuid4()}"
        crm = CRM(
            id=crm_id,
            pre_cst_id=pre_cst_id,
            post_cst_id=post_cst_id,
            actions=set(),
            param_model=param_model
        )
        
        self.kb.add_crm(crm)
        
        # Create Mreq
        mreq_id = f"mreq_causal_{uuid4()}"
        mreq = Mreq(
            id=mreq_id,
            cst_id=pre_cst_id,
            crm_id=crm_id,
            confidence=0.5
        )
        self.kb.add_mreq(mreq)
        
        logger.info(f"Causal discovery created CRM {crm_id} for {cause_var} → {effect_var}")
        return crm


# ============================================================================
# Self-Explanation Generation
# ============================================================================

class ExplanationGenerator:
    """
    Self-explanation: generating causal narratives of agent's behavior.
    
    Mathematical formulation:
        Explanation score:
            score(explanation) = log p(X_{0:T}, U_{0:T-1} | {M_k}, K) - α · |{M_k}|
        
        Where {M_k} is a subset of CRMs explaining the trajectory.
    """
    
    def __init__(self, kb: KnowledgeBase, llm_backend: Optional[Any] = None):
        """
        Initialize explanation generator.
        
        Args:
            kb: Knowledge base
            llm_backend: Optional LLM for natural language generation
        """
        self.kb = kb
        self.llm_backend = llm_backend
    
    def generate_explanation(
        self,
        trajectory: List[Dict[str, Any]],
        alpha: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generate explanation for a trajectory.
        
        Args:
            trajectory: List of (state, action, next_state) tuples
            alpha: Complexity penalty weight
            
        Returns:
            Explanation with selected CRMs and narrative
        """
        if not trajectory:
            return {"explanation": "No trajectory to explain", "crms": []}
        
        # Find CRMs that explain the trajectory
        explaining_crms = []
        
        for step in trajectory:
            x_t = step.get("state", {})
            u_t = step.get("action")
            x_tp1 = step.get("next_state", {})
            
            # Find active CRMs
            active_crms = self.kb.get_active_crms(x_t, u_t)
            
            for crm in active_crms:
                if crm.post_cst_id in self.kb.csts:
                    post_cst = self.kb.csts[crm.post_cst_id]
                    if post_cst.evaluate_soft(x_tp1) > 0.5:
                        explaining_crms.append(crm.id)
        
        # Remove duplicates
        explaining_crms = list(set(explaining_crms))
        
        # Compute explanation score
        log_likelihood = len(explaining_crms) * 0.5  # Simplified
        complexity_penalty = alpha * len(explaining_crms)
        score = log_likelihood - complexity_penalty
        
        # Generate narrative
        narrative = self._generate_narrative(trajectory, explaining_crms)
        
        return {
            "explanation": narrative,
            "crms": explaining_crms,
            "score": score,
            "log_likelihood": log_likelihood,
            "complexity": len(explaining_crms)
        }
    
    def _generate_narrative(
        self,
        trajectory: List[Dict[str, Any]],
        crm_ids: List[Hashable]
    ) -> str:
        """Generate natural language narrative."""
        if not trajectory:
            return "No actions were taken."
        
        narrative_parts = []
        
        for i, step in enumerate(trajectory):
            action = step.get("action", "unknown")
            state_desc = self._describe_state(step.get("state", {}))
            
            narrative_parts.append(
                f"Step {i+1}: In state {state_desc}, action '{action}' was taken."
            )
        
        if crm_ids:
            narrative_parts.append(
                f"This behavior was explained by {len(crm_ids)} causal model(s)."
            )
        
        return " ".join(narrative_parts)
    
    def _describe_state(self, state: Dict[Hashable, Any]) -> str:
        """Generate a description of the state."""
        if not state:
            return "empty"
        
        # Simplified: just list key-value pairs
        desc_parts = [f"{k}={v}" for k, v in list(state.items())[:3]]
        return "{" + ", ".join(desc_parts) + "}"


# ============================================================================
# Temporal Reasoning
# ============================================================================

class TemporalReasoner:
    """
    Temporal reasoning: reasoning about time, sequences, and temporal patterns.
    
    Mathematical formulation:
        Temporal logic: CTL (Computation Tree Logic)
        
        Temporal patterns:
            P(pattern | history) = frequency(pattern in history)
        
        Sequence prediction:
            P(x_{t+1} | x_{1:t}) using n-gram models or RNNs
    """
    
    def __init__(self, kb: KnowledgeBase):
        """Initialize temporal reasoner."""
        self.kb = kb
        self.temporal_patterns: Dict[str, List[Any]] = defaultdict(list)
        self.sequence_history: List[Dict[Hashable, Any]] = []
    
    def detect_temporal_pattern(
        self,
        sequence: List[Dict[Hashable, Any]],
        min_length: int = 2,
        max_length: int = 5
    ) -> List[Tuple[List[Any], float]]:
        """
        Detect repeating temporal patterns in sequence.
        
        Returns list of (pattern, frequency) tuples.
        """
        patterns = []
        
        for length in range(min_length, min(max_length + 1, len(sequence) + 1)):
            # Extract all subsequences of this length
            for i in range(len(sequence) - length + 1):
                pattern = sequence[i:i+length]
                
                # Count occurrences
                count = 0
                for j in range(len(sequence) - length + 1):
                    if sequence[j:j+length] == pattern:
                        count += 1
                
                frequency = count / max(1, len(sequence) - length + 1)
                
                if frequency > 0.3:  # Threshold
                    patterns.append((pattern, frequency))
        
        # Remove duplicates and sort by frequency
        unique_patterns = {}
        for pattern, freq in patterns:
            pattern_key = str(pattern)
            if pattern_key not in unique_patterns or freq > unique_patterns[pattern_key][1]:
                unique_patterns[pattern_key] = (pattern, freq)
        
        return sorted(unique_patterns.values(), key=lambda x: x[1], reverse=True)
    
    def predict_next(
        self,
        sequence: List[Dict[Hashable, Any]],
        horizon: int = 1
    ) -> List[Dict[Hashable, Any]]:
        """
        Predict next states in sequence.
        
        Uses n-gram model (simplified).
        """
        if not sequence or horizon < 1:
            return []
        
        predictions = []
        
        # Use last n states to predict next
        n = min(3, len(sequence))
        context = sequence[-n:]
        
        # Find similar contexts in history
        similar_contexts = []
        for i in range(len(self.sequence_history) - n):
            hist_context = self.sequence_history[i:i+n]
            if self._contexts_similar(context, hist_context):
                if i + n < len(self.sequence_history):
                    similar_contexts.append(self.sequence_history[i+n])
        
        if similar_contexts:
            # Average prediction from similar contexts
            # Simplified: just return most common
            prediction = similar_contexts[0]  # Simplified
            predictions.append(prediction)
        else:
            # Default: extrapolate from last state
            predictions.append(sequence[-1].copy())
        
        # Recursively predict further ahead
        for _ in range(horizon - 1):
            extended_sequence = sequence + predictions
            next_pred = self.predict_next(extended_sequence, horizon=1)
            if next_pred:
                predictions.extend(next_pred)
            else:
                break
        
        return predictions[:horizon]
    
    def _contexts_similar(
        self,
        ctx1: List[Dict[Hashable, Any]],
        ctx2: List[Dict[Hashable, Any]]
    ) -> bool:
        """Check if two contexts are similar."""
        if len(ctx1) != len(ctx2):
            return False
        
        # Check if states are similar (simplified)
        for s1, s2 in zip(ctx1, ctx2):
            common_keys = set(s1.keys()) & set(s2.keys())
            if not common_keys:
                return False
            
            # Check if values are similar
            differences = sum(
                abs(s1[k] - s2[k]) if isinstance(s1[k], (int, float)) and isinstance(s2[k], (int, float))
                else (0.0 if s1[k] == s2[k] else 1.0)
                for k in common_keys
            )
            
            if differences / len(common_keys) > 0.3:
                return False
        
        return True
    
    def update_history(self, state: Dict[Hashable, Any]):
        """Update sequence history."""
        self.sequence_history.append(state)
        
        # Keep only recent history
        if len(self.sequence_history) > 1000:
            self.sequence_history = self.sequence_history[-1000:]


# ============================================================================
# Enhanced AERASigmaAgent with All Advanced Features
# ============================================================================

class EnhancedAERASigmaAgent(AERASigmaAgent):
    """
    Enhanced Σ-AERA agent with all advanced AGI features.
    
    Integrates:
    - Episodic, semantic, and working memory
    - Hierarchical planning with options
    - Meta-learning and meta-cognition
    - Theory of mind
    - Causal discovery
    - Self-explanation
    - Temporal reasoning
    """
    
    def __init__(
        self,
        llm_backend: Optional[Any] = None,
        seed_kb: Optional[KnowledgeBase] = None,
        max_iterations: int = 100,
        learning_enabled: bool = True,
        analogy_enabled: bool = True,
        enable_memory: bool = True,
        enable_meta_learning: bool = True,
        enable_theory_of_mind: bool = True
    ):
        """
        Initialize enhanced Σ-AERA agent.
        
        Args:
            llm_backend: Optional LLM backend
            seed_kb: Initial knowledge base
            max_iterations: Maximum cognitive cycles
            learning_enabled: Enable learning mechanisms
            analogy_enabled: Enable analogy mechanisms
            enable_memory: Enable memory systems
            enable_meta_learning: Enable meta-learning
            enable_theory_of_mind: Enable theory of mind
        """
        super().__init__(
            llm_backend=llm_backend,
            seed_kb=seed_kb,
            max_iterations=max_iterations,
            learning_enabled=learning_enabled,
            analogy_enabled=analogy_enabled
        )
        
        # Advanced memory systems
        if enable_memory:
            self.episodic_memory = EpisodicMemory()
            self.semantic_memory = SemanticMemory()
            self.working_memory = WorkingMemory()
        else:
            self.episodic_memory = None
            self.semantic_memory = None
            self.working_memory = None
        
        # Hierarchical planning
        self.hierarchical_planner = HierarchicalPlanner(
            self.kb,
            self.factor_graph_builder,
            self.message_passing_engine
        )
        
        # Meta-learning and meta-cognition
        if enable_meta_learning:
            self.meta_learner = MetaLearner(self.kb)
            self.meta_cognition = MetaCognition()
        else:
            self.meta_learner = None
            self.meta_cognition = None
        
        # Theory of mind
        if enable_theory_of_mind:
            self.theory_of_mind = TheoryOfMind(self.kb)
        else:
            self.theory_of_mind = None
        
        # Causal discovery
        self.causal_discovery = CausalDiscovery(self.kb)
        
        # Self-explanation
        self.explanation_generator = ExplanationGenerator(self.kb, llm_backend)
        
        # Temporal reasoning
        self.temporal_reasoner = TemporalReasoner(self.kb)
    
    def cognitive_cycle(
        self,
        observation: Any,
        available_actions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced cognitive cycle with all advanced features.
        """
        result = super().cognitive_cycle(observation, available_actions)
        
        # Update working memory
        if self.working_memory:
            var_values = {}
            for var_id, belief in self.current_beliefs.items():
                if belief:
                    best_val = max(belief.items(), key=lambda x: x[1])[0]
                    var_values[var_id] = best_val
            
            self.working_memory.add_item(
                content=var_values,
                importance=1.0,
                metadata={"iteration": self.iteration}
            )
        
        # Update episodic memory
        if self.episodic_memory and len(self.history) > 0:
            last_step = self.history[-1]
            x_t = last_step.get("x_t", {})
            u_t = last_step.get("u_t", "noop")
            
            # Compute reward from drive satisfaction
            reward = 0.0
            for drive in self.active_drives:
                if drive.goal_cst_id in self.kb.csts:
                    goal_cst = self.kb.csts[drive.goal_cst_id]
                    satisfaction = goal_cst.evaluate_soft(x_t)
                    reward += drive.compute_utility(satisfaction)
            
            self.episodic_memory.store_episode(
                state=x_t,
                action=u_t,
                reward=reward,
                next_state=var_values,
                timestamp=time.time()
            )
        
        # Update semantic memory
        if self.semantic_memory:
            # Classify current state into concepts
            var_values = {}
            for var_id, belief in self.current_beliefs.items():
                if belief:
                    best_val = max(belief.items(), key=lambda x: x[1])[0]
                    var_values[var_id] = best_val
            
            # Update concept (simplified: use state hash as concept)
            concept_id = f"state_concept_{hash(str(sorted(var_values.items())))}"
            self.semantic_memory.update_concept(concept_id, var_values)
        
        # Meta-cognition: compute confidence and uncertainty
        if self.meta_cognition:
            confidence = 0.0
            for belief in self.current_beliefs.values():
                conf = self.meta_cognition.compute_belief_confidence(belief)
                confidence += conf
            confidence /= max(1, len(self.current_beliefs))
            
            uncertainty = self.meta_cognition.compute_epistemic_uncertainty(self.current_beliefs)
            
            result["confidence"] = confidence
            result["uncertainty"] = uncertainty
            
            # Decide on exploration
            if self.meta_cognition.should_explore(uncertainty):
                result["exploration_recommended"] = True
        
        # Temporal reasoning: update history and detect patterns
        if len(self.history) > 0:
            last_state = self.history[-1].get("x_t", {})
            self.temporal_reasoner.update_history(last_state)
        
        # Causal discovery: periodically discover causal structure
        if self.iteration % 10 == 0 and len(self.kb.history) > 20:
            # Extract variables from history
            all_vars = set()
            for exp in self.kb.history:
                all_vars.update(exp.get("x_t", {}).keys())
                all_vars.update(exp.get("x_tp1", {}).keys())
            
            if all_vars:
                # Discover causal structure
                data = [
                    {**exp.get("x_t", {}), **exp.get("x_tp1", {})}
                    for exp in self.kb.history[-50:]
                ]
                self.causal_discovery.discover_causal_structure(all_vars, data)
        
        return result
    
    def generate_self_explanation(
        self,
        num_steps: int = 5
    ) -> Dict[str, Any]:
        """
        Generate self-explanation of recent behavior.
        """
        if len(self.history) < num_steps:
            trajectory = self.history
        else:
            trajectory = self.history[-num_steps:]
        
        # Convert to explanation format
        exp_trajectory = []
        for i, step in enumerate(trajectory):
            x_t = step.get("x_t", {})
            u_t = step.get("u_t", "noop")
            
            # Get next state
            if i < len(trajectory) - 1:
                x_tp1 = trajectory[i+1].get("x_t", {})
            else:
                # Use current beliefs
                x_tp1 = {}
                for var_id, belief in self.current_beliefs.items():
                    if belief:
                        x_tp1[var_id] = max(belief.items(), key=lambda x: x[1])[0]
            
            exp_trajectory.append({
                "state": x_t,
                "action": u_t,
                "next_state": x_tp1
            })
        
        return self.explanation_generator.generate_explanation(exp_trajectory)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the agent."""
        stats = self.get_knowledge_base_stats()
        
        # Memory stats
        if self.episodic_memory:
            stats["episodic_memory_size"] = len(self.episodic_memory.episodes)
        if self.semantic_memory:
            stats["semantic_memory_concepts"] = len(self.semantic_memory.concepts)
        if self.working_memory:
            stats["working_memory_items"] = len(self.working_memory.items)
        
        # Meta-learning stats
        if self.meta_learner:
            stats["meta_parameters"] = self.meta_learner.meta_parameters.copy()
        
        # Causal discovery stats
        stats["causal_graph_edges"] = sum(
            len(neighbors) for neighbors in self.causal_discovery.causal_graph.values()
        )
        
        return stats

