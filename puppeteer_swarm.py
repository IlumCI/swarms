"""
PuppeteerSwarm: Dynamic Orchestration Multi-Agent System

A centralized policy network (puppeteer) that dynamically selects and sequences
specialized agents based on evolving task states, with reinforcement learning
to optimize both effectiveness and efficiency.

This implementation combines ChatDev-puppeteer's dynamic orchestration paradigm
with Swarms framework patterns.
"""

import copy
import math
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
    JAX_AVAILABLE = True
except (ImportError, KeyError, AttributeError):
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    optax = None

# Lazy import for Flax to avoid compatibility issues
_nn = None
_train_state = None

def _get_flax_nn():
    """Lazy import of Flax linen."""
    global _nn
    if _nn is None and JAX_AVAILABLE:
        try:
            from flax import linen as nn
            _nn = nn
        except (ImportError, KeyError, AttributeError, Exception):
            _nn = None
    return _nn

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoTokenizer = None

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.conversation import Conversation


class PathState(str, Enum):
    """States for a reasoning path."""

    INITIALIZED = "initialized"
    ACTIVE = "active"
    SPLITTING = "splitting"
    FINALIZED = "finalized"
    DISCARDED = "discarded"


class StateEncoder:
    """Encodes workflow state for policy network. Supports reward model embeddings (primary) and simple concatenation (fallback)."""

    def __init__(self, reward_model_name: Optional[str] = None, state_dim: int = 8192, use_reward_model: bool = True, device: str = "cpu"):
        self.state_dim = state_dim
        self.use_reward_model = use_reward_model
        self.device = device
        self.reward_model = None
        self.tokenizer = None
        self.reward_model_name = reward_model_name or "nvidia/Llama-3.1-Nemotron-70B-Reward-HF"
        if use_reward_model and TRANSFORMERS_AVAILABLE:
            try:
                self._initialize_reward_model()
            except Exception:
                self.use_reward_model = False

    def _initialize_reward_model(self):
        """Initialize reward model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.reward_model_name)
        self.reward_model = AutoModelForCausalLM.from_pretrained(self.reward_model_name)
        if self.device == "cuda":
            try:
                import torch
                self.reward_model = self.reward_model.to(self.device)
            except (ImportError, AttributeError):
                pass
        self.reward_model.eval()

    def encode(self, workflow_history: List[Dict], task: str) -> Tuple[np.ndarray, Optional[float]]:
        """Encode workflow state into fixed-size vector."""
        return self._encode_with_reward_model(workflow_history, task) if (self.use_reward_model and self.reward_model is not None) else self._encode_simple(workflow_history, task)

    def _encode_with_reward_model(self, workflow_history: List[Dict], task: str) -> Tuple[np.ndarray, Optional[float]]:
        """Encode using reward model."""
        try:
            messages = [{"role": "system", "content": task}] + [{"role": "assistant", "content": f"{entry.get('agent', 'agent')}: {entry.get('result', '')}"} for entry in workflow_history]
            if sum(len(m.get("content", "")) for m in messages) > 12000:
                messages = [messages[0]] + messages[-5:]
            tokenized = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt", max_length=4096)
            input_ids, attention_mask = tokenized["input_ids"].to(self.device), tokenized["attention_mask"].to(self.device)
            try:
                import torch
                with torch.no_grad():
                    outputs = self.reward_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, output_hidden_states=True)
                    reward = outputs.scores[0][0][0].item() if outputs.scores else None
                    if outputs.hidden_states and len(outputs.hidden_states) > 0:
                        state_vector = outputs.hidden_states[0][-1][:, -1, :].cpu().numpy().flatten()
                        if len(state_vector) != self.state_dim:
                            state_vector = state_vector[:self.state_dim] if len(state_vector) > self.state_dim else np.concatenate([state_vector, np.zeros(self.state_dim - len(state_vector))])
                        return state_vector, reward
                    return self._encode_simple(workflow_history, task)
            except (ImportError, AttributeError):
                return self._encode_simple(workflow_history, task)
        except Exception:
            return self._encode_simple(workflow_history, task)

    def _encode_simple(self, workflow_history: List[Dict], task: str) -> Tuple[np.ndarray, Optional[float]]:
        """Simple concatenation-based encoding."""
        text = " | ".join([task] + [f"{entry.get('agent', '')}: {entry.get('result', '')}" for entry in workflow_history])
        state_vector = np.zeros(self.state_dim, dtype=np.float32)
        for i, char in enumerate(text[:self.state_dim * 4]):
            state_vector[i % self.state_dim] += ord(char) / 1000.0
        norm = np.linalg.norm(state_vector)
        return (state_vector / norm, None) if norm > 0 else (state_vector, None)


class WorkflowTracker:
    """Tracks action history and state transitions."""

    def __init__(self):
        self.actions: List[Dict[str, Any]] = []
        self.valid_results: List[str] = []
        self.total_cost: float = 0.0
        self.total_tokens: int = 0

    def add_action(self, agent: Union[str, Agent], action: str, result: str, success: bool = True, cost: float = 0.0, tokens: int = 0):
        """Record an action."""
        agent_name = agent.agent_name if isinstance(agent, Agent) else str(agent)
        entry = {"agent": agent_name, "action": action, "result": result, "success": success, "cost": cost, "tokens": tokens}
        self.actions.append(entry)
        if success:
            self.valid_results.append(result)
            self.total_cost += cost
            self.total_tokens += tokens

    def get_state_summary(self) -> Dict[str, Any]:
        """Generate state summary for encoding."""
        return {"actions": self.actions, "valid_results": self.valid_results, "total_cost": self.total_cost, "total_tokens": self.total_tokens, "num_actions": len(self.actions), "num_successful": len(self.valid_results)}

    def get_workflow_history(self) -> List[Dict]:
        """Get workflow history for state encoding."""
        return self.actions.copy()

    def reset(self):
        """Reset workflow tracker."""
        self.actions, self.valid_results, self.total_cost, self.total_tokens = [], [], 0.0, 0


class RewardCalculator:
    """Calculates rewards for REINFORCE training."""

    def __init__(self, correctness_weight: float = 1.0, step_penalty_scale: float = 1.0, token_cost_scale: float = 0.001, agent_factors: Optional[Dict[str, float]] = None, max_steps: int = 10, growth_rate: float = 10.0):
        self.correctness_weight = correctness_weight
        self.step_penalty_scale = step_penalty_scale
        self.token_cost_scale = token_cost_scale
        self.agent_factors = agent_factors or {"web_search": -1.5, "terminator": 0.5, "default": -1.0}
        self.max_steps = max_steps
        self.growth_rate = growth_rate

    def calculate_reward(self, is_correct: bool, step: int, tokens: int = 0, agent_name: Optional[str] = None, task_completed: bool = True) -> float:
        """Calculate reward for a trajectory."""
        correctness_reward = self.correctness_weight if (task_completed and is_correct) else (-self.correctness_weight if task_completed else -0.5)
        normalized_step = (step + 1) / (self.max_steps + 1)
        step_cost = self.step_penalty_scale * math.log(1 + self.growth_rate * normalized_step) / math.log(1 + self.growth_rate)
        token_cost = self.token_cost_scale * (tokens / 100000.0)
        agent_factor = self.agent_factors.get(agent_name or "default", self.agent_factors["default"])
        return correctness_reward - step_cost - token_cost + (agent_factor * 0.1)

    def calculate_logarithmic_cost(self, step: int) -> float:
        """Calculate logarithmic cost for a step."""
        normalized_step = (step + 1) / (self.max_steps + 1)
        return self.step_penalty_scale * math.log(1 + self.growth_rate * normalized_step) / math.log(1 + self.growth_rate)


class ReasoningPath:
    """Tracks a single reasoning path through agent sequence."""

    def __init__(self, path_id: str, policy: Any, workflow_tracker: WorkflowTracker, global_info: Dict[str, Any], max_steps: int = 10):
        self.path_id = path_id
        self.policy = policy
        self.workflow_tracker = workflow_tracker
        self.global_info = global_info
        self.max_steps = max_steps
        self.agent_sequence: List[str] = []
        self.current_agent: Optional[Agent] = None
        self.next_agents: List[Agent] = []
        self.state = PathState.INITIALIZED
        self.answer: Optional[str] = None
        self.reward: Optional[float] = None
        self.trajectory: List[Dict] = []

    def step(self) -> bool:
        """Execute one step: current agent takes action, policy selects next."""
        if self.state == PathState.FINALIZED:
            return False
        if self.current_agent is not None:
            try:
                task = self.global_info.get("task", "")
                context = self.workflow_tracker.get_state_summary()
                task_with_context = f"{task}\n\nPrevious results:\n" + "\n".join([f"- {r[:200]}" for r in context["valid_results"][-3:]]) if context.get("valid_results") else task
                result = self.current_agent.run(task_with_context)
                self.workflow_tracker.add_action(agent=self.current_agent, action="reasoning", result=str(result), success=True, cost=0.0, tokens=0)
                if self._is_termination(result):
                    self.state, self.answer = PathState.FINALIZED, str(result)
                    return False
            except Exception as e:
                self.workflow_tracker.add_action(agent=self.current_agent, action="reasoning", result=f"Error: {e}", success=False)
        if len(self.agent_sequence) >= self.max_steps:
            self.state = PathState.FINALIZED
            return False
        try:
            available_agents = self.global_info.get("agents", [])
            self.next_agents = [available_agents[idx] for idx in self.policy.forward(self.workflow_tracker, self.global_info) if idx < len(available_agents)]
            if not self.next_agents:
                self.state = PathState.FINALIZED
                return False
            if len(self.next_agents) == 1:
                self.current_agent = self.next_agents[0]
                self.agent_sequence.append(self.current_agent.agent_name if isinstance(self.current_agent, Agent) else str(self.current_agent))
                self.state = PathState.ACTIVE
            else:
                self.state = PathState.SPLITTING
        except Exception:
            self.state = PathState.FINALIZED
            return False
        return True

    def _is_termination(self, result: str) -> bool:
        """Check if result indicates termination."""
        return any(kw in str(result).lower() for kw in ["task complete", "final answer", "terminate", "done", "<done>"])

    def split(self) -> List["ReasoningPath"]:
        """Create new paths when multiple agents selected."""
        if len(self.next_agents) <= 1:
            return []
        new_paths = []
        if self.next_agents:
            self.current_agent = self.next_agents[0]
            self.agent_sequence.append(self.current_agent.agent_name if isinstance(self.current_agent, Agent) else str(self.current_agent))
            self.state, self.next_agents = PathState.ACTIVE, []
        for agent in self.next_agents[1:]:
            new_tracker = WorkflowTracker()
            new_tracker.actions, new_tracker.valid_results = self.workflow_tracker.actions.copy(), self.workflow_tracker.valid_results.copy()
            new_path = ReasoningPath(path_id=str(uuid.uuid4()), policy=self.policy, workflow_tracker=new_tracker, global_info=copy.deepcopy(self.global_info), max_steps=self.max_steps)
            new_path.current_agent = agent
            new_path.agent_sequence = self.agent_sequence.copy()
            new_path.agent_sequence.append(agent.agent_name if isinstance(agent, Agent) else str(agent))
            new_path.state = PathState.ACTIVE
            new_paths.append(new_path)
        return new_paths

    def finalize(self, reward_calculator: RewardCalculator) -> float:
        """Finalize path and calculate reward."""
        self.state = PathState.FINALIZED
        self.reward = reward_calculator.calculate_reward(is_correct=self.global_info.get("is_correct", False), step=len(self.agent_sequence), tokens=self.workflow_tracker.total_tokens, agent_name=self.agent_sequence[-1] if self.agent_sequence else None, task_completed=self.answer is not None)
        return self.reward


class PuppeteerPolicyBase(ABC):
    """Abstract base class for policy networks."""

    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.0001, gamma: float = 0.99, max_agents_per_step: int = 3, selection_threshold: Optional[float] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_agents_per_step = max_agents_per_step
        self.selection_threshold = selection_threshold or (2.0 / action_dim if action_dim > 0 else 0.1)
        self.trajectories: List[List[Dict]] = []

    @abstractmethod
    def forward(self, workflow_tracker: WorkflowTracker, global_info: Dict[str, Any]) -> List[int]:
        """Select agents based on current state."""
        pass

    @abstractmethod
    def update(self):
        """Update policy using REINFORCE algorithm."""
        pass

    def select_agents_by_threshold(self, action_probs: Union[np.ndarray, jnp.ndarray]) -> List[int]:
        """Select agents using threshold-based filtering."""
        if JAX_AVAILABLE and isinstance(action_probs, jnp.ndarray):
            action_probs = np.array(action_probs)
        above_threshold = np.where(action_probs >= self.selection_threshold)[0]
        if len(above_threshold) == 0:
            return np.random.multinomial(1, action_probs).nonzero()[0].tolist()[:self.max_agents_per_step]
        return sorted(above_threshold, key=lambda i: action_probs[i], reverse=True)[:self.max_agents_per_step]


if JAX_AVAILABLE:
    nn = _get_flax_nn()
    if nn is not None:
        class MLP(nn.Module):
            """Flax MLP network for policy."""
            hidden_layers: List[int]
            action_dim: int
            @nn.compact
            def __call__(self, x):
                for hidden_dim in self.hidden_layers:
                    x = nn.Dense(hidden_dim)(x)
                    x = nn.relu(x)
                return nn.Dense(self.action_dim)(x)
    else:
        class MLP:
            """Dummy MLP class when Flax not available."""
            pass
else:
    class MLP:
        """Dummy MLP class when JAX not available."""
        pass

class PuppeteerPolicyJAX(PuppeteerPolicyBase):
    """JAX+Flax-based MLP policy network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int] = None, learning_rate: float = 0.0001, gamma: float = 0.99, max_agents_per_step: int = 3, selection_threshold: Optional[float] = None, rng_key: Optional[Any] = None):
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for PuppeteerPolicyJAX. Install with: pip install jax jaxlib optax")
        nn = _get_flax_nn()
        if nn is None:
            raise ImportError(
                "Flax is required for PuppeteerPolicyJAX but could not be imported. "
                "This may be due to Python 3.14 compatibility issues with Flax. "
                "Try: pip install flax, or use Python 3.11/3.12 instead."
            )
        super().__init__(state_dim, action_dim, learning_rate, gamma, max_agents_per_step, selection_threshold)
        self.hidden_layers = hidden_layers or [512, 128, 32]
        self.rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(42)
        self.network = MLP(hidden_layers=self.hidden_layers, action_dim=action_dim)
        self.params = self.network.init(self.rng_key, jnp.zeros((1, state_dim)))
        self.optimizer = optax.adam(learning_rate=learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        self.log_probs_history: List[jnp.ndarray] = []
        self.rewards_history: List[float] = []
        self.states_history: List[jnp.ndarray] = []
        self.actions_history: List[int] = []

    def forward(self, workflow_tracker: WorkflowTracker, global_info: Dict[str, Any]) -> List[int]:
        """Forward pass through network."""
        state_encoder = StateEncoder(use_reward_model=False, state_dim=self.state_dim)
        state_vector, _ = state_encoder.encode(workflow_tracker.get_workflow_history(), global_info.get("task", ""))
        state_jax = jnp.array(state_vector).reshape(1, -1)
        logits = self.network.apply(self.params, state_jax)
        action_probs = jax.nn.softmax(logits)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        action_idx = int(jax.random.categorical(subkey, logits[0]))
        log_prob = jax.nn.log_softmax(logits)[0, action_idx]
        self.log_probs_history.append(log_prob)
        self.states_history.append(state_jax)
        self.actions_history.append(action_idx)
        return self.select_agents_by_threshold(np.array(action_probs[0]))

    def update(self):
        """Update policy using REINFORCE."""
        if not self.log_probs_history or not self.states_history:
            return
        returns = []
        G = 0.0
        for reward in reversed(self.rewards_history):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns_jax = jnp.array(returns)
        if len(returns_jax) > 1:
            returns_jax = (returns_jax - returns_jax.mean()) / (returns_jax.std() + 1e-8)
        def compute_loss(params):
            log_probs_recomputed = [jax.nn.log_softmax(self.network.apply(params, state)[0])[action_idx] for state, action_idx in zip(self.states_history, self.actions_history)]
            return -jnp.mean(jnp.stack(log_probs_recomputed) * returns_jax)
        loss_value, grads = jax.value_and_grad(compute_loss)(self.params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        self.log_probs_history, self.rewards_history, self.states_history, self.actions_history = [], [], [], []

    def add_reward(self, reward: float):
        """Add reward to current trajectory."""
        self.rewards_history.append(reward)


class PuppeteerSwarm(BaseSwarm):
    """Main orchestrator for dynamic multi-agent reasoning. Manages multiple parallel reasoning paths and coordinates execution using a centralized policy network (puppeteer)."""

    def __init__(self, agents: List[Union[Agent, Callable]], name: Optional[str] = "PuppeteerSwarm", description: Optional[str] = "Dynamic orchestration multi-agent system", max_parallel_paths: int = 4, max_steps: int = 10, enable_training: bool = True, state_dim: int = 8192, use_reward_model: bool = True, reward_model_name: Optional[str] = None, learning_rate: float = 0.0001, gamma: float = 0.99, max_agents_per_step: int = 3, aggregation_method: str = "majority_vote", correctness_evaluator: Optional[Callable] = None, verbose: bool = False, *args, **kwargs):
        super().__init__(name=name, description=description, agents=agents, max_loops=1, *args, **kwargs)
        self.max_parallel_paths = max_parallel_paths
        self.max_steps = max_steps
        self.enable_training = enable_training
        self.aggregation_method = aggregation_method
        self.correctness_evaluator = correctness_evaluator
        self.verbose = verbose
        self.state_encoder = StateEncoder(reward_model_name=reward_model_name, state_dim=state_dim, use_reward_model=use_reward_model)
        action_dim = len(agents)
        if not JAX_AVAILABLE:
            raise ImportError("JAX and Flax are required for PuppeteerSwarm. Install with: pip install jax jaxlib flax optax")
        self.policy = PuppeteerPolicyJAX(state_dim=state_dim, action_dim=action_dim, learning_rate=learning_rate, gamma=gamma, max_agents_per_step=max_agents_per_step)
        self.reward_calculator = RewardCalculator(max_steps=max_steps)
        self.reasoning_paths: List[ReasoningPath] = []
        if NETWORKX_AVAILABLE:
            self.agent_graph = nx.DiGraph()
            for i, agent in enumerate(agents):
                self.agent_graph.add_node(agent.agent_name if isinstance(agent, Agent) else f"agent_{i}")
        else:
            self.agent_graph = None
        self.global_info: Dict[str, Any] = {"task": None, "agents": agents, "is_correct": False}
        self.final_answer: Optional[str] = None
        self.path_answers: List[str] = []

    def start(self, task: str):
        """Initialize reasoning with policy-selected initial agents."""
        self.global_info["task"], self.reasoning_paths, self.path_answers = task, [], []
        selected_indices = self.policy.forward(WorkflowTracker(), self.global_info)
        for idx in selected_indices[:self.max_parallel_paths]:
            if idx < len(self.agents):
                path = ReasoningPath(path_id=str(uuid.uuid4()), policy=self.policy, workflow_tracker=WorkflowTracker(), global_info=self.global_info.copy(), max_steps=self.max_steps)
                path.current_agent = self.agents[idx]
                path.agent_sequence.append(self.agents[idx].agent_name if isinstance(self.agents[idx], Agent) else f"agent_{idx}")
                path.state = PathState.ACTIVE
                self.reasoning_paths.append(path)

    def step(self):
        """Execute one step across all active paths."""
        new_paths = []
        for path in self.reasoning_paths:
            if path.state in [PathState.ACTIVE, PathState.INITIALIZED]:
                path.step()
                if path.state == PathState.SPLITTING:
                    split_paths = path.split()
                    new_paths.extend(split_paths[:self.max_parallel_paths - len(self.reasoning_paths)])
        self.reasoning_paths.extend(new_paths)
        if self.agent_graph is not None:
            for path in self.reasoning_paths:
                if len(path.agent_sequence) >= 2:
                    for i in range(len(path.agent_sequence) - 1):
                        if not self.agent_graph.has_edge(path.agent_sequence[i], path.agent_sequence[i + 1]):
                            self.agent_graph.add_edge(path.agent_sequence[i], path.agent_sequence[i + 1])

    def finalize(self) -> str:
        """Aggregate results, calculate rewards, update policy."""
        for path in self.reasoning_paths:
            if path.state != PathState.FINALIZED:
                path.finalize(self.reward_calculator)
            if path.answer:
                self.path_answers.append(path.answer)
        if self.correctness_evaluator:
            for path in self.reasoning_paths:
                self.global_info["is_correct"] = self.correctness_evaluator(path.answer, self.global_info["task"])
        else:
            self.global_info["is_correct"] = len(self.path_answers) > 0
        self.final_answer = self._aggregate_answers()
        if self.enable_training:
            for path in self.reasoning_paths:
                self.policy.trajectories.append([{"agent": agent_name, "step": i, "reward": path.reward if i == len(path.agent_sequence) - 1 else 0.0} for i, agent_name in enumerate(path.agent_sequence)])
            self.policy.update()
        return self.final_answer

    def _aggregate_answers(self) -> str:
        """Aggregate answers from multiple paths."""
        if not self.path_answers:
            return "No answer generated."
        if self.aggregation_method == "majority_vote":
            return self.path_answers[0]
        elif self.aggregation_method == "weighted":
            path_rewards = [path.reward or 0.0 for path in self.reasoning_paths if path.answer]
            return self.path_answers[path_rewards.index(max(path_rewards))] if path_rewards else self.path_answers[0]
        return self.path_answers[0]

    def run(self, task: str) -> str:
        """Main execution loop."""
        self.start(task)
        for iteration in range(self.max_steps * self.max_parallel_paths):
            if all(path.state == PathState.FINALIZED for path in self.reasoning_paths):
                break
            self.step()
        return self.finalize()

    def visualize_graph(self, output_path: Optional[str] = None):
        """Visualize agent graph."""
        if not NETWORKX_AVAILABLE or self.agent_graph is None:
            return
        try:
            import matplotlib.pyplot as plt
            nx.draw(self.agent_graph, nx.spring_layout(self.agent_graph), with_labels=True, node_color="lightblue", node_size=1500, font_size=10, font_weight="bold", arrows=True)
            plt.savefig(output_path) if output_path else plt.show()
            plt.close()
        except ImportError:
            pass

