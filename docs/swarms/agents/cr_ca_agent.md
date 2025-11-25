# CR-CA Lite Agent

The CR-CA Lite Agent is a lightweight, pure-Python causal reasoning helper that
lets you sketch small structural causal models, propagate interventions, and
produce counterfactual scenarios without pulling in a heavy scientific stack.
It exposes the original `CRCAAgent` name for backwards compatibility, but the
implementation now focuses on the essentials:

- pure dictionaries for the causal graph (no NetworkX dependency)
- z-score standardization utilities
- counterfactual scenario generation via the built-in evolution operator
- causal chain, confounder, and adjustment-set helpers

If you need the earlier, fully featured SCM stack (gradient optimizers,
Bayesian inference, etc.), pin an older Swarms release. The current version
favors easy embedding inside lightweight agents or notebooks.

---

## Feature Snapshot

- **Graph construction** – add nodes and directed edges with optional strengths.
- **Pure Python topological sort** – works without external graph packages.
- **Do-operator propagation** – `_predict_outcomes` applies interventions in
  topological order and returns de-standardized values.
- **Counterfactual scenario generation** – explore multiple "what-if" targets
  with `generate_counterfactual_scenarios`.
- **Confounder and adjustment helpers** – detect common ancestors and simple
  back-door adjustment sets.
- **Simulation entry point** – `run` evolves the system for a few steps and
  bundles graph metadata for inspection.

Dependencies: `numpy` (for a couple of math utilities) and the Python standard
library (`typing`, `dataclasses`, `enum`).

---

## Quick Start

```python
from swarms.agents import CRCAAgent

# 1) Build the graph
agent = CRCAAgent(variables=["supply", "inventory", "backlog", "price"])
agent.add_causal_relationship("supply", "inventory", strength=0.8)
agent.add_causal_relationship("inventory", "backlog", strength=-0.6)
agent.add_causal_relationship("backlog", "price", strength=0.4)

# 2) Provide simple standardization stats (optional but recommended)
stats = {
    "supply": (100.0, 15.0),
    "inventory": (500.0, 60.0),
    "backlog": (80.0, 20.0),
    "price": (10.0, 1.5),
}
for var, (mean, std) in stats.items():
    agent.set_standardization_stats(var, mean=mean, std=std)

# 3) Predict a factual state plus an intervention
baseline = {"supply": 110.0, "inventory": 520.0, "backlog": 70.0, "price": 9.5}
intervention = {"supply": 90.0}  # do(supply = 90)
prediction = agent._predict_outcomes(baseline, intervention)

# 4) Generate lightweight counterfactuals
scenarios = agent.generate_counterfactual_scenarios(
    factual_state=baseline,
    target_variables=["price", "backlog"],
    max_scenarios=2,
)
```

Each `CounterfactualScenario` contains the intervention, expected outcomes, a
probability score (based on a Mahalanobis-style distance), and a natural
language reasoning string.

---

## Inspecting the Graph

```python
print(agent.get_nodes())
# ['supply', 'inventory', 'backlog', 'price']

print(agent.get_edges())
# [('supply', 'inventory'), ('inventory', 'backlog'), ('backlog', 'price')]

print(agent.identify_causal_chain("supply", "price"))
# ['supply', 'inventory', 'backlog', 'price']

print(agent.detect_confounders("inventory", "price"))
# [] in this toy example

print(agent.is_dag())
# True
```

Use `identify_adjustment_set` to obtain a minimal set of variables that block
back-door paths for simple treatment/outcome pairs.

---

## Counterfactual Scenario Walkthrough

```python
baseline = {"supply": 100.0, "inventory": 500.0, "backlog": 80.0, "price": 10.0}
agent.set_standardization_stats("price", mean=10.0, std=1.0)

for scenario in agent.generate_counterfactual_scenarios(
    factual_state=baseline,
    target_variables=["price"],
    max_scenarios=1,
):
    print(scenario.name)
    print("do(price) =", scenario.interventions["price"])
    print("expected backlog:", scenario.expected_outcomes.get("backlog"))
    print("probability:", scenario.probability)
```

The helper automatically scans a handful of z-score offsets (`-2σ` … `2σ`) for
each requested variable and evaluates the downstream impact using the evolution
operator. This keeps scenario exploration ergonomic without extra code.

---

## API Cheatsheet

| Method | Purpose |
| --- | --- |
| `add_causal_relationship(source, target, strength)` | Register a directed edge and optional effect strength. |
| `identify_causal_chain(start, end)` | Returns the shortest path between two nodes (BFS). |
| `detect_confounders(treatment, outcome)` | Finds ancestors common to both treatment and outcome. |
| `identify_adjustment_set(treatment, outcome)` | Suggests a simple back-door adjustment set. |
| `_predict_outcomes(factual_state, interventions)` | Applies the do-operator and propagates effects in z-space. |
| `generate_counterfactual_scenarios(...)` | Produces `CounterfactualScenario` objects for selected targets. |
| `analyze_causal_strength(source, target)` | Quick lookup of an edge’s stored strength and chain length. |
| `run(initial_state, target_variables, max_steps)` | Convenience wrapper that evolves the state and bundles summary info. |

> **Note:** methods prefixed with `_` remain public for practical reasons but are
> marked private to signal that their signatures may evolve faster.

---

## Differences vs. the Legacy Agent

| Area | Lite Version | Legacy Version (Deprecated) |
| --- | --- | --- |
| Dependencies | Standard library + NumPy | NetworkX, pandas, scipy, cvxpy, etc. |
| Graph storage | Python dict adjacency | NetworkX DiGraph with rich metadata |
| Model fitting | Manual/statistical strengths only | Weighted least squares, ridge, Bayesian priors |
| Optimization | Not included | Gradient, DP, evolutionary solvers |
| Explainability | Chain lookup + strength summary | Shapley values, integrated gradients, causal decomposition |

The lite build intentionally keeps the surface small so it can run anywhere the
swarms runtime runs (CLI tools, notebooks, or edge devices). If you need any of
the removed capabilities, import them from archival releases or implement them
on top of the exposed hooks.

---

## Tips & Troubleshooting

- **Always set standardization stats** before calling `_predict_outcomes` or
  `generate_counterfactual_scenarios`. Without them, values propagate in their
  raw scale, which can make mixed-unit graphs unstable.
- **Keep graphs acyclic.** `is_dag()` helps guard the execution order. The
  topological sort currently assumes no cycles exist.
- **Use strengths consistently.** Edge strengths act like linear coefficients.
  Signs (positive vs. negative) drive the qualitative behavior during
  propagation.
- **Scenario probability is heuristic.** The Mahalanobis-based probability is
  meant for ranking scenarios, not as a calibrated statistical guarantee.

---

## Minimal Example Script

Check out `examples/demos/logistics/crca_supply_shock_agent.py` for a runnable
walk-through that assembles a small supply/demand DAG, sets baseline stats, and
prints predicted interventions plus counterfactual scenarios.

---

Questions or suggestions? Open an issue in the Swarms repository referencing
the CR-CA Lite Agent. Contributions are welcome as long as they keep the lite
surface stable and dependency-free.
