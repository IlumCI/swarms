#!/usr/bin/env python3
"""
Deterministic demo script for the Time Engine example.

Runs a short simulation and a particle filter in-process using fixed rng seeds.
Keeps output minimal and machine-readable.
"""
import json
from engine import SimulacrumEngine


def run_demo():
    eng = SimulacrumEngine(db_path=":memory:")
    cfg = eng.get_default_universe_config()
    seed = cfg["initial"]

    sim_out = eng.run_simulation(seed_state=seed, horizon=4, ensemble=8, rng_seed=42)
    sim_id = sim_out.get("sim_id")

    # simple particle filter run using same seed for reproducibility
    observations = {0: {list(seed.keys())[0]: list(seed.values())[0]}}
    pf = eng.particle_filter(initial_state=seed, observations=observations, num_particles=32, horizon=2, rng_seed=42)

    out = {"sim_id": sim_id, "sim_summary": sim_out.get("summary"), "pf_summary": {"num_particles": pf.get("num_particles"), "particles_history_len": len(pf.get("particles_history", []))}}
    print(json.dumps(out))


if __name__ == "__main__":
    run_demo()


