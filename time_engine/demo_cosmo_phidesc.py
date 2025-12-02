#!/usr/bin/env python3
"""
Minimal ASTT Phase-1 demo: constructs a phidesc, initializes state, runs a short sim, runs PF.
"""
import json
from engine import SimulacrumEngine

def main():
    engine = SimulacrumEngine(db_path=":memory:")
    phidesc = {
        "type": "cosmology_lcdm",
        "physics_modes": {"gravity_model":"FLRW","field_model":"scalar_EFT","quantum_backreaction":False},
        "resolution_config": {"spatial_scale":"Mpc", "field_variables":["a","phi","phi_dot"]},
        "evolution_solver": {"time_integrator":"RK45","ensemble_size":8,"pruning_threshold":1e-6},
        "params": {"H0":2.2e-18, "Omega_m":0.315, "dt":1e13},
        "seed_config": {"t_init":1e-36, "a_0":1.0, "phi_0":1.0, "noise_seed":42},
        "boundary_model": {"type":"causal_sampling", "params":{"sigma":1e-6, "vars":["a","phi"]}}
    }
    x0 = engine.initialize_state(phidesc, phidesc.get("seed_config"))
    out = engine.run_simulation(seed_state=x0, horizon=3, ensemble=4, phidesc=phidesc, rng_seed=42)
    pf = engine.particle_filter(initial_state=x0, observations={0: {"a": x0.get("a")}}, num_particles=32, horizon=2, rng_seed=42)
    print(json.dumps({"sim": out.get("summary"), "pf": {"num_particles": pf.get("num_particles"), "particles_history_len": len(pf.get("particles_history", []))}}))

if __name__ == "__main__":
    main()


