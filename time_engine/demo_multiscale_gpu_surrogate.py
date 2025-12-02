#!/usr/bin/env python3
"""
Demo: start a multiscale sim via server using GPU adapter if available, else CPU; optionally use surrogate.
"""
import asyncio
import json
import sys

async def run(host="127.0.0.1", port=8765, use_surrogate=False):
    reader, writer = await asyncio.open_connection(host, port)
    ph = {
        "type": "cosmology_lcdm",
        "physics_modes": {"gravity_model": "FLRW"},
        "tiles": {"n_tiles": 2},
        "seed_config": {"a_0": 1.0, "phi_0": 1.0, "noise_seed": 42},
        "params": {"H0": 2.2e-18},
    }
    if use_surrogate:
        ph["field_model"] = "surrogate_torch"
        ph["surrogate"] = {"model_path": "surrogate.pt"}
    else:
        # prefer GPU if available
        ph["field_model"] = "pde_2d_cpu"
    payload = {"cmd": "start_multiscale_sim", "phidesc": ph, "horizon": 4, "ensemble": 1}
    writer.write(json.dumps(payload).encode() + b"\n")
    await writer.drain()
    line = await reader.readline()
    print("server:", line.decode().strip())
    # optionally subscribe tiles (if server provides job id)
    await asyncio.sleep(1.0)
    writer.write(json.dumps({"cmd":"list_simulations", "limit":5}).encode() + b"\n")
    await writer.drain()
    line = await reader.readline()
    print("recent:", line.decode().strip())
    writer.close()
    await writer.wait_closed()

if __name__ == "__main__":
    use_sur = "--surrogate" in sys.argv
    asyncio.run(run(use_surrogate=use_sur))


