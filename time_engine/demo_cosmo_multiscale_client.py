#!/usr/bin/env python3
"""
Demo that uses the running main.py server API to start a multiscale simulation,
polls job status, and fetches tile results.
"""
import asyncio
import json
import time

async def run_demo(host="127.0.0.1", port=8765):
    reader, writer = await asyncio.open_connection(host, port)
    # build phidesc
    ph = {
        "type": "cosmology_lcdm",
        "physics_modes": {"gravity_model": "FLRW"},
        "tiles": {"n_tiles": 4},
        "seed_config": {"a_0": 1.0, "phi_0": 1.0, "noise_seed": 42},
        "params": {"H0": 2.2e-18}
    }
    payload = {"cmd": "start_multiscale_sim", "phidesc": ph, "horizon": 5, "ensemble": 1}
    writer.write(json.dumps(payload).encode() + b"\n")
    await writer.drain()
    job_id = None
    # read response
    line = await reader.readline()
    if line:
        resp = json.loads(line.decode())
        job_id = resp.get("multiscale", {}).get("job_id") or resp.get("multiscale", {}).get("job_id")
        print("started multiscale job:", job_id)
    # poll status until done
    while True:
        await asyncio.sleep(1.0)
        writer.write(json.dumps({"cmd":"status"}).encode() + b"\n")
        await writer.drain()
        line = await reader.readline()
        if not line:
            break
        s = json.loads(line.decode())
        jobs = s.get("jobs", {})
        j = jobs.get(str(job_id)) if isinstance(jobs, dict) else jobs.get(job_id)
        # if job set done, break
        # simplify: request get_simulation once
        if j and (j.get("status") == "done" or j.get("status") == "error"):
            # fetch sim id
            # use list_simulations to find latest
            writer.write(json.dumps({"cmd":"list_simulations", "limit":5}).encode() + b"\n")
            await writer.drain()
            line2 = await reader.readline()
            if not line2:
                break
            sims = json.loads(line2.decode()).get("simulations", [])
            print("recent sims:", sims)
            break
    writer.close()
    await writer.wait_closed()

if __name__ == "__main__":
    asyncio.run(run_demo())


