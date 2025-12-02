#!/usr/bin/env python3
"""
Minimal interactive JSON-lines client for the time_engine server.

Usage:
  python client.py --host 127.0.0.1 --port 8765
 Then type JSON commands per line, for example:
  {"cmd":"simulate","seed":{"ETH":2000},"horizon":12,"ensemble":32}
"""
import argparse
import asyncio
import json


async def run_client(host: str, port: int):
    reader, writer = await asyncio.open_connection(host, port)
    print(f"Connected to {host}:{port}. Type JSON commands, or 'quit' to exit.")
    loop = asyncio.get_event_loop()
    api_key = None

    async def reader_loop():
        try:
            while True:
                resp = await reader.readline()
                if not resp:
                    print("Server closed connection.")
                    break
                s = resp.decode().strip()
                # try to pretty parse JSON
                try:
                    j = json.loads(s)
                    # print concise preview for large payloads
                    # if this is a PF start response, print a clear message
                    if isinstance(j, dict) and ("start_pf" in j or "pf_job_id" in j):
                        pf_info = j.get("start_pf") or {}
                        pf_id = j.get("pf_job_id") or pf_info.get("job_id")
                        print(f"PF started: job_id={pf_id}")
                    else:
                        preview = json.dumps(j if isinstance(j, dict) and len(str(j)) < 400 else {"preview": "large_json", "keys": list(j.keys()) if isinstance(j, dict) else None})
                        print("INCOMING:", preview)
                except Exception:
                    print("INCOMING:", s[:400] + ("..." if len(s) > 400 else ""))
        except Exception:
            pass

    # spawn background reader
    asyncio.create_task(reader_loop())

    try:
        while True:
            line = await loop.run_in_executor(None, input, "> ")
            if not line:
                continue
            if line.strip().lower() in ("quit", "exit"):
                break
            # allow raw JSON or shortcuts
            line_to_send = line
            if not line.strip().startswith("{"):
                # try to parse simple simulate shortcut: simulate 24
                parts = line.strip().split()
                if parts and parts[0].lower() == "simulate":
                    try:
                        horizon = int(parts[1]) if len(parts) > 1 else 24
                    except Exception:
                        horizon = 24
                    payload = {"cmd": "simulate", "seed": {}, "horizon": horizon, "ensemble": 32}
                    line_to_send = json.dumps(payload)
            # admin shortcuts
            if line.strip().lower().startswith("list_simulations"):
                payload = {"cmd": "list_simulations", "limit": 20}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("list_checkpoints"):
                payload = {"cmd": "list_checkpoints", "limit": 50}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("reconstruct_test"):
                payload = {"cmd": "reconstruct_test", "horizon": 10, "ensemble": 8}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("calibrate"):
                payload = {"cmd": "calibrate_merge", "horizon": 12, "ensemble": 32, "percentile": 10.0}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("pf"):
                # particle filter shortcut
                payload = {"cmd": "particle_filter", "seed": {}, "observations": {}, "num_particles": 128, "horizon": 24}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("reps"):
                # get representatives for a simulation id: reps 5
                parts = line.strip().split()
                sid = int(parts[1]) if len(parts) > 1 else 1
                payload = {"cmd": "get_representatives", "sim_id": sid}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("particles"):
                parts = line.strip().split()
                sid = int(parts[1]) if len(parts) > 1 else 1
                payload = {"cmd": "get_particles", "sim_id": sid}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("reglaw"):
                # reglaw name
                parts = line.strip().split(maxsplit=1)
                name = parts[1] if len(parts) > 1 else "cosmo_law"
                phidesc = {"type": "cosmology_lcdm", "params": {"H0": 2.2e-18, "Omega_m": 0.315, "Omega_r": 9e-5, "Omega_L": 0.685}}
                payload = {"cmd": "register_law", "name": name, "phidesc": phidesc, "description": "LCDM law"}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("listlaws"):
                payload = {"cmd": "list_laws"}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("getlaw"):
                parts = line.strip().split()
                lid = int(parts[1]) if len(parts) > 1 else 1
                payload = {"cmd": "get_law", "law_id": lid}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("startpf"):
                # startpf num_particles
                parts = line.strip().split()
                nump = int(parts[1]) if len(parts) > 1 else 128
                payload = {"cmd": "start_pf", "num_particles": nump, "horizon": 100}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("startmultiscale"):
                # startmultiscale optional_json_phidesc
                parts = line.strip().split(maxsplit=1)
                ph = {}
                if len(parts) > 1:
                    try:
                        ph = json.loads(parts[1])
                    except Exception:
                        ph = {}
                payload = {"cmd": "start_multiscale_sim", "phidesc": ph, "horizon": 10, "ensemble": 1}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("list_tiles"):
                parts = line.strip().split()
                jid = int(parts[1]) if len(parts) > 1 else 1
                payload = {"cmd": "list_tiles", "job_id": jid}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("subscribe_tiles"):
                parts = line.strip().split()
                jid = int(parts[1]) if len(parts) > 1 else 1
                payload = {"cmd": "subscribe_tiles", "job_id": jid}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("unsubscribe_tiles"):
                parts = line.strip().split()
                jid = int(parts[1]) if len(parts) > 1 else 1
                payload = {"cmd": "unsubscribe_tiles", "job_id": jid}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("list_branches"):
                parts = line.strip().split()
                if len(parts) > 1:
                    # accept job:<id> or sim:<id>
                    arg = parts[1]
                    if arg.startswith("job:"):
                        jid = int(arg.split(":",1)[1])
                        payload = {"cmd": "list_branches", "job_id": jid}
                    elif arg.startswith("sim:"):
                        sid = int(arg.split(":",1)[1])
                        payload = {"cmd": "list_branches", "sim_id": sid}
                    else:
                        jid = int(arg)
                        payload = {"cmd": "list_branches", "job_id": jid}
                else:
                    payload = {"cmd": "list_branches", "job_id": 1}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("cancel_branch"):
                parts = line.strip().split()
                if len(parts) > 1:
                    arg = parts[1]
                    if arg.startswith("branch:"):
                        bid = int(arg.split(":",1)[1])
                        payload = {"cmd": "cancel_branch", "branch_id": bid}
                    elif arg.startswith("child:"):
                        cid = int(arg.split(":",1)[1])
                        payload = {"cmd": "cancel_branch", "child_job_id": cid}
                    else:
                        bid = int(arg)
                        payload = {"cmd": "cancel_branch", "branch_id": bid}
                else:
                    payload = {"cmd": "cancel_branch"}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("pfobs"):
                parts = line.strip().split(maxsplit=2)
                job = int(parts[1]) if len(parts) > 1 else 1
                # observation as JSON string
                try:
                    obs = json.loads(parts[2]) if len(parts) > 2 else {}
                except Exception:
                    obs = {}
                payload = {"cmd": "pf_observe", "job_id": job, "observation": obs}
                line_to_send = json.dumps(payload)
            if line.strip().lower().startswith("stoppf"):
                parts = line.strip().split()
                job = int(parts[1]) if len(parts) > 1 else 1
                payload = {"cmd": "stop_pf", "job_id": job}
                line_to_send = json.dumps(payload)
            # send command and continue; background reader will print incoming messages
            try:
                # inject api_key for mutation commands if provided via arg
                try:
                    payload_obj = json.loads(line_to_send)
                    if isinstance(payload_obj, dict) and payload_obj.get("cmd", "").lower() in ("simulate", "register_law", "start_pf", "pf_observe", "stop_pf", "snapshot", "particle_filter"):
                        if api_key:
                            payload_obj.setdefault("api_key", api_key)
                        line_to_send = json.dumps(payload_obj)
                except Exception:
                    pass
                writer.write(line_to_send.encode() + b"\n")
                await writer.drain()
            except Exception:
                break
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    try:
        # pass api_key into client coroutine via closure
        async def _run():
            nonlocal args
            await run_client(args.host, args.port)
        # set module-level api_key for use in run_client
        global_api_key = args.api_key
        # monkeypatch local variable inside coroutine by setting attribute on function (simple approach)
        run_client.__globals__['api_key'] = args.api_key
        asyncio.run(run_client(args.host, args.port))
    except KeyboardInterrupt:
        print("Exiting client.")


if __name__ == "__main__":
    main()


