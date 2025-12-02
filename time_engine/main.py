#!/usr/bin/env python3
"""
Async JSON-lines TCP server for the Simulacrum Engine.

Commands (JSON per line):
 - {"cmd":"simulate","seed":{"ETH":2000},"horizon":24,"ensemble":64,"rng_seed":123}
 - {"cmd":"snapshot","name":"midday"}
 - {"cmd":"query_graph"}
 - {"cmd":"status"}
 - {"cmd":"shutdown"}
"""
import argparse
import asyncio
import json
import logging
from typing import Dict, Any, List
import os
import time
import concurrent.futures
import resource
try:
    import psutil
except Exception:
    psutil = None
import threading

from engine import SimulacrumEngine

LOG = logging.getLogger("time_engine.server")
logging.basicConfig(level=logging.INFO)


class JobManager:
    def __init__(self, engine: SimulacrumEngine):
        self.engine = engine
        self._jobs: Dict[int, Dict[str, Any]] = {}
        self._next_job_id = 1
        # executor for heavy jobs (use processes for CPU-bound sims)
        maxw = os.cpu_count() or 2
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=maxw)
        # particle-filter streaming jobs: job_id -> state dict
        self._pf_jobs: Dict[int, Dict[str, Any]] = {}
        # tile streaming subscribers: job_id -> list of async-callable callbacks
        self._tile_subscribers: Dict[int, List[Any]] = {}
        # subscribers for streaming PF updates: job_id -> list of async-callable callbacks
        self._pf_subscribers: Dict[int, List[Any]] = {}
        # throttling/windowing state
        self._pf_last_ts: Dict[int, float] = {}
        self._pf_obs_buffers: Dict[int, List[Dict[str, float]]] = {}
        self._pf_scheduled: Dict[int, bool] = {}
        # metrics counters
        self._metrics = {"total_jobs": 0, "completed_jobs": 0, "pf_jobs": 0, "observations_processed": 0, "observations_dropped": 0}
        self._metrics_lock = threading.Lock()
        # PF buffering cap
        try:
            self._pf_buffer_cap = int(os.environ.get("PF_BUFFER_CAP", "1000"))
        except Exception:
            self._pf_buffer_cap = 1000
        # store per-job runtime/cpu info
        self._job_cpu_before: Dict[int, float] = {}
        # thread executor for multiscale sims (allows in-process callbacks)
        self._thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        # global branch budget
        try:
            self._global_branch_budget = int(os.environ.get("GLOBAL_BRANCH_BUDGET", "100"))
        except Exception:
            self._global_branch_budget = 100
        self._global_branch_count = 0

    async def submit_simulation(self, params: Dict[str, Any], progress_coro=None) -> Dict[str, Any]:
        job_id = self._next_job_id
        self._next_job_id += 1
        self._jobs[job_id] = {"status": "queued", "params": params}

        loop = asyncio.get_running_loop()
        self._jobs[job_id]["status"] = "running"
        # record start time in parent for wall-clock timing
        start_ts = time.time()
        self._jobs[job_id]["start_ts"] = start_ts
        # record child CPU baseline (cumulative children)
        try:
            child_cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
        except Exception:
            child_cpu_before = 0.0
        self._job_cpu_before[job_id] = child_cpu_before

        def run():
            try:
                with self._metrics_lock:
                    self._metrics["total_jobs"] += 1
                # provide a thread-safe progress callback that schedules the coroutine on the loop
                if progress_coro is None:
                    def progress_coro(payload: Dict[str, Any]):
                        async def _noop():
                            return None
                        return _noop()

                def progress_cb(payload: Dict[str, Any]):
                    try:
                        loop.call_soon_threadsafe(asyncio.create_task, progress_coro(payload))
                    except Exception:
                        pass

                out = self.engine.run_simulation(
                    seed_state=params.get("seed", {}),
                    horizon=int(params.get("horizon", 24)),
                    branching=int(params.get("branching", 4)),
                    ensemble=int(params.get("ensemble", 64)),
                    name=params.get("name", f"job_{job_id}"),
                    rng_seed=params.get("rng_seed", None),
                    method=params.get("method", "agent"),
                    merge_threshold=float(params.get("merge_threshold", 1.0)),
                    # progress callback invoked inside run_simulation
                    # we pass progress_cb via phidesc to avoid changing signature further
                    phidesc={**(params.get("phidesc", {}) or {}), "_progress_cb": progress_cb},
                )
                self._jobs[job_id]["status"] = "done"
                self._jobs[job_id]["result"] = out
                with self._metrics_lock:
                    self._metrics["completed_jobs"] += 1
            except Exception as exc:
                self._jobs[job_id]["status"] = "error"
                self._jobs[job_id]["error"] = str(exc)

        # run in executor and then record end time / cpu delta
        await loop.run_in_executor(self._executor, run)
        end_ts = time.time()
        self._jobs[job_id]["end_ts"] = end_ts
        self._jobs[job_id]["duration"] = end_ts - start_ts
        try:
            child_cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
        except Exception:
            child_cpu_after = 0.0
        cpu_delta = child_cpu_after - self._job_cpu_before.get(job_id, 0.0)
        self._jobs[job_id]["cpu_child_seconds"] = float(cpu_delta)
        # update metrics last job duration
        self._metrics["last_job_duration"] = float(self._jobs[job_id]["duration"])
        return {"job_id": job_id, "status": self._jobs[job_id]["status"], "result": self._jobs[job_id].get("result")}

    def status(self) -> Dict[int, Dict[str, Any]]:
        return self._jobs

    def start_pf_job(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start an online particle-filter job and return job_id."""
        job_id = self._next_job_id
        self._next_job_id += 1
        self._jobs[job_id] = {"status": "pf_running", "params": params}
        # initialize PF state
        seed = params.get("seed", self.engine.get_default_universe_config()["initial"])
        num_particles = int(params.get("num_particles", 128))
        horizon = int(params.get("horizon", 100))
        phidesc = params.get("phidesc", None)
        boundary_model = params.get("boundary_model", None)
        pf_state = self.engine.particle_filter_init(seed, phidesc=phidesc, boundary_model=boundary_model, num_particles=num_particles, horizon=horizon, rng_seed=params.get("rng_seed", None))
        self._pf_jobs[job_id] = {"state": pf_state, "phidesc": phidesc, "obs_var": float(params.get("obs_var", 1.0))}
        self._pf_subscribers[job_id] = []
        with self._metrics_lock:
            self._metrics["pf_jobs"] += 1
        return {"job_id": job_id, "status": "pf_started"}

    def start_multiscale_sim(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start a multiscale simulation (ASTT) in the executor and return job id."""
        job_id = self._next_job_id
        self._next_job_id += 1
        self._jobs[job_id] = {"status": "queued", "params": params}

        loop = asyncio.get_running_loop()

        def run_sim():
            try:
                with self._metrics_lock:
                    self._metrics["total_jobs"] += 1
                # use engine.run_simulation with provided phidesc
                out = self.engine.run_simulation(
                    seed_state=params.get("seed", {}),
                    horizon=int(params.get("horizon", 24)),
                    branching=int(params.get("branching", 4)),
                    ensemble=int(params.get("ensemble", 64)),
                    name=params.get("name", f"multiscale_{job_id}"),
                    rng_seed=params.get("rng_seed", None),
                    method=params.get("method", "rk45"),
                    max_materialize=int(params.get("max_materialize", 128)),
                    merge_threshold=float(params.get("merge_threshold", 1.0)),
                    phidesc=params.get("phidesc", {}),
                )
                self._jobs[job_id]["status"] = "done"
                self._jobs[job_id]["result"] = out
                self._jobs[job_id]["sim_id"] = out.get("sim_id")
                with self._metrics_lock:
                    self._metrics["completed_jobs"] += 1
            except Exception as exc:
                self._jobs[job_id]["status"] = "error"
                self._jobs[job_id]["error"] = str(exc)

        # run in thread executor so progress_cb can schedule loop callbacks
        def run_thread():
            # inject job_id into phidesc for engine progress payloads
            if isinstance(params.get("phidesc", None), dict):
                params["phidesc"]["_job_id"] = job_id
            else:
                params["phidesc"] = {"_job_id": job_id}

            def progress_cb(payload: Dict[str, Any]):
                try:
                    # handle auto-branch requests emitted by engine
                    if isinstance(payload, dict) and payload.get("action") == "auto_branch_request":
                        try:
                            branch_info = payload.get("branch", {})
                            parent_entry = self._jobs.get(job_id, {"params": {}})
                            parent_depth = int(parent_entry.get("branch_depth", 0))
                            parent_branch_count = int(parent_entry.get("branch_count", 0))
                            branch_budget = int(parent_entry.get("params", {}).get("branch_budget", int(os.environ.get("BRANCH_BUDGET", "10"))))
                            max_depth = int(parent_entry.get("params", {}).get("max_branch_depth", int(os.environ.get("MAX_BRANCH_DEPTH", "3"))))
                            if parent_depth >= max_depth:
                                # notify rejection due to depth
                                notify = {"action": "auto_branch_rejected", "reason": "max_depth_exceeded", "parent_job": job_id, "tile_id": payload.get("tile_id"), "t": payload.get("t")}
                                for cbn in list(self._tile_subscribers.get(job_id, [])):
                                    try:
                                        coro = cbn(notify)
                                        loop.call_soon_threadsafe(asyncio.create_task, coro)
                                    except Exception:
                                        pass
                            elif parent_branch_count >= branch_budget:
                                notify = {"action": "auto_branch_rejected", "reason": "budget_exhausted", "parent_job": job_id, "tile_id": payload.get("tile_id"), "t": payload.get("t")}
                                for cbn in list(self._tile_subscribers.get(job_id, [])):
                                    try:
                                        coro = cbn(notify)
                                        loop.call_soon_threadsafe(asyncio.create_task, coro)
                                    except Exception:
                                        pass
                            else:
                                # check global budget
                                if self._global_branch_count >= self._global_branch_budget:
                                    notify = {"action": "auto_branch_rejected", "reason": "global_budget_exhausted", "parent_job": job_id, "tile_id": payload.get("tile_id"), "t": payload.get("t")}
                                    for cbn in list(self._tile_subscribers.get(job_id, [])):
                                        try:
                                            coro = cbn(notify)
                                            loop.call_soon_threadsafe(asyncio.create_task, coro)
                                        except Exception:
                                            pass
                                    # do not schedule branch due to global budget
                                    pass
                                # schedule new multiscale job for branch
                                new_job = self.start_multiscale_sim(branch_info)
                                new_job_id = new_job.get("job_id")
                                # update bookkeeping
                                parent_entry["branch_count"] = parent_branch_count + 1
                                self._jobs[job_id] = parent_entry
                                if new_job_id is not None:
                                    # annotate child job metadata
                                    self._jobs[new_job_id]["parent_job"] = job_id
                                    self._jobs[new_job_id]["branch_depth"] = parent_depth + 1
                                    # increment global branch counter
                                    try:
                                        self._global_branch_count += 1
                                    except Exception:
                                        pass
                                # persist branch metadata via engine
                                try:
                                    self.engine.register_branch(parent_job_id=job_id, child_job_id=new_job_id, parent_sim_id=parent_entry.get("sim_id"), child_sim_id=None, tile_id=payload.get("tile_id"), t_val=payload.get("t"), payload=branch_info, depth=parent_depth + 1, status="scheduled")
                                except Exception:
                                    pass
                                # notify subscribers about scheduled branch
                                notify = {"action": "auto_branch_scheduled", "parent_job": job_id, "new_job": new_job, "branch_id": payload.get("branch_id"), "tile_id": payload.get("tile_id"), "t": payload.get("t")}
                                subs_notify = list(self._tile_subscribers.get(job_id, []))
                                for cbn in subs_notify:
                                    try:
                                        coro = cbn(notify)
                                        loop.call_soon_threadsafe(asyncio.create_task, coro)
                                    except Exception:
                                        pass
                        except Exception:
                            LOG.exception("failed to schedule auto-branch")
                    # publish to tile subscribers for this job
                    subs = list(self._tile_subscribers.get(job_id, []))
                    for cb in subs:
                        try:
                            coro = cb(payload)
                            loop.call_soon_threadsafe(asyncio.create_task, coro)
                        except Exception:
                            continue
                except Exception:
                    pass

            try:
                with self._metrics_lock:
                    self._metrics["total_jobs"] += 1
                # call engine multiscale with progress callback
                out = self.engine.run_multiscale_simulation(seed_state=params.get("seed", {}), horizon=int(params.get("horizon", 24)), phidesc=params.get("phidesc", {}), ensemble=int(params.get("ensemble", 1)), rng_seed=params.get("rng_seed", None), dt=float((params.get("phidesc", {}) or {}).get("evolution_solver", {}).get("dt", 1.0)), progress_cb=progress_cb)
                self._jobs[job_id]["status"] = "done"
                self._jobs[job_id]["result"] = out
                self._jobs[job_id]["sim_id"] = out.get("sim_id")
                # if this job was created as a branch, update branch record with sim_id
                try:
                    parent_job = self._jobs[job_id].get("parent_job")
                    if parent_job is not None:
                        # update any branch rows referencing this child job id
                        self.engine.update_branch_with_child_sim(child_job_id=job_id, child_sim_id=out.get("sim_id"))
                except Exception:
                    pass
                # persist parent_job link into simulations table if possible
                try:
                    sim_id_new = out.get("sim_id")
                    parent_job = self._jobs[job_id].get("parent_job")
                    if sim_id_new and parent_job is not None:
                        cur = self.engine._conn.cursor()
                        cur.execute("UPDATE simulations SET parent_job_id = ? WHERE id = ?", (int(parent_job), int(sim_id_new)))
                        self.engine._conn.commit()
                except Exception:
                    pass
                with self._metrics_lock:
                    self._metrics["completed_jobs"] += 1
            except Exception as exc:
                self._jobs[job_id]["status"] = "error"
                self._jobs[job_id]["error"] = str(exc)

        loop.run_in_executor(self._thread_executor, run_thread)
        return {"job_id": job_id, "status": "multiscale_queued"}

    def pf_observe(self, job_id: int, observation: Dict[str, float]) -> Dict[str, Any]:
        """Feed an observation to a running PF job and return updated summary."""
        if job_id not in self._pf_jobs:
            return {"error": "job not found"}
        pf_entry = self._pf_jobs[job_id]
        pf_state = pf_entry["state"]
        phidesc = pf_entry.get("phidesc")
        obs_var = pf_entry.get("obs_var", 1.0)
        # throttling/windowing: if observations arrive faster than interval, buffer them and schedule aggregated processing
        loop = asyncio.get_running_loop()
        throttle_interval = float(pf_entry.get("throttle_interval", 0.5))  # seconds
        now = time.time()
        last = self._pf_last_ts.get(job_id, 0.0)
        # if within throttle interval, buffer observation
        if now - last < throttle_interval:
            buf = self._pf_obs_buffers.setdefault(job_id, [])
            # enforce buffer cap: drop oldest if necessary
            if len(buf) >= self._pf_buffer_cap:
                try:
                    buf.pop(0)
                    with self._metrics_lock:
                        self._metrics["observations_dropped"] += 1
                except Exception:
                    pass
            buf.append(observation)
            # schedule drain if not scheduled
            if not self._pf_scheduled.get(job_id, False):
                self._pf_scheduled[job_id] = True
                delay = max(0.0, throttle_interval - (now - last))
                # schedule async drain
                loop.call_later(delay, lambda: asyncio.create_task(self._async_drain_pf(job_id)))
            return {"job_id": job_id, "status": "buffered"}
        # else process immediately
        summary = self.engine.particle_filter_step(pf_state, observation, phidesc=phidesc, obs_var=obs_var)
        with self._metrics_lock:
            self._metrics["observations_processed"] += 1
        self._pf_last_ts[job_id] = now
        # publish to subscribers (schedule async callbacks)
        try:
            subs = list(self._pf_subscribers.get(job_id, []))
            for cb in subs:
                try:
                    coro = cb({"job_id": job_id, "summary": summary})
                    loop.call_soon_threadsafe(asyncio.create_task, coro)
                except Exception:
                    continue
        except Exception:
            pass
        # save back
        pf_entry["state"] = pf_state
        return {"job_id": job_id, "summary": summary}

    async def _async_drain_pf(self, job_id: int) -> None:
        """Async handler to drain buffered observations for job_id, aggregate them, and process."""
        try:
            if job_id not in self._pf_jobs:
                return
            pf_entry = self._pf_jobs[job_id]
            buf = self._pf_obs_buffers.get(job_id, [])
            if not buf:
                self._pf_scheduled[job_id] = False
                return
            # aggregate buffered observations by averaging each observed key
            agg = {}
            counts = {}
            for obs in buf:
                for k, v in obs.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
            for k in list(agg.keys()):
                agg[k] = agg[k] / max(1, counts.get(k, 1))
            # clear buffer
            self._pf_obs_buffers[job_id] = []
            self._pf_scheduled[job_id] = False
            # run PF step in executor to avoid blocking
            loop = asyncio.get_running_loop()
            def step_call():
                return self.pf_observe(job_id, agg)
            # call synchronously via run_in_executor to avoid recursion; pf_observe will bypass buffering path now
            res = await loop.run_in_executor(self._executor, step_call)
            return res
        except Exception:
            self._pf_scheduled[job_id] = False
            return None

    def stop_pf_job(self, job_id: int) -> Dict[str, Any]:
        """Stop PF job and persist its final posterior as a simulation record."""
        if job_id not in self._pf_jobs:
            return {"error": "job not found"}
        pf_entry = self._pf_jobs.pop(job_id)
        pf_state = pf_entry["state"]
        # build posterior summary
        posterior = {"particles": pf_state.get("particles"), "weights": pf_state.get("weights"), "t": pf_state.get("t")}
        try:
            sim_id = self.engine._store_simulation_result(name=f"pf_job_{job_id}", x0=pf_state.get("particles")[0] if pf_state.get("particles") else {}, phidesc=pf_entry.get("phidesc", {}), config={"job_id": job_id}, result={"particle_filter_result": posterior})
        except Exception:
            sim_id = None
        return {"job_id": job_id, "stopped": True, "sim_id": sim_id}


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, mgr: JobManager):
    addr = writer.get_extra_info("peername")
    LOG.info("Client connected: %s", addr)
    try:
        # helper to create progress coroutine for thread callbacks
        async def _send_progress(payload: Dict[str, Any]):
            try:
                writer.write(json.dumps(payload).encode() + b"\n")
                await writer.drain()
            except Exception:
                pass
        # track pf subscriptions by this client (job_id -> callback)
        pf_subs = []
        # per-connection rate-limiting: simple sliding window
        RATE_LIMIT = float(os.environ.get("RATE_LIMIT", 20))  # requests
        RATE_WINDOW = float(os.environ.get("RATE_WINDOW", 1.0))  # seconds
        req_timestamps = []

        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode("utf-8").strip())
            except Exception as e:
                await _send_progress({"error": "invalid json", "detail": str(e)})
                continue

            # rate limit check
            now = time.time()
            # remove old timestamps
            req_timestamps = [ts for ts in req_timestamps if now - ts <= RATE_WINDOW]
            if len(req_timestamps) >= RATE_LIMIT:
                await _send_progress({"error": "rate_limited", "limit": RATE_LIMIT, "window": RATE_WINDOW})
                continue
            req_timestamps.append(now)

            # simple API key check for mutation endpoints
            API_KEY = os.environ.get("API_KEY", "")
            MUTATION_CMDS = {"simulate", "register_law", "start_pf", "pf_observe", "stop_pf", "snapshot", "particle_filter"}
            if msg.get("cmd", "").lower() in MUTATION_CMDS and API_KEY:
                provided = msg.get("api_key", "")
                if provided != API_KEY:
                    await _send_progress({"error": "invalid_api_key"})
                    continue

            cmd = msg.get("cmd", "").lower()
            if cmd == "simulate":
                # submit simulation and stream progress updates via progress_coro
                def progress_coro_creator(payload: Dict[str, Any]):
                    return _send_progress(payload)

                # send accepted immediately
                await _send_progress({"status": "accepted", "job_id": mgr._next_job_id})
                out = await mgr.submit_simulation(msg, progress_coro=progress_coro_creator)
                await _send_progress({"status": "finished", "job": out})
            elif cmd == "snapshot":
                name = msg.get("name", "manual_snapshot")
                sid = mgr.engine.snapshot(name)
                writer.write(json.dumps({"status": "snapshot", "sim_id": sid}).encode() + b"\n")
                await writer.drain()
            elif cmd == "reconstruct_test":
                out = mgr.engine.run_reconstructability_test(horizon=int(msg.get("horizon", 10)), ensemble=int(msg.get("ensemble", 16)))
                writer.write(json.dumps({"status": "reconstruct_test", "result": out}).encode() + b"\n")
                await writer.drain()
            elif cmd == "gc":
                out = mgr.engine.gc_sweep(retain_days=int(msg.get("retain_days", 30)))
                writer.write(json.dumps({"status": "gc", "result": out}).encode() + b"\n")
                await writer.drain()
            elif cmd == "list_simulations":
                sims = mgr.engine.list_simulations(limit=int(msg.get("limit", 50)))
                writer.write(json.dumps({"status": "ok", "simulations": sims}).encode() + b"\n")
                await writer.drain()
            elif cmd == "get_simulation":
                sid = int(msg.get("sim_id"))
                sim = mgr.engine.get_simulation(sid)
                writer.write(json.dumps({"status": "ok", "simulation": sim}).encode() + b"\n")
                await writer.drain()
            elif cmd == "list_checkpoints":
                sid = msg.get("sim_id")
                cps = mgr.engine.list_checkpoints(sim_id=int(sid) if sid is not None else None, limit=int(msg.get("limit", 100)))
                writer.write(json.dumps({"status": "ok", "checkpoints": cps}).encode() + b"\n")
                await writer.drain()
            elif cmd == "get_checkpoint":
                cid = int(msg.get("checkpoint_id"))
                cp = mgr.engine.get_checkpoint(cid)
                writer.write(json.dumps({"status": "ok", "checkpoint": cp}).encode() + b"\n")
                await writer.drain()
            elif cmd == "list_blobs":
                blobs = mgr.engine.list_blobs(limit=int(msg.get("limit", 100)))
                writer.write(json.dumps({"status": "ok", "blobs": blobs}).encode() + b"\n")
                await writer.drain()
            elif cmd == "list_pf_jobs":
                # return list of active pf jobs with simple metadata
                pf_list = []
                for jid, entry in mgr._pf_jobs.items():
                    pf_list.append({"job_id": jid, "status": mgr._jobs.get(jid, {}).get("status", "pf_running"), "subscribers": len(mgr._pf_subscribers.get(jid, []))})
                writer.write(json.dumps({"status": "ok", "pf_jobs": pf_list}).encode() + b"\n")
                await writer.drain()
            elif cmd == "start_multiscale_sim":
                out = mgr.start_multiscale_sim(msg)
                writer.write(json.dumps({"status": "ok", "multiscale": out}).encode() + b"\n")
                await writer.drain()
            elif cmd == "list_tiles":
                # list tile diagnostics for a multiscale sim (if present in stored sim result)
                job_id = int(msg.get("job_id"))
                job = mgr._jobs.get(job_id, {})
                sim_id = job.get("sim_id") or job.get("result", {}).get("sim_id")
                if not sim_id:
                    writer.write(json.dumps({"status": "error", "error": "sim_id not available yet"}).encode() + b"\n")
                    await writer.drain()
                else:
                    sim = mgr.engine.get_simulation(int(sim_id))
                    tiles = sim.get("result", {}).get("tiles") if isinstance(sim, dict) else None
                    writer.write(json.dumps({"status": "ok", "tiles": tiles}).encode() + b"\n")
                    await writer.drain()
            elif cmd == "get_representatives":
                sid = int(msg.get("sim_id"))
                reps = mgr.engine.get_representatives(sid)
                writer.write(json.dumps({"status": "ok", "representatives": reps}).encode() + b"\n")
                await writer.drain()
            elif cmd == "get_particles":
                sid = int(msg.get("sim_id"))
                pf = mgr.engine.get_particle_results(sid)
                writer.write(json.dumps({"status": "ok", "particle_results": pf}).encode() + b"\n")
                await writer.drain()
            elif cmd == "register_law":
                name = msg.get("name", "law")
                ph = msg.get("phidesc", {})
                desc = msg.get("description", "")
                try:
                    # use extended registration if available
                    lid = mgr.engine.register_law_extended(name, ph, description=desc) if hasattr(mgr.engine, "register_law_extended") else mgr.engine.register_law(name, ph, description=desc)
                    writer.write(json.dumps({"status": "ok", "law_id": lid}).encode() + b"\n")
                except Exception as exc:
                    writer.write(json.dumps({"status": "error", "error": str(exc)}).encode() + b"\n")
                await writer.drain()
            elif cmd == "list_laws":
                laws = mgr.engine.list_laws()
                writer.write(json.dumps({"status": "ok", "laws": laws}).encode() + b"\n")
                await writer.drain()
            elif cmd == "get_law":
                lid = int(msg.get("law_id"))
                law = mgr.engine.get_law(lid)
                writer.write(json.dumps({"status": "ok", "law": law}).encode() + b"\n")
                await writer.drain()
            elif cmd == "calibrate_merge":
                phidesc = msg.get("phidesc", None)
                out = mgr.engine.calibrate_merge_threshold(phidesc=phidesc, horizon=int(msg.get("horizon", 12)), ensemble=int(msg.get("ensemble", 32)), percentile=float(msg.get("percentile", 10.0)))
                writer.write(json.dumps({"status": "ok", "calibration": out}).encode() + b"\n")
                await writer.drain()
            elif cmd == "particle_filter":
                # expects: seed, observations (dict t->{var:val}), phidesc, boundary_model, num_particles
                seed = msg.get("seed", mgr.engine.get_default_universe_config()["initial"])
                observations = msg.get("observations", {})
                phidesc = msg.get("phidesc", None)
                boundary_model = msg.get("boundary_model", None)
                num_particles = int(msg.get("num_particles", 128))
                horizon = int(msg.get("horizon", 24))
                # run PF in executor to avoid blocking
                loop = asyncio.get_running_loop()
                def pf_run():
                    return mgr.engine.particle_filter(initial_state=seed, observations=observations, phidesc=phidesc, boundary_model=boundary_model, num_particles=num_particles, horizon=horizon, rng_seed=msg.get("rng_seed", None))
                out = await loop.run_in_executor(None, pf_run)
                # persist particle filter result as a simulation record for visualization
                try:
                    name = msg.get("name", "particle_filter")
                    x0 = seed
                    phidesc_save = phidesc or {}
                    config = {"num_particles": num_particles, "horizon": horizon}
                    sim_id_pf = mgr.engine._store_simulation_result(name, x0, phidesc_save, config, {"particle_filter_result": out})
                except Exception:
                    sim_id_pf = None
                writer.write(json.dumps({"status": "ok", "particle_filter_result": out, "sim_id": sim_id_pf}).encode() + b"\n")
                await writer.drain()
            elif cmd == "start_pf":
                out = mgr.start_pf_job(msg)
                pf_job_id = out.get("job_id")
                writer.write(json.dumps({"status": "ok", "start_pf": out, "pf_job_id": pf_job_id}).encode() + b"\n")
                await writer.drain()
            elif cmd == "pf_observe":
                job_id = int(msg.get("job_id"))
                obs = msg.get("observation", {})
                out = mgr.pf_observe(job_id, obs)
                writer.write(json.dumps({"status": "ok", "pf_observe": out}).encode() + b"\n")
                await writer.drain()
            elif cmd == "stop_pf":
                job_id = int(msg.get("job_id"))
                out = mgr.stop_pf_job(job_id)
                writer.write(json.dumps({"status": "ok", "stop_pf": out}).encode() + b"\n")
                await writer.drain()
            elif cmd == "subscribe_pf":
                job_id = int(msg.get("job_id"))
                # register this client's _send_progress as subscriber
                mgr._pf_subscribers.setdefault(job_id, []).append(_send_progress)
                pf_subs.append((job_id, _send_progress))
                writer.write(json.dumps({"status": "ok", "subscribed": job_id}).encode() + b"\n")
                await writer.drain()
            elif cmd == "unsubscribe_pf":
                job_id = int(msg.get("job_id"))
                try:
                    mgr._pf_subscribers.get(job_id, []).remove(_send_progress)
                except Exception:
                    pass
                writer.write(json.dumps({"status": "ok", "unsubscribed": job_id}).encode() + b"\n")
                await writer.drain()
            elif cmd == "subscribe_tiles":
                job_id = int(msg.get("job_id"))
                mgr._tile_subscribers.setdefault(job_id, []).append(_send_progress)
                pf_subs.append((("tile", job_id), _send_progress))
                writer.write(json.dumps({"status": "ok", "subscribed_tiles": job_id}).encode() + b"\n")
                await writer.drain()
            elif cmd == "unsubscribe_tiles":
                job_id = int(msg.get("job_id"))
                try:
                    mgr._tile_subscribers.get(job_id, []).remove(_send_progress)
                except Exception:
                    pass
                writer.write(json.dumps({"status": "ok", "unsubscribed_tiles": job_id}).encode() + b"\n")
                await writer.drain()
            elif cmd == "query_graph":
                g = mgr.engine.query_graph()
                writer.write(json.dumps({"graph": g}).encode() + b"\n")
                await writer.drain()
            elif cmd == "status":
                writer.write(json.dumps({"jobs": mgr.status()}).encode() + b"\n")
                await writer.drain()
            elif cmd == "shutdown":
                writer.write(json.dumps({"status": "shutting_down"}).encode() + b"\n")
                await writer.drain()
                break
            elif cmd == "start_universe_sim":
                # start a universe-mode simulation using provided phidesc or law_id
                params_local = msg
                # if law_id provided, load law and place into phidesc
                law_id = params_local.get("law_id")
                if law_id:
                    try:
                        law = mgr.engine.get_law(int(law_id))
                        params_local["phidesc"] = law.get("phidesc", {})
                    except Exception:
                        pass
                out = mgr.start_multiscale_sim(params_local)
                writer.write(json.dumps({"status": "ok", "universe_job": out}).encode() + b"\n")
                await writer.drain()
            elif cmd == "get_event_legality":
                # expects: sim_id and optionally node index or states
                sim_id = msg.get("sim_id")
                node = msg.get("node", None)
                if sim_id is None:
                    writer.write(json.dumps({"status": "error", "error": "sim_id required"}).encode() + b"\n")
                    await writer.drain()
                else:
                    sim = mgr.engine.get_simulation(int(sim_id))
                    if not sim:
                        writer.write(json.dumps({"status": "error", "error": "simulation not found"}).encode() + b"\n")
                        await writer.drain()
                    else:
                        # simplification: compare last two states if present
                        # attempt to extract stored result blob
                        try:
                            # decompress and read result blob directly
                            cur = mgr.engine._conn.cursor()
                            cur.execute("SELECT result_blob FROM simulations WHERE id = ?", (int(sim_id),))
                            row = cur.fetchone()
                            if not row:
                                raise RuntimeError("no result blob")
                            import zlib as _zlib, json as _json
                            data = _zlib.decompress(row[0])
                            result = _json.loads(data.decode("utf-8"))
                            trajs = result.get("results") or result.get("materialized") or []
                            if not trajs or len(trajs) < 1:
                                writer.write(json.dumps({"status": "error", "error": "no trajectories"}).encode() + b"\n")
                                await writer.drain()
                            else:
                                last = trajs[-1]
                                prev = trajs[-2] if len(trajs) >= 2 else trajs[-1]
                                legality = mgr.engine.is_event_legal(prev.get("final", prev), last.get("final", last))
                                writer.write(json.dumps({"status": "ok", "legality": legality}).encode() + b"\n")
                                await writer.drain()
                        except Exception as exc:
                            writer.write(json.dumps({"status": "error", "error": str(exc)}).encode() + b"\n")
                            await writer.drain()
            elif cmd == "list_branches":
                # support both job_id and sim_id
                job_id = msg.get("job_id")
                sim_id = msg.get("sim_id")
                try:
                    if sim_id is not None:
                        branches = mgr.engine.list_branches_for_sim(int(sim_id))
                    elif job_id is not None:
                        branches = mgr.engine.list_branches_for_job(int(job_id))
                    else:
                        branches = []
                    writer.write(json.dumps({"status": "ok", "branches": branches}).encode() + b"\n")
                except Exception as exc:
                    writer.write(json.dumps({"status": "error", "error": str(exc)}).encode() + b"\n")
                await writer.drain()
            elif cmd == "cancel_branch":
                # accept branch_id or child_job_id
                branch_id = msg.get("branch_id")
                child_job_id = msg.get("child_job_id")
                try:
                    if branch_id is not None:
                        # mark branch as cancelled in DB
                        mgr.engine.update_branch_status(int(branch_id), "cancelled")
                        # attempt to cancel child job if exists
                        cur = mgr.engine._conn.cursor()
                        cur.execute("SELECT child_job_id FROM branches WHERE id = ?", (int(branch_id),))
                        row = cur.fetchone()
                        if row and row[0]:
                            cid = int(row[0])
                            if cid in mgr._jobs:
                                mgr._jobs[cid]["status"] = "cancelled"
                        writer.write(json.dumps({"status": "ok", "cancelled_branch_id": int(branch_id)}).encode() + b"\n")
                    elif child_job_id is not None:
                        # find branch row(s) and mark cancelled
                        cur = mgr.engine._conn.cursor()
                        cur.execute("SELECT id FROM branches WHERE child_job_id = ?", (int(child_job_id),))
                        rows = cur.fetchall()
                        for r in rows:
                            bid = int(r[0])
                            mgr.engine.update_branch_status(bid, "cancelled")
                        if int(child_job_id) in mgr._jobs:
                            mgr._jobs[int(child_job_id)]["status"] = "cancelled"
                        writer.write(json.dumps({"status": "ok", "cancelled_child_job_id": int(child_job_id)}).encode() + b"\n")
                    else:
                        writer.write(json.dumps({"status": "error", "error": "branch_id or child_job_id required"}).encode() + b"\n")
                except Exception as exc:
                    writer.write(json.dumps({"status": "error", "error": str(exc)}).encode() + b"\n")
                await writer.drain()
            else:
                writer.write(json.dumps({"error": "unknown command", "cmd": cmd}).encode() + b"\n")
                await writer.drain()
    finally:
        # cleanup any PF subscriptions for this client
        for job_id, cb in pf_subs:
            try:
                mgr._pf_subscribers.get(job_id, []).remove(cb)
            except Exception:
                pass
        writer.close()
        await writer.wait_closed()
        LOG.info("Client disconnected: %s", addr)


async def run_server(host: str, port: int, db_path: str):
    engine = SimulacrumEngine(db_path=db_path)
    mgr = JobManager(engine)
    server = await asyncio.start_server(lambda r, w: handle_client(r, w, mgr), host, port)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets or [])
    LOG.info("Server listening on %s", addrs)
    # lightweight HTTP metrics endpoint
    async def metrics_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            # read request line
            line = await reader.readline()
            if not line:
                writer.close()
                await writer.wait_closed()
                return
            try:
                parts = line.decode("utf-8").strip().split()
                method = parts[0] if len(parts) > 0 else "GET"
                path = parts[1] if len(parts) > 1 else "/"
            except Exception:
                method, path = "GET", "/"
            # consume headers
            while True:
                h = await reader.readline()
                if not h or h == b"\r\n" or h == b"\n":
                    break
            if method.upper() != "GET" or path not in ("/metrics", "/metrics.json"):
                # support simple health check endpoint
                if method.upper() == "GET" and path == "/health":
                    # basic DB check
                    db_ok = False
                    db_err = None
                    try:
                        cur = engine._conn.cursor()
                        cur.execute("SELECT 1")
                        _ = cur.fetchone()
                        db_ok = True
                    except Exception as e:
                        db_ok = False
                        db_err = str(e)
                    body = json.dumps({"status": "ok" if db_ok else "error", "db_ok": db_ok, "db_error": db_err}).encode()
                    resp = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
                    writer.write(resp)
                    await writer.drain()
                    writer.close()
                    await writer.wait_closed()
                    return
                body = json.dumps({"error": "not_found"}).encode()
                resp = f"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
                writer.write(resp)
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                return
            if path == "/health":
                # basic DB check
                db_ok = False
                db_err = None
                try:
                    cur = engine._conn.cursor()
                    cur.execute("SELECT 1")
                    _ = cur.fetchone()
                    db_ok = True
                except Exception as e:
                    db_ok = False
                    db_err = str(e)
                body = json.dumps({"status": "ok" if db_ok else "error", "db_ok": db_ok, "db_error": db_err}).encode()
                resp = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
                writer.write(resp)
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                return
            # assemble per-job summaries
            job_summaries = {}
            for jid, info in mgr._jobs.items():
                job_summaries[jid] = {
                    "status": info.get("status"),
                    "start_ts": info.get("start_ts"),
                    "end_ts": info.get("end_ts"),
                    "duration": info.get("duration"),
                    "cpu_child_seconds": info.get("cpu_child_seconds"),
                }

            # system/process metrics
            sys_mem = None
            cpu_user = None
            cpu_system = None
            try:
                if psutil is not None:
                    p = psutil.Process(os.getpid())
                    mem_info = p.memory_info()
                    sys_mem = int(getattr(mem_info, "rss", 0))
                    ct = p.cpu_times()
                    cpu_user = float(getattr(ct, "user", 0.0))
                    cpu_system = float(getattr(ct, "system", 0.0))
                else:
                    ru = resource.getrusage(resource.RUSAGE_SELF)
                    sys_mem = int(getattr(ru, "ru_maxrss", 0))
                    cpu_user = float(getattr(ru, "ru_utime", 0.0))
                    cpu_system = float(getattr(ru, "ru_stime", 0.0))
            except Exception:
                pass

            payload = {
                "metrics": mgr._metrics,
                "jobs_total": len(mgr._jobs),
                "pf_jobs_active": len([j for j in mgr._pf_jobs.keys()]),
                "timestamp": time.time(),
                "system": {"rss_bytes": sys_mem, "cpu_user_seconds": cpu_user, "cpu_system_seconds": cpu_system},
                "jobs": job_summaries,
            }
            body = json.dumps(payload).encode()
            resp = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
            writer.write(resp)
            await writer.drain()
        except Exception:
            pass
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

    metrics_enabled = os.environ.get("METRICS_ENABLED", "1").lower() not in ("0", "false", "no", "")
    if metrics_enabled:
        metrics_port = int(os.environ.get("METRICS_PORT", "8000"))
        metrics_server = await asyncio.start_server(metrics_handler, host, metrics_port)
        addrs_metrics = ", ".join(str(s.getsockname()) for s in metrics_server.sockets or [])
        LOG.info("Metrics endpoint listening on %s", addrs_metrics)
        async with server, metrics_server:
            await asyncio.gather(server.serve_forever(), metrics_server.serve_forever())
    else:
        LOG.info("Metrics endpoint disabled by METRICS_ENABLED=0")
        async with server:
            await server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--db", default="time_engine.db")
    args = parser.parse_args()
    try:
        asyncio.run(run_server(args.host, args.port, args.db))
    except KeyboardInterrupt:
        LOG.info("Server interrupted, exiting.")


if __name__ == "__main__":
    main()


