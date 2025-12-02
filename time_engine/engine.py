import os
import time
import json
import sqlite3
import zlib
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
import math
import threading

try:
    import numpy as np
except Exception:
    np = None

try:
    # Prefer the project's CRCAAgent if available
    from swarms.agents.cr_ca_agent import CRCAAgent  # type: ignore
    HAS_CRCA = True
except Exception:
    CRCAAgent = None  # type: ignore
    HAS_CRCA = False

LOG = logging.getLogger("time_engine")
LOG.setLevel(logging.INFO)


def _compress_bytes(b: bytes) -> bytes:
    return zlib.compress(b)


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class SimulacrumEngine:
    """
    Minimal Simulacrum Engine core.

    - stores DB in the local folder (time_engine.db by default)
    - provides `run_simulation` which produces an ensemble of futures and persists a compressed snapshot
    - lightweight APIs: ingest, snapshot, query_graph

    This implementation intentionally keeps all code inside the time_engine folder.
    """

    def __init__(self, db_path: str = "time_engine.db", seed: Optional[int] = None):
        self.db_path = os.path.abspath(db_path)
        self._ensure_dir()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._init_db()
        self._jobs: Dict[int, Dict[str, Any]] = {}
        self._next_job_id = 1
        self.rng = np.random.default_rng(seed) if np is not None else None
        self._last_sim_id: Optional[int] = None
        # Instantiate CRCAAgent if available and enabled via env (can disable to avoid LLM checks)
        enable_agent = os.environ.get("ENABLE_AGENT", "1").lower() not in ("0", "false", "no", "")
        if HAS_CRCA and enable_agent:
            try:
                self.agent = CRCAAgent()
            except Exception:
                LOG.warning("CRCAAgent initialization failed; continuing without agent", exc_info=True)
                self.agent = None
        else:
            self.agent = None
        # thread-safety for agent access
        self._agent_lock = threading.Lock()
        # runtime knobs
        try:
            self._max_particles = int(os.environ.get("MAX_PARTICLES", "4096"))
        except Exception:
            self._max_particles = 4096

    def _ensure_dir(self) -> None:
        d = os.path.dirname(self.db_path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY,
                created_ts REAL,
                name TEXT,
                x0_json TEXT,
                phidesc_json TEXT,
                config_json TEXT,
                result_blob BLOB
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                sim_id INTEGER,
                ts REAL,
                parent_id INTEGER,
                state_macro_json TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY,
                sim_id INTEGER,
                name TEXT,
                proto_blob BLOB,
                created_ts REAL
            );
            """
        )
        # Blobs table for content-addressed storage (deduplication)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS blobs (
                hash TEXT PRIMARY KEY,
                blob BLOB,
                size INTEGER,
                created_ts REAL
            );
            """
        )
        # Checkpoints table stores compressed microstate checkpoints
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY,
                sim_id INTEGER,
                node_id INTEGER,
                ts REAL,
                blob_hash TEXT,
                resolution_tier TEXT
            );
            """
        )
        # Laws / models registry
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS laws (
                id INTEGER PRIMARY KEY,
                name TEXT,
                description TEXT,
                phidesc_json TEXT,
                created_ts REAL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS branches (
                id INTEGER PRIMARY KEY,
                parent_job_id INTEGER,
                child_job_id INTEGER,
                parent_sim_id INTEGER,
                child_sim_id INTEGER,
                tile_id INTEGER,
                t INTEGER,
                payload_blob BLOB,
                created_ts REAL,
                status TEXT,
                depth INTEGER
            );
            """
        )
        self._conn.commit()
        # ensure extended schema for laws and checkpoints
        self._ensure_extended_schema()

    def _ensure_extended_schema(self) -> None:
        """Add optional columns to laws and checkpoints tables if missing."""
        cur = self._conn.cursor()
        # laws: add type, legal_constraints, canonical_seed_template, adapter_hooks
        cur.execute("PRAGMA table_info(laws)")
        cols = [r[1] for r in cur.fetchall()]
        if "type" not in cols:
            try:
                cur.execute("ALTER TABLE laws ADD COLUMN type TEXT")
            except Exception:
                pass
        if "legal_constraints" not in cols:
            try:
                cur.execute("ALTER TABLE laws ADD COLUMN legal_constraints TEXT")
            except Exception:
                pass
        if "canonical_seed_template" not in cols:
            try:
                cur.execute("ALTER TABLE laws ADD COLUMN canonical_seed_template TEXT")
            except Exception:
                pass
        if "adapter_hooks" not in cols:
            try:
                cur.execute("ALTER TABLE laws ADD COLUMN adapter_hooks TEXT")
            except Exception:
                pass
        # checkpoints: add boundary_descriptor_hash, entropy_snapshot
        cur.execute("PRAGMA table_info(checkpoints)")
        cols_cp = [r[1] for r in cur.fetchall()]
        if "boundary_descriptor_hash" not in cols_cp:
            try:
                cur.execute("ALTER TABLE checkpoints ADD COLUMN boundary_descriptor_hash TEXT")
            except Exception:
                pass
        if "entropy_snapshot" not in cols_cp:
            try:
                cur.execute("ALTER TABLE checkpoints ADD COLUMN entropy_snapshot TEXT")
            except Exception:
                pass
        # simulations: add parent_job_id column to record job lineage
        cur.execute("PRAGMA table_info(simulations)")
        sim_cols = [r[1] for r in cur.fetchall()]
        if "parent_job_id" not in sim_cols:
            try:
                cur.execute("ALTER TABLE simulations ADD COLUMN parent_job_id INTEGER")
            except Exception:
                pass
        self._conn.commit()

    # ================== Laws validation & provenance ==================
    def validate_law(self, phidesc: Dict[str, Any]) -> bool:
        """Basic validation of a phidesc/law. Returns True if acceptable."""
        if not isinstance(phidesc, dict):
            return False
        # require a type and params keys
        if "type" not in phidesc:
            return False
        if "params" not in phidesc:
            return False
        # legal_constraints if present must be dict
        if "legal_constraints" in phidesc and not isinstance(phidesc["legal_constraints"], dict):
            return False
        return True

    def register_law_extended(self, name: str, phidesc: Dict[str, Any], description: str = "") -> int:
        """Register a phidesc (law/model) in the DB and return id. Validates extended fields."""
        if not self.validate_law(phidesc):
            raise ValueError("phidesc failed validation; must include 'type' and 'params'")
        cur = self._conn.cursor()
        ltype = phidesc.get("type", "")
        legal_constraints = json.dumps(phidesc.get("legal_constraints", {}))
        canonical_seed = json.dumps(phidesc.get("seed_config", {}))
        adapter_hooks = json.dumps(phidesc.get("adapter_hooks", {}))
        cur.execute("INSERT INTO laws(name, description, phidesc_json, created_ts, type, legal_constraints, canonical_seed_template, adapter_hooks) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (name, description, json.dumps(phidesc), time.time(), ltype, legal_constraints, canonical_seed, adapter_hooks))
        lid = cur.lastrowid
        self._conn.commit()
        return lid

    # ================== Conservation & legality checks ==================
    def compute_energy_momentum(self, state: Dict[str, Any], phidesc: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Compute a simple effective energy/momentum diagnostics for the macro state."""
        # For cosmology use rho_total if present; else sum of squares proxy
        out = {}
        rho = state.get("rho", None)
        if rho is None and "rho_total" in state:
            rho = state.get("rho_total")
        if rho is not None:
            out["rho"] = float(rho)
        else:
            # proxy: sum squares
            vals = [float(v) for v in state.values() if isinstance(v, (int, float))]
            out["rho_proxy"] = float(sum(v * v for v in vals))
        # placeholder momentum/pressure
        out["p_proxy"] = float(state.get("p", 0.0))
        return out

    def check_conservation(self, parent_T: Dict[str, float], child_T: Dict[str, float], tol: float = 1e-6) -> bool:
        """Check simple conservation: rho difference relative tolerance."""
        pr = parent_T.get("rho", parent_T.get("rho_proxy", 0.0))
        cr = child_T.get("rho", child_T.get("rho_proxy", 0.0))
        if pr == 0.0:
            return abs(cr - pr) <= tol
        rel = abs(cr - pr) / max(abs(pr), 1e-30)
        return rel <= tol

    def is_event_legal(self, parent_state: Dict[str, Any], candidate_state: Dict[str, Any], law_id: Optional[int] = None, dt: float = 1.0, phidesc: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return legality assessment for candidate transition."""
        reason = {"legal": True, "checks": []}
        # compute energy diagnostics
        pT = self.compute_energy_momentum(parent_state, None)
        cT = self.compute_energy_momentum(candidate_state, None)
        conserved = self.check_conservation(pT, cT, tol=float((self.get_default_universe_config().get("conservation_tol", 1e-6))))
        reason["checks"].append({"conservation": conserved, "parent_rho": pT.get("rho", pT.get("rho_proxy")), "child_rho": cT.get("rho", cT.get("rho_proxy"))})
        if not conserved:
            reason["legal"] = False
            reason["reason"] = "conservation_violation"
            reason["suggested_action"] = "branch_or_repair"
            return reason
        # simple causality check: norm delta must be <= c * dt * factor
        # compute vector norms
        import math as _math
        pvec = [float(v) for v in parent_state.values() if isinstance(v, (int, float))]
        cvec = [float(v) for v in candidate_state.values() if isinstance(v, (int, float))]
        if len(pvec) and len(cvec):
            import numpy as _np
            n = min(len(pvec), len(cvec))
            diff = _np.linalg.norm(_np.array(cvec[:n]) - _np.array(pvec[:n]))
            # rough speed limit threshold
            if diff > (1.0 * dt * 1e6):  # heuristic scale, depends on units
                reason["legal"] = False
                reason["reason"] = "causality_violation"
                reason["suggested_action"] = "branch"
                reason["delta"] = float(diff)
                return reason
        # Phase 7: Quantum energy conservation check
        if phidesc and phidesc.get("physics_modes", {}).get("quantum_backreaction", False):
            quantum_check = self.check_quantum_energy_conservation(parent_state, candidate_state, phidesc)
            reason["checks"].append({"quantum_energy": quantum_check})
            if not quantum_check.get("conserved", True):
                reason["legal"] = False
                reason["reason"] = "quantum_energy_violation"
                reason["suggested_action"] = "repair"
        
        return reason

    def check_quantum_energy_conservation(
        self,
        parent_state: Dict[str, Any],
        candidate_state: Dict[str, Any],
        phidesc: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check quantum energy conservation between parent and candidate states.
        Returns dict with 'conserved' (bool) and 'delta_energy' (float).
        """
        try:
            # Extract field modes from states
            parent_modes = []
            candidate_modes = []
            for k in range(8):  # default n_modes
                mode_key = f"phi_mode_{k}"
                parent_modes.append(float(parent_state.get(mode_key, 0.0)))
                candidate_modes.append(float(candidate_state.get(mode_key, 0.0)))
            
            if not parent_modes and not candidate_modes:
                # No quantum modes, skip check
                return {"conserved": True, "delta_energy": 0.0}
            
            # Get QFT adapter to compute energy
            adapter = self.get_physics_adapter(phidesc)
            if adapter is None or not hasattr(adapter, "compute_stress_energy"):
                return {"conserved": True, "delta_energy": 0.0}
            
            # Compute energy in parent and candidate
            parent_se = adapter.compute_stress_energy(parent_modes)
            candidate_se = adapter.compute_stress_energy(candidate_modes)
            
            parent_energy = parent_se.get("rho", 0.0)
            candidate_energy = candidate_se.get("rho", 0.0)
            delta_energy = abs(candidate_energy - parent_energy)
            
            # Tolerance: allow small fluctuations (quantum uncertainty)
            tol = float(phidesc.get("params", {}).get("quantum_energy_tol", 1e-6))
            conserved = delta_energy < tol * max(abs(parent_energy), 1e-12)
            
            return {"conserved": conserved, "delta_energy": float(delta_energy), "parent_energy": float(parent_energy), "candidate_energy": float(candidate_energy)}
        except Exception:
            # On error, assume conserved
            return {"conserved": True, "delta_energy": 0.0}

    # ================== Holographic boundary encoding ==================
    def holographic_encode(self, boundary_state: Dict[str, Any]) -> Optional[str]:
        """Compress and store boundary descriptor; return blob hash."""
        try:
            b = json.dumps(boundary_state).encode("utf-8")
            h = self._store_blob(b)
            return h
        except Exception:
            return None

    # ================== Lightcone / nested controller / provenance ==================
    def lightcone_prune(self, tiles: List[Dict[str, Any]], center: int = 0, horizon: int = 1, c: float = 1.0, resolution_scale: float = 1.0) -> List[int]:
        """
        Simple lightcone pruning: return list of tile ids that lie within causal radius.
        Uses tile index distance as proxy for spatial distance (1D logical tiling).
        """
        keep = []
        for t in tiles:
            tid = int(t.get("id", 0))
            dist = abs(tid - center)
            # convert to metric units using resolution_scale (user provided)
            if dist * resolution_scale <= c * horizon:
                keep.append(tid)
        return keep

    def _capture_provenance(self, x0: Dict[str, Any], phidesc: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Capture reproducible provenance: seed_hash, rng_state_hash, adapter info."""
        prov: Dict[str, Any] = {}
        try:
            seed_hash = self.store_sim_seed(x0, phidesc)
            prov["seed_hash"] = seed_hash
        except Exception:
            prov["seed_hash"] = None
        try:
            rng_state = None
            if hasattr(self, "rng") and self.rng is not None:
                try:
                    rng_state = json.dumps(self.rng.bit_generator.state)
                except Exception:
                    rng_state = None
            if rng_state is not None:
                prov["rng_state_hash"] = _sha256_hex(rng_state.encode("utf-8"))
                # store rng_state as blob for reproducibility
                self._store_blob(rng_state.encode("utf-8"))
            else:
                prov["rng_state_hash"] = None
        except Exception:
            prov["rng_state_hash"] = None
        prov["adapter"] = phidesc.get("field_model") if phidesc else None
        prov["law_type"] = phidesc.get("type") if phidesc else None
        prov["created_ts"] = time.time()
        return prov

    def repair_candidate(self, parent: Dict[str, Any], candidate: Dict[str, Any], conserve_key: str = "rho", eps: float = 1e-12) -> Dict[str, Any]:
        """
        Heuristic repair: scale candidate changes towards parent to reduce conservation violation.
        Returns a new candidate dict.
        """
        try:
            pT = self.compute_energy_momentum(parent)
            cT = self.compute_energy_momentum(candidate)
            pr = float(pT.get(conserve_key, pT.get("rho_proxy", 0.0)))
            cr = float(cT.get(conserve_key, cT.get("rho_proxy", 0.0)))
            if cr <= 0 or pr <= 0:
                # fallback: small interpolation
                alpha = 0.5
            else:
                ratio = pr / (cr + eps)
                alpha = min(1.0, ratio)
            repaired = {}
            for k in parent.keys():
                pv = float(parent.get(k, 0.0))
                cv = float(candidate.get(k, pv))
                repaired[k] = float(pv + (cv - pv) * alpha)
            return repaired
        except Exception:
            return candidate

    # ================== Content-addressed blob helpers ==================
    def _store_blob(self, data: bytes) -> str:
        """Compress+store bytes in blobs table and return sha256 hex."""
        h = _sha256_hex(data)
        cur = self._conn.cursor()
        cur.execute("SELECT 1 FROM blobs WHERE hash = ?", (h,))
        if cur.fetchone():
            return h
        comp = _compress_bytes(data)
        cur.execute("INSERT INTO blobs(hash, blob, size, created_ts) VALUES (?, ?, ?, ?)", (h, comp, len(comp), time.time()))
        self._conn.commit()
        return h

    def _get_blob(self, h: str) -> Optional[bytes]:
        cur = self._conn.cursor()
        cur.execute("SELECT blob FROM blobs WHERE hash = ?", (h,))
        row = cur.fetchone()
        if not row:
            return None
        try:
            return zlib.decompress(row[0])
        except Exception:
            return None

    # ================== Checkpoint helpers ==================
    def create_checkpoint(self, sim_id: int, node_id: int, state: Dict[str, Any], resolution_tier: str = "micro") -> int:
        """Create a compressed checkpoint for state and reference it in checkpoints table."""
        data = json.dumps(state).encode("utf-8")
        h = self._store_blob(data)
        cur = self._conn.cursor()
        cur.execute("INSERT INTO checkpoints(sim_id, node_id, ts, blob_hash, resolution_tier, boundary_descriptor_hash, entropy_snapshot) VALUES (?, ?, ?, ?, ?, ?, ?)", (sim_id, node_id, time.time(), h, resolution_tier, None, None))
        cid = cur.lastrowid
        self._conn.commit()
        return cid

    def update_checkpoint_metadata(self, checkpoint_id: int, boundary_descriptor_hash: Optional[str], entropy_snapshot: Optional[Dict[str, Any]]) -> None:
        """Attach boundary descriptor hash and entropy snapshot to a checkpoint row."""
        cur = self._conn.cursor()
        try:
            cur.execute("UPDATE checkpoints SET boundary_descriptor_hash = ?, entropy_snapshot = ? WHERE id = ?", (boundary_descriptor_hash, json.dumps(entropy_snapshot) if entropy_snapshot is not None else None, checkpoint_id))
            self._conn.commit()
        except Exception:
            pass

    # ================== Branch persistence ==================
    def register_branch(self, parent_job_id: int, child_job_id: int, parent_sim_id: Optional[int], child_sim_id: Optional[int], tile_id: Optional[int], t_val: Optional[int], payload: Dict[str, Any], depth: int = 0, status: str = "scheduled") -> int:
        """Persist branch metadata linking parent job -> child job. Returns branch id."""
        try:
            blob = _compress_bytes(json.dumps(payload).encode("utf-8"))
            cur = self._conn.cursor()
            cur.execute("INSERT INTO branches(parent_job_id, child_job_id, parent_sim_id, child_sim_id, tile_id, t, payload_blob, created_ts, status, depth) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (parent_job_id, child_job_id, parent_sim_id, child_sim_id, tile_id, t_val, blob, time.time(), status, int(depth)))
            bid = cur.lastrowid
            self._conn.commit()
            return bid
        except Exception:
            return -1

    def update_branch_with_child_sim(self, child_job_id: int, child_sim_id: int) -> None:
        """Update branches rows that reference child_job_id with the produced child_sim_id."""
        try:
            cur = self._conn.cursor()
            cur.execute("UPDATE branches SET child_sim_id = ?, status = ? WHERE child_job_id = ?", (int(child_sim_id), "completed", int(child_job_id)))
            self._conn.commit()
        except Exception:
            pass

    def list_branches_for_job(self, job_id: int) -> List[Dict[str, Any]]:
        """Return branches where job_id is parent or child."""
        out = []
        try:
            cur = self._conn.cursor()
            cur.execute("SELECT id, parent_job_id, child_job_id, parent_sim_id, child_sim_id, tile_id, t, payload_blob, created_ts, status, depth FROM branches WHERE parent_job_id = ? OR child_job_id = ? ORDER BY created_ts DESC", (int(job_id), int(job_id)))
            for row in cur.fetchall():
                bid, pj, cj, psi, csi, tid, tval, pblob, cts, status, depth = row
                try:
                    payload = json.loads(zlib.decompress(pblob).decode("utf-8"))
                except Exception:
                    payload = None
                out.append({"id": int(bid), "parent_job_id": pj, "child_job_id": cj, "parent_sim_id": psi, "child_sim_id": csi, "tile_id": tid, "t": tval, "payload": payload, "created_ts": float(cts), "status": status, "depth": int(depth)})
        except Exception:
            pass
        return out

    def list_branches_for_sim(self, sim_id: int) -> List[Dict[str, Any]]:
        """Return branches where parent_sim_id or child_sim_id matches sim_id."""
        out = []
        try:
            cur = self._conn.cursor()
            cur.execute("SELECT id, parent_job_id, child_job_id, parent_sim_id, child_sim_id, tile_id, t, payload_blob, created_ts, status, depth FROM branches WHERE parent_sim_id = ? OR child_sim_id = ? ORDER BY created_ts DESC", (int(sim_id), int(sim_id)))
            for row in cur.fetchall():
                bid, pj, cj, psi, csi, tid, tval, pblob, cts, status, depth = row
                try:
                    payload = json.loads(zlib.decompress(pblob).decode("utf-8"))
                except Exception:
                    payload = None
                out.append({"id": int(bid), "parent_job_id": pj, "child_job_id": cj, "parent_sim_id": psi, "child_sim_id": csi, "tile_id": tid, "t": tval, "payload": payload, "created_ts": float(cts), "status": status, "depth": int(depth)})
        except Exception:
            pass
        return out

    def update_branch_status(self, branch_id: int, status: str) -> None:
        """Update branch status (e.g., cancelled, scheduled, completed)."""
        try:
            cur = self._conn.cursor()
            cur.execute("UPDATE branches SET status = ? WHERE id = ?", (str(status), int(branch_id)))
            self._conn.commit()
        except Exception:
            pass

    def nested_multiscale_controller(self, tile_states: List[Dict[str, Any]], global_state: Dict[str, Any], ph: Dict[str, Any], rng: Any, horizon: int, dt: float, progress_cb: Optional[Any] = None) -> Dict[str, Any]:
        """
        Nested multiscale controller skeleton:
        - runs coarse global update
        - selects tiles for high-res evolution
        - checks event legality and updates entropy budget
        - checkpoints and holographic-encodes boundary when needed
        """
        tiles_cfg = ph.get("tiles", {}) or {}
        n_tiles = int(tiles_cfg.get("n_tiles", max(1, len(tile_states))))
        tile_history = {t["id"]: [] for t in tile_states}
        entropy_budget = float(ph.get("entropy_budget", 1e6))
        refine_n = int(tiles_cfg.get("refine_n", max(1, n_tiles // 2)))
        # per-tile previous states
        prev_states = {t["id"]: dict(t["state"]) for t in tile_states}

        branches: List[Dict[str, Any]] = []
        policy = (ph.get("branching_policy") or "annotate").lower()
        
        # Phase 7: Check for quantum backreaction
        quantum_backreaction_enabled = ph.get("physics_modes", {}).get("quantum_backreaction", False)
        qft_adapter = None
        if quantum_backreaction_enabled:
            qft_adapter = self.get_physics_adapter(ph)
        
        for t in range(int(horizon)):
            # Phase 7: Compute quantum backreaction from tiles if enabled
            quantum_correction = {"delta_rho": 0.0, "delta_p": 0.0}
            if quantum_backreaction_enabled and qft_adapter is not None:
                total_delta_rho = 0.0
                total_delta_p = 0.0
                for tile in tile_states:
                    field_modes = []
                    for k in range(8):
                        mode_key = f"phi_mode_{k}"
                        if mode_key in tile["state"]:
                            field_modes.append(float(tile["state"][mode_key]))
                    if field_modes:
                        correction = self.compute_quantum_backreaction(qft_adapter, field_modes, ph.get("params", {}))
                        total_delta_rho += correction.get("delta_rho", 0.0)
                        total_delta_p += correction.get("delta_p", 0.0)
                if n_tiles > 0:
                    quantum_correction = {
                        "delta_rho": float(total_delta_rho / n_tiles),
                        "delta_p": float(total_delta_p / n_tiles),
                    }
            
            # coarse global update
            if ph.get("physics_modes", {}).get("gravity_model", "").upper() in ("FLRW", "GR_PERTURBATIONS"):
                try:
                    x_vec = (float(global_state.get("a", 1.0)), float(global_state.get("phi", 0.0)), float(global_state.get("phi_dot", 0.0)))
                    params_corrected = dict(ph.get("params", {}))
                    if quantum_backreaction_enabled and quantum_correction["delta_rho"] != 0.0:
                        params_corrected["Omega_L"] = float(params_corrected.get("Omega_L", 0.685)) + quantum_correction["delta_rho"]
                    x_next_vec, diag = self.evolve_spacetime_full(x_vec, params_corrected, dt)
                    global_state["a"], global_state["phi"], global_state["phi_dot"] = float(x_next_vec[0]), float(x_next_vec[1]), float(x_next_vec[2])
                except Exception:
                    global_state["a"] = float(global_state.get("a", 1.0) * (1.0 + 1e-3 * dt))

            # compute simple per-tile error/importance metric (variance of state values)
            metrics = []
            for tile in tile_states:
                vals = [float(v) for v in tile["state"].values() if isinstance(v, (int, float))]
                var = float(np.var(vals)) if np is not None and vals else 0.0
                metrics.append((tile["id"], var))
            metrics.sort(key=lambda x: x[1], reverse=True)
            to_refine = [mid for mid, _ in metrics[:refine_n]]

            for tile in tile_states:
                tid = tile["id"]
                s = tile["state"]
                prev = dict(s)
                # decide resolution: high-res if selected
                high_res = tid in to_refine
                # apply boundary (correlated) as before
                bparams = (ph.get("boundary_model") or {}).get("params", {}) if ph.get("boundary_model") else {}
                if ph.get("boundary_model", {}).get("type", "") == "causal_sampling":
                    prevb = tile.get("boundary", 0.0)
                    newb = self._ou_step(prevb, float(bparams.get("theta", 0.1)), float(bparams.get("mu", 0.0)), float(bparams.get("sigma", 1e-6)), dt, rng)
                    tile["boundary"] = newb
                    for bv in bparams.get("vars", list(s.keys())):
                        s[bv] = float(s.get(bv, 0.0) + newb)

                # choose adapter or method depending on high_res flag
                ph_local = dict(ph)
                if high_res:
                    ph_local["evolution_solver"] = ph_local.get("evolution_solver", {})
                    ph_local["evolution_solver"]["time_integrator"] = ph_local.get("evolution_solver", {}).get("time_integrator", "RK45")
                else:
                    ph_local["evolution_solver"] = ph_local.get("evolution_solver", {})
                    ph_local["evolution_solver"]["time_integrator"] = ph_local.get("evolution_solver", {}).get("coarse_integrator", "Leapfrog")
                
                # Phase 7: Scale-dependent physics switching
                coherence_length = float(ph.get("physics_modes", {}).get("quantum_coherence_length", 1e-35))
                tile_scale = float(ph.get("tiles", {}).get("tile_scale", 1e-3))
                use_quantum = tile_scale < coherence_length
                if use_quantum and not ph_local.get("field_model"):
                    ph_local["field_model"] = "qft_scalar"

                # propose candidate update
                try:
                    candidate = self.evolve_state(dict(s), dt=dt, method=ph_local.get("evolution_solver", {}).get("time_integrator", "RK45"), phidesc=ph_local)
                except Exception:
                    candidate = dict(s)

                # legality check (includes quantum energy conservation if enabled)
                legality = self.is_event_legal(prev, candidate, law_id=ph.get("law_id", None), dt=dt, phidesc=ph_local)
                diag = {"t": t, "mean": {k: float(v) for k, v in candidate.items()}, "legality": legality, "high_res": bool(high_res), "adapter_ms": float(getattr(self, "_last_adapter_ms", 0.0))}
                # handle automatic branching policy
                if not legality.get("legal", True):
                    # REJECT: revert to previous state
                    if policy == "reject":
                        diag["action"] = "rejected"
                        diag["reason"] = legality.get("reason")
                        tile_history[tid].append(diag)
                        tile["state"] = prev
                        # small checkpoint noting rejection
                        cid = self.create_checkpoint(sim_id=0, node_id=tid, state=prev, resolution_tier="micro")
                        self.update_checkpoint_metadata(cid, None, {"rejection_reason": legality.get("reason")})
                        continue
                    # REPAIR: attempt to adjust candidate to satisfy conservation heuristically
                    if policy == "repair":
                        repaired = self.repair_candidate(prev, candidate)
                        new_legality = self.is_event_legal(prev, repaired, law_id=ph.get("law_id", None), dt=dt, phidesc=ph_local)
                        if new_legality.get("legal", False):
                            diag["action"] = "repaired"
                            diag["repair_info"] = {"before": legality.get("reason"), "after": "ok"}
                            candidate = repaired
                            tile_history[tid].append(diag)
                            tile["state"] = candidate
                        else:
                            diag["action"] = "repair_failed"
                            diag["reason"] = new_legality.get("reason")
                            tile_history[tid].append(diag)
                            tile["state"] = prev
                            cid = self.create_checkpoint(sim_id=0, node_id=tid, state=repaired, resolution_tier="micro")
                            self.update_checkpoint_metadata(cid, None, {"repair_failed": True})
                        continue
                    # BRANCH: create a side branch and persist it, keep main timeline unchanged
                    if policy == "branch" or policy == "auto_branch":
                        branch = {"tile_id": tid, "t": t, "parent": prev, "candidate": candidate}
                        branches.append(branch)
                        bid = len(branches) - 1
                        diag["action"] = "branched"
                        diag["branch_id"] = bid
                        # checkpoint branch candidate
                        cid = self.create_checkpoint(sim_id=0, node_id=tid, state=candidate, resolution_tier="micro")
                        b_desc_hash = self.holographic_encode({"tile_id": tid, "t": t, "boundary": tile.get("boundary", 0.0)})
                        self.update_checkpoint_metadata(cid, b_desc_hash, {"branch_id": bid})
                        tile_history[tid].append(diag)
                        # schedule auto-branch if requested: emit special payload
                        if policy == "auto_branch" and progress_cb:
                            try:
                                # create branch job descriptor: use candidate as seed and include ph descriptors
                                branch_phidesc = dict(ph or {})
                                branch_seed = dict(candidate)
                                branch_params = {"seed": branch_seed, "phidesc": branch_phidesc, "horizon": int(ph.get("branch_horizon", max(1, horizon // 2))), "ensemble": int(ph.get("branch_ensemble", 1))}
                                payload_branch = {"action": "auto_branch_request", "parent_job": ph.get("_job_id"), "branch": branch_params, "tile_id": tid, "t": t, "branch_id": bid}
                                progress_cb(payload_branch)
                            except Exception:
                                pass
                        # do not commit candidate to main tile
                        tile["state"] = prev
                        continue
                    # default: annotate and keep candidate
                    diag["action"] = "annotate"
                    diag["flag"] = legality.get("reason", "illegal")
                    tile_history[tid].append(diag)
                    tile["state"] = candidate
                else:
                    # legal -> commit
                    diag["action"] = "commit"
                    tile_history[tid].append(diag)
                    tile["state"] = candidate
                # entropy accounting: proxy by sum(abs(delta))
                try:
                    prev_vals = np.array([float(v) for v in prev.values()]) if np is not None else None
                    cand_vals = np.array([float(v) for v in candidate.values()]) if np is not None else None
                    delta_entropy = float(np.sum(np.abs(cand_vals - prev_vals))) if prev_vals is not None else 0.0
                except Exception:
                    delta_entropy = 0.0
                entropy_budget -= delta_entropy
                if entropy_budget < 0:
                    diag["entropy_violation"] = True
                    diag["entropy_remaining"] = float(entropy_budget)
                else:
                    diag["entropy_remaining"] = float(entropy_budget)

                # checkpoint occasionally with holographic boundary encode
                if t % max(1, horizon // 4) == 0 or diag.get("flag") or diag.get("entropy_violation"):
                    cid = self.create_checkpoint(sim_id=0, node_id=tid, state=candidate, resolution_tier="micro")
                    # holographic encode boundary descriptor
                    b_desc_hash = self.holographic_encode({"tile_id": tid, "t": t, "boundary": tile.get("boundary", 0.0)})
                    self.update_checkpoint_metadata(cid, b_desc_hash, {"entropy": float(entropy_budget)})
                    diag["checkpoint_id"] = cid

                # push progress
                if progress_cb:
                    try:
                        payload = {"job_id": ph.get("_job_id") if ph else None, "tile_id": tid, "t": t, "diagnostics": diag}
                        progress_cb(payload)
                    except Exception:
                        pass

            # end per-tile loop

        # assemble tile summaries
        tiles_out = []
        for tid, hist in tile_history.items():
            tiles_out.append({"id": tid, "steps": len(hist), "last": hist[-1] if hist else {}})

        aggregate = {"tiles": len(tiles_out), "entropy_remaining": float(entropy_budget)}
        x0_seed = tile_states[0]["state"] if tile_states else {}
        result = {"seed": x0_seed, "horizon": horizon, "ensemble": 1, "tiles": tiles_out, "aggregate": aggregate, "created_ts": time.time()}
        # use first tile state as representative seed if no explicit seed provided
        x0_seed = tile_states[0]["state"] if tile_states else {}
        sim_id = self._store_simulation_result("multiscale_nested_run", x0_seed, ph or {}, {"horizon": horizon, "controller": "nested"}, result)
        return {"sim_id": sim_id, "summary": aggregate, "tiles": tiles_out}

    def load_checkpoint_state(self, checkpoint_id: int) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT blob_hash FROM checkpoints WHERE id = ?", (checkpoint_id,))
        row = cur.fetchone()
        if not row:
            return None
        h = row[0]
        b = self._get_blob(h)
        if b is None:
            return None
        try:
            return json.loads(b.decode("utf-8"))
        except Exception:
            return None

    # ================== High-level storage API (Σ mapping) ==================
    def store_sim_seed(self, x0: Dict[str, Any], phidesc: Dict[str, Any]) -> str:
        """Store a procedural seed descriptor (X0 + phidesc) as a blob and return its hash."""
        payload = {"x0": x0, "phidesc": phidesc}
        return self._store_blob(json.dumps(payload).encode("utf-8"))

    def load_sim_seed(self, seed_hash: str) -> Optional[Dict[str, Any]]:
        b = self._get_blob(seed_hash)
        if b is None:
            return None
        return json.loads(b.decode("utf-8"))

    # ================== Dynamic load manager (D) & GC (G) ==================
    def load_state(self, node_id: int, resolution: str = "macro") -> Optional[Dict[str, Any]]:
        """
        Load a state by node id. If resolution=='macro' returns event macro; if 'micro' attempts to find a checkpoint.
        """
        cur = self._conn.cursor()
        if resolution == "macro":
            cur.execute("SELECT state_macro_json FROM events WHERE id = ?", (node_id,))
            row = cur.fetchone()
            if not row:
                return None
            try:
                return json.loads(row[0])
            except Exception:
                return None
        else:
            # try latest checkpoint for node
            cur.execute("SELECT id FROM checkpoints WHERE node_id = ? ORDER BY ts DESC LIMIT 1", (node_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self.load_checkpoint_state(row[0])

    def gc_sweep(self, retain_days: int = 30) -> Dict[str, int]:
        """
        Simple GC: archive/delete blobs not referenced by simulations/checkpoints older than retain_days.
        Returns counts of removed items.
        """
        cutoff = time.time() - float(retain_days) * 24 * 3600.0
        cur = self._conn.cursor()
        # find blobs referenced by recent checkpoints
        cur.execute("SELECT DISTINCT blob_hash FROM checkpoints WHERE ts >= ?", (cutoff,))
        recent = set(r[0] for r in cur.fetchall())
        # find all blobs
        cur.execute("SELECT hash FROM blobs")
        all_hashes = [r[0] for r in cur.fetchall()]
        removed = 0
        for h in all_hashes:
            if h in recent:
                continue
            # safe delete (no foreign key enforced), remove blob
            cur.execute("DELETE FROM blobs WHERE hash = ?", (h,))
            removed += 1
        self._conn.commit()
        return {"blobs_removed": removed}

    # ================== Utility: default universe config & simple analysis ==================
    def get_default_universe_config(self) -> Dict[str, Any]:
        """Return a compact default universe config (toy economic variables)."""
        return {
            "variables": ["field1", "field2", "volume", "volatility"],
            "edges": [("field2", "field1"), ("volume", "field1")],
            "initial": {"field1": 2000.0, "field2": 40000.0, "volume": 1e6, "volatility": 0.05},
        }

    def get_domain_phi(self, phidesc: Dict[str, Any]) -> Any:
        """
        Return a domain-specific discrete evolution function f(state, t, rng) -> next_state.
        Supported phidesc['type'] values:
          - 'econ_simple' : simple economic interacting variables (ETH, BTC, volume, volatility)
          - 'physics_simple': toy physical dynamics (harmonic oscillator)
        """
        typ = phidesc.get("type", "econ_simple")
        params = phidesc.get("params", {})

        if typ == "econ_simple":
            alpha = float(params.get("alpha", 0.001))   # coupling BTC->ETH
            beta = float(params.get("beta", 1e-7))      # volume effect scale
            gamma = float(params.get("gamma", 0.0001))  # volatility damping
            rho = float(params.get("rho", 0.9))         # volume persistence

            def step(state: Dict[str, float], t: int, rng: Any) -> Dict[str, float]:
                s = dict(state)
                ETH = float(s.get("ETH_price", 0.0))
                BTC = float(s.get("BTC_price", 0.0))
                vol = float(s.get("volume", 0.0))
                sigma = float(s.get("volatility", 0.0))
                # Coupling: ETH drifts toward BTC with friction, affected by volume and volatility
                dETH = alpha * (BTC - ETH) + beta * (vol - 1e6) - gamma * sigma * ETH
                # BTC mean reverts slowly to a baseline (random walk with small drift)
                dBTC = 0.0001 * (BTC * 0.0) + 0.00001 * BTC
                # Volume AR(1)
                new_vol = rho * vol + (1.0 - rho) * 1e6 + float(rng.normal(0, max(vol * 0.01, 1.0)))
                # Volatility mean-reversion
                theta = 0.05
                kappa = 0.1
                dSigma = kappa * (theta - sigma) + float(rng.normal(0, 0.001))
                next_state = {
                    "ETH_price": float(max(0.0, ETH + dETH + float(rng.normal(0, abs(ETH) * 0.005 + 1e-6)))),
                    "BTC_price": float(max(0.0, BTC + dBTC + float(rng.normal(0, abs(BTC) * 0.005 + 1e-6)))),
                    "volume": float(max(0.0, new_vol)),
                    "volatility": float(max(0.0, sigma + dSigma)),
                }
                return next_state

            return step

        if typ == "physics_simple":
            # simple harmonic oscillator for a 1D particle: x'' = -omega^2 x - c x'
            omega = float(params.get("omega", 1.0))
            damping = float(params.get("damping", 0.01))

            def step(state: Dict[str, float], t: int, rng: Any) -> Dict[str, float]:
                # expect keys 'x' and 'v'
                x = float(state.get("x", 0.0))
                v = float(state.get("v", 0.0))
                dt = 1.0
                a = -omega * omega * x - damping * v
                v_next = v + a * dt
                x_next = x + v_next * dt
                return {"x": float(x_next), "v": float(v_next)}

            return step

        if typ == "cosmology_lcdm" or typ == "cosmology":
            # use cosmology step generator
            return self._cosmo_step_generator(phidesc)

        # Phase 7: GR perturbations mode
        physics_modes = phidesc.get("physics_modes", {})
        gravity_model = (physics_modes.get("gravity_model") or "").upper()
        if gravity_model == "GR_PERTURBATIONS":
            # Return step function that uses GR perturbations
            return self._gr_perturbations_step_generator(phidesc)

        # default fallback: identity map with tiny noise
        def step_default(state: Dict[str, float], t: int, rng: Any) -> Dict[str, float]:
            out = {}
            for k, v in state.items():
                out[k] = float(v + float(rng.normal(0, max(abs(v) * 0.001, 1e-6))))
            return out

        return step_default

    def _cosmo_step_generator(self, phidesc: Dict[str, Any]) -> Any:
        """
        Create a step function implementing a simple FLRW + scalar field + ΛCDM toy model.
        Units: user-specified H0 in s^-1; densities are fractions of critical density.
        phidesc params:
          - H0: Hubble constant in s^-1 (default ~2.2e-18 s^-1 ~ 67.4 km/s/Mpc)
          - Omega_m, Omega_r, Omega_L
          - phi0, phi_dot0, potential: {"type":"quadratic","m":...}
          - dt: timestep in seconds (default 1e13)
        """
        params = phidesc.get("params", {})
        H0 = float(params.get("H0", 2.2e-18))
        G = float(params.get("G", 6.67430e-11))
        Omega_m = float(params.get("Omega_m", 0.315))
        Omega_r = float(params.get("Omega_r", 9e-5))
        Omega_L = float(params.get("Omega_L", 0.685))
        k = float(params.get("k", 0.0))
        dt = float(params.get("dt", 1e13))

        # critical density at present
        rho_crit0 = 3 * H0 * H0 / (8 * math.pi * G)
        rho_m0 = Omega_m * rho_crit0
        rho_r0 = Omega_r * rho_crit0
        rho_L = Omega_L * rho_crit0

        potential = params.get("potential", {"type": "quadratic", "m": 1e-6})
        m = float(potential.get("m", 1e-6))

        def V(phi: float) -> float:
            if potential.get("type", "quadratic") == "quadratic":
                return 0.5 * m * m * phi * phi
            return 0.0

        def dVdphi(phi: float) -> float:
            if potential.get("type", "quadratic") == "quadratic":
                return m * m * phi
            return 0.0

        def step(state: Dict[str, float], t: int, rng: Any) -> Dict[str, float]:
            # Expect state contains: a, phi, phi_dot (if missing use defaults)
            a = float(state.get("a", params.get("a0", 1.0)))
            phi = float(state.get("phi", params.get("phi0", 0.0)))
            phi_dot = float(state.get("phi_dot", params.get("phi_dot0", 0.0)))

            # Use RK45 integration for the coupled system [a, phi, phi_dot]
            def f_vec(x_arr, tt):
                a_loc = float(x_arr[0])
                phi_loc = float(x_arr[1])
                phi_dot_loc = float(x_arr[2])
                # energy densities
                rho_m_loc = rho_m0 / (a_loc ** 3)
                rho_r_loc = rho_r0 / (a_loc ** 4)
                rho_phi_loc = 0.5 * phi_dot_loc * phi_dot_loc + V(phi_loc)
                rho_total_loc = rho_m_loc + rho_r_loc + rho_phi_loc + rho_L
                H_loc = math.sqrt(max(0.0, (8 * math.pi * G / 3.0) * rho_total_loc - (k / (a_loc * a_loc))))
                da_dt = H_loc * a_loc
                dphi_dt = phi_dot_loc
                dphi_dot_dt = -3.0 * H_loc * phi_dot_loc - dVdphi(phi_loc)
                return np.array([da_dt, dphi_dt, dphi_dot_dt], dtype=float)

            x0_vec = np.array([a, phi, phi_dot], dtype=float)
            atol_loc = float(params.get("atol", 1e-8))
            rtol_loc = float(params.get("rtol", 1e-6))
            x_next = self.integrate_rk45(x0_vec, f_vec, 0.0, dt, atol=atol_loc, rtol=rtol_loc)
            a_next, phi_next, phi_dot_next = float(x_next[0]), float(x_next[1]), float(x_next[2])

            # recompute diagnostics at new state
            rho_m_new = rho_m0 / (a_next ** 3) if a_next != 0 else rho_m0
            rho_r_new = rho_r0 / (a_next ** 4) if a_next != 0 else rho_r0
            rho_phi_new = 0.5 * phi_dot_next * phi_dot_next + V(phi_next)
            rho_total_new = rho_m_new + rho_r_new + rho_phi_new + rho_L
            H_new = math.sqrt(max(0.0, (8 * math.pi * G / 3.0) * rho_total_new - (k / (a_next * a_next))))

            # small stochastic fluctuations to mimic quantum uncertainty
            phi_next += float(rng.normal(0, abs(phi_next) * 1e-8 + 1e-16))

            return {"a": float(a_next), "phi": float(phi_next), "phi_dot": float(phi_dot_next), "H": float(H_new), "rho_total": float(rho_total_new)}

        return step

    def _gr_perturbations_step_generator(self, phidesc: Dict[str, Any]) -> Any:
        """
        Create a step function for GR perturbations on FLRW background.
        State should include background (a, phi, phi_dot) and perturbations (delta_rho, delta_phi, h_plus, h_cross).
        """
        params = phidesc.get("params", {})
        dt = float(params.get("dt", 1e13))

        def step(state: Dict[str, float], t: int, rng: Any) -> Dict[str, float]:
            # Extract background
            a = float(state.get("a", 1.0))
            phi = float(state.get("phi", 0.0))
            phi_dot = float(state.get("phi_dot", 0.0))
            x_vec = (a, phi, phi_dot)
            
            # Extract perturbations
            perturbations = {
                "delta_rho": float(state.get("delta_rho", 0.0)),
                "delta_phi": float(state.get("delta_phi", 0.0)),
                "h_plus": float(state.get("h_plus", 0.0)),
                "h_cross": float(state.get("h_cross", 0.0)),
            }
            
            # Evolve with GR perturbations
            try:
                x_next, pert_next, diag = self.evolve_spacetime_gr_perturbations(x_vec, perturbations, params, dt)
                result = {
                    "a": float(x_next[0]),
                    "phi": float(x_next[1]),
                    "phi_dot": float(x_next[2]),
                    "delta_rho": pert_next["delta_rho"],
                    "delta_phi": pert_next["delta_phi"],
                    "h_plus": pert_next["h_plus"],
                    "h_cross": pert_next["h_cross"],
                    "H": diag.get("H", 0.0),
                }
            except Exception:
                # Fallback to FLRW only
                x_next, bg_diag = self.evolve_spacetime_full(x_vec, params, dt)
                result = {
                    "a": float(x_next[0]),
                    "phi": float(x_next[1]),
                    "phi_dot": float(x_next[2]),
                    "delta_rho": perturbations["delta_rho"],
                    "delta_phi": perturbations["delta_phi"],
                    "h_plus": perturbations["h_plus"],
                    "h_cross": perturbations["h_cross"],
                    "H": bg_diag.get("H", 0.0),
                }
            
            return result

        return step

    # Phase 2-A: FLRW RK45 wrapper and Leapfrog integrator
    def evolve_spacetime_full(self, x_vec: Tuple[float, float, float], phidesc_params: Dict[str, Any], dt: float = 1.0) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Evolve [a, phi, phi_dot] forward by dt using RK45 (wrapper around integrate_rk45).
        Returns (x_next_vec, diagnostics).
        """
        if np is None:
            raise RuntimeError("numpy required for evolve_spacetime_full")
        params = phidesc_params or {}
        H0 = float(params.get("H0", 2.2e-18))
        G = float(params.get("G", 6.67430e-11))
        Omega_m = float(params.get("Omega_m", 0.315))
        Omega_r = float(params.get("Omega_r", 9e-5))
        Omega_L = float(params.get("Omega_L", 0.685))
        k = float(params.get("k", 0.0))
        potential = params.get("potential", {"type": "quadratic", "m": 1e-6})
        m = float(potential.get("m", 1e-6))

        def V(phi: float) -> float:
            if potential.get("type", "quadratic") == "quadratic":
                return 0.5 * m * m * phi * phi
            return 0.0

        def dVdphi(phi: float) -> float:
            if potential.get("type", "quadratic") == "quadratic":
                return m * m * phi
            return 0.0

        rho_crit0 = 3 * H0 * H0 / (8 * math.pi * G)
        rho_m0 = Omega_m * rho_crit0
        rho_r0 = Omega_r * rho_crit0
        rho_L = Omega_L * rho_crit0

        def f_vec(x_arr, tt):
            a_loc = float(x_arr[0])
            phi_loc = float(x_arr[1])
            phi_dot_loc = float(x_arr[2])
            rho_m_loc = rho_m0 / (a_loc ** 3) if a_loc != 0 else rho_m0
            rho_r_loc = rho_r0 / (a_loc ** 4) if a_loc != 0 else rho_r0
            rho_phi_loc = 0.5 * phi_dot_loc * phi_dot_loc + V(phi_loc)
            rho_total_loc = rho_m_loc + rho_r_loc + rho_phi_loc + rho_L
            H_loc = math.sqrt(max(0.0, (8 * math.pi * G / 3.0) * rho_total_loc - (k / (a_loc * a_loc)))) if a_loc != 0 else 0.0
            da_dt = H_loc * a_loc
            dphi_dt = phi_dot_loc
            dphi_dot_dt = -3.0 * H_loc * phi_dot_loc - dVdphi(phi_loc)
            return np.array([da_dt, dphi_dt, dphi_dot_dt], dtype=float)

        x0_vec = np.array([float(x_vec[0]), float(x_vec[1]), float(x_vec[2])], dtype=float)
        atol_loc = float(params.get("atol", 1e-8))
        rtol_loc = float(params.get("rtol", 1e-6))
        x_next = self.integrate_rk45(x0_vec, f_vec, 0.0, dt, atol=atol_loc, rtol=rtol_loc)
        a_next, phi_next, phi_dot_next = float(x_next[0]), float(x_next[1]), float(x_next[2])
        rho_m_new = rho_m0 / (a_next ** 3) if a_next != 0 else rho_m0
        rho_r_new = rho_r0 / (a_next ** 4) if a_next != 0 else rho_r0
        rho_phi_new = 0.5 * phi_dot_next * phi_dot_next + V(phi_next)
        rho_total_new = rho_m_new + rho_r_new + rho_phi_new + rho_L
        H_new = math.sqrt(max(0.0, (8 * math.pi * G / 3.0) * rho_total_new - (k / (a_next * a_next)))) if a_next != 0 else 0.0
        diagnostics = {"H": float(H_new), "rho_total": float(rho_total_new)}
        return x_next, diagnostics

    # ================== Phase 7: GR Perturbations ==================
    def evolve_spacetime_gr_perturbations(
        self,
        x_vec: Tuple[float, float, float],
        perturbations: Dict[str, float],
        phidesc_params: Dict[str, Any],
        dt: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
        """
        Evolve FLRW background + linear metric perturbations.
        
        Args:
            x_vec: Background [a, phi, phi_dot]
            perturbations: Dict with keys like 'delta_rho', 'delta_phi', 'h_plus', 'h_cross' (tensor modes)
            phidesc_params: Physics parameters
            dt: Timestep
            
        Returns:
            (x_next_vec, perturbations_next, diagnostics)
        """
        if np is None:
            raise RuntimeError("numpy required for GR perturbations")
        
        # Evolve background first
        x_next, bg_diag = self.evolve_spacetime_full(x_vec, phidesc_params, dt)
        a_next = float(x_next[0])
        H_next = bg_diag.get("H", 0.0)
        
        params = phidesc_params or {}
        H0 = float(params.get("H0", 2.2e-18))
        G = float(params.get("G", 6.67430e-11))
        c = float(params.get("c", 2.99792458e8))  # speed of light
        
        # Extract perturbation amplitudes
        delta_rho = float(perturbations.get("delta_rho", 0.0))
        delta_phi = float(perturbations.get("delta_phi", 0.0))
        h_plus = float(perturbations.get("h_plus", 0.0))  # tensor mode
        h_cross = float(perturbations.get("h_cross", 0.0))  # tensor mode
        
        # Scalar perturbations: density fluctuations evolve via continuity + Poisson
        # Simplified: δρ' = -3H δρ - k^2 v (for sub-horizon modes)
        # For super-horizon: δρ' ≈ -3H δρ (conservation)
        k_wavenumber = float(params.get("k_perturbation", 1e-3))  # comoving wavenumber
        H_a = H_next * a_next if a_next > 0 else H0
        
        # Super-horizon evolution (k << aH)
        if k_wavenumber < 0.1 * H_a:
            # Conservation mode: δρ/ρ constant
            delta_rho_next = delta_rho
        else:
            # Sub-horizon: decay due to expansion
            delta_rho_next = delta_rho * math.exp(-3.0 * H_next * dt)
        
        # Scalar field perturbation: δφ' ≈ -3H δφ (simplified)
        delta_phi_next = delta_phi * math.exp(-3.0 * H_next * dt)
        
        # Tensor perturbations (gravitational waves): h'' + 3H h' + (k/a)^2 h = 0
        # For super-horizon: h constant
        # For sub-horizon: h decays as 1/a
        if k_wavenumber < 0.1 * H_a:
            h_plus_next = h_plus
            h_cross_next = h_cross
        else:
            # Sub-horizon: oscillate and decay
            omega_gw = k_wavenumber / a_next if a_next > 0 else k_wavenumber
            decay_factor = math.exp(-1.5 * H_next * dt)
            h_plus_next = h_plus * decay_factor * math.cos(omega_gw * dt)
            h_cross_next = h_cross * decay_factor * math.sin(omega_gw * dt)
        
        perturbations_next = {
            "delta_rho": float(delta_rho_next),
            "delta_phi": float(delta_phi_next),
            "h_plus": float(h_plus_next),
            "h_cross": float(h_cross_next),
        }
        
        diagnostics = {
            "H": float(H_next),
            "k_over_aH": float(k_wavenumber / H_a if H_a > 0 else 0.0),
            "perturbation_scale": "super_horizon" if k_wavenumber < 0.1 * H_a else "sub_horizon",
        }
        
        return x_next, perturbations_next, diagnostics

    def compute_quantum_backreaction(
        self,
        field_adapter: Any,
        field_modes: List[float],
        phidesc_params: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compute quantum stress-energy tensor from field fluctuations and return correction to Einstein equations.
        
        Args:
            field_adapter: QFTFieldAdapter instance
            field_modes: List of field mode amplitudes
            phidesc_params: Physics parameters
            
        Returns:
            Dict with 'delta_rho' and 'delta_p' corrections to energy density and pressure
        """
        if not hasattr(field_adapter, "compute_stress_energy"):
            return {"delta_rho": 0.0, "delta_p": 0.0}
        
        try:
            stress_energy = field_adapter.compute_stress_energy(field_modes)
            # Convert to density correction (normalized by critical density)
            G = float(phidesc_params.get("G", 6.67430e-11))
            H0 = float(phidesc_params.get("H0", 2.2e-18))
            rho_crit = 3 * H0 * H0 / (8 * math.pi * G)
            
            # Quantum correction to energy density
            delta_rho = stress_energy.get("rho", 0.0) / rho_crit if rho_crit > 0 else 0.0
            delta_p = stress_energy.get("p", 0.0) / rho_crit if rho_crit > 0 else 0.0
            
            return {"delta_rho": float(delta_rho), "delta_p": float(delta_p)}
        except Exception:
            return {"delta_rho": 0.0, "delta_p": 0.0}

    def leapfrog_step(self, x: np.ndarray, p: np.ndarray, force_fn: Any, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple leapfrog integrator for position x and momentum p with force function force_fn(x) -> a.
        Returns (x_next, p_next).
        """
        # half step momentum
        a = force_fn(x)
        p_half = p + 0.5 * dt * a
        # full position step
        x_next = x + dt * p_half
        # full acceleration at new position
        a_next = force_fn(x_next)
        p_next = p_half + 0.5 * dt * a_next
        return x_next, p_next

    # ================== Phase 3: Adapter API for PDE / Surrogates ==================
    class PDEAdapterCpu:
        """Minimal CPU PDE adapter implementing a 1D diffusion stencil."""
        def __init__(self, params: Optional[Dict[str, Any]] = None):
            self.params = params or {}
            self.alpha = float(self.params.get("alpha", 1e-3))

        def step(self, local_grid: List[float], dt: float, params: Dict[str, Any]) -> List[float]:
            # simple explicit diffusion: u_t = alpha * u_xx, 1D periodic boundary
            if not local_grid:
                return []
            N = len(local_grid)
            arr = np.array(local_grid, dtype=float)
            alpha = float(params.get("alpha", self.alpha))
            # discrete Laplacian (periodic)
            lap = np.roll(arr, -1) - 2 * arr + np.roll(arr, 1)
            out = arr + dt * alpha * lap
            return out.tolist()

    class PDEAdapter2DCpu:
        """CPU 2D PDE adapter - explicit diffusion with simple BCs."""
        def __init__(self, params: Optional[Dict[str, Any]] = None):
            self.params = params or {}
            self.alpha = float(self.params.get("alpha", 1e-3))
            self.bc = self.params.get("bc", "neumann")

        def step(self, local_grid: Any, dt: float, params: Dict[str, Any]) -> Any:
            import numpy as _np
            arr = _np.asarray(local_grid, dtype=float)
            if arr.ndim != 2:
                raise ValueError("PDEAdapter2DCpu expects 2D grid")
            alpha = float(params.get("alpha", self.alpha))
            # compute Laplacian with Neumann (zero-gradient) by padding
            if self.bc == "neumann":
                padded = _np.pad(arr, pad_width=1, mode="edge")
            else:
                padded = _np.pad(arr, pad_width=1, mode="constant", constant_values=0.0)
            lap = padded[2:,1:-1] + padded[0:-2,1:-1] + padded[1:-1,2:] + padded[1:-1,0:-2] - 4 * padded[1:-1,1:-1]
            out = arr + dt * alpha * lap
            return out

    class SurrogateAdapterDummy:
        """Dummy surrogate performing a linear transform as placeholder."""
        def __init__(self, params: Optional[Dict[str, Any]] = None):
            self.params = params or {}
            self.scale = float(self.params.get("scale", 0.99))

        def predict(self, local_state: List[float], dt: float) -> List[float]:
            arr = np.array(local_state, dtype=float)
            return (arr * self.scale).tolist()

    # ================== Phase 7: QFT Field Adapter ==================
    class QFTFieldAdapter:
        """
        Minimal QFT implementation for scalar field with mode expansion.
        Computes vacuum stress-energy tensor from field fluctuations.
        """
        def __init__(self, params: Optional[Dict[str, Any]] = None):
            self.params = params or {}
            self.m = float(self.params.get("m", 1e-6))  # field mass
            self.n_modes = int(self.params.get("n_modes", 8))  # number of modes
            self.cutoff = float(self.params.get("cutoff", 1e-2))  # UV cutoff
            # Initialize mode amplitudes (vacuum state)
            self.mode_amplitudes = [0.0] * self.n_modes
            self.mode_phases = [0.0] * self.n_modes
            # Zero-point energy per mode: ω_k / 2
            self.zero_point_energies = []
            for k in range(self.n_modes):
                k_val = float(k + 1) * self.cutoff / self.n_modes
                omega_k = math.sqrt(k_val * k_val + self.m * self.m)
                self.zero_point_energies.append(0.5 * omega_k)

        def step(self, local_grid: List[float], dt: float, params: Dict[str, Any]) -> List[float]:
            """
            Evolve field modes forward by dt.
            local_grid represents field values at different spatial points (or mode amplitudes).
            """
            if not local_grid:
                return []
            
            # If local_grid is mode amplitudes, evolve directly
            if len(local_grid) == self.n_modes:
                modes = np.array(local_grid, dtype=float)
            else:
                # Interpret as spatial field values, convert to modes (simplified)
                modes = np.array(local_grid[:self.n_modes], dtype=float)
            
            # Evolve each mode: φ_k'' + ω_k² φ_k = 0 (harmonic oscillator)
            modes_next = []
            for k, phi_k in enumerate(modes):
                k_val = float(k + 1) * self.cutoff / self.n_modes
                omega_k = math.sqrt(k_val * k_val + self.m * self.m)
                # Simple Euler step for mode evolution
                # For proper evolution, would use creation/annihilation operators
                # Simplified: φ_k(t+dt) ≈ φ_k(t) * cos(ω_k dt)
                phi_next = phi_k * math.cos(omega_k * dt)
                modes_next.append(float(phi_next))
            
            return modes_next[:len(local_grid)]

        def compute_stress_energy(self, modes: List[float]) -> Dict[str, float]:
            """
            Compute vacuum expectation value of stress-energy tensor ⟨T_μν⟩.
            Returns dict with 'rho' (energy density) and 'p' (pressure).
            """
            if not modes:
                return {"rho": 0.0, "p": 0.0}
            
            # Sum over modes: ⟨T_00⟩ = Σ_k (ω_k / 2) for vacuum
            # For excited modes, add mode energy: ω_k * |φ_k|²
            rho = 0.0
            p = 0.0
            
            for k, phi_k in enumerate(modes[:self.n_modes]):
                k_val = float(k + 1) * self.cutoff / self.n_modes
                omega_k = math.sqrt(k_val * k_val + self.m * self.m)
                # Vacuum energy
                rho += self.zero_point_energies[k] if k < len(self.zero_point_energies) else 0.5 * omega_k
                # Excitation energy
                rho += omega_k * float(phi_k * phi_k)
                # Pressure: p = ρ/3 for relativistic modes (simplified)
                if omega_k > self.m:
                    p += (omega_k * float(phi_k * phi_k)) / 3.0
            
            return {"rho": float(rho), "p": float(p)}

    def get_physics_adapter(self, phidesc: Dict[str, Any]) -> Optional[Any]:
        """
        Factory returning an adapter instance based on phidesc['field_model'].
        Supported: 'pde_cpu', 'surrogate_dummy'
        """
        if not phidesc:
            return None
        fm = (phidesc.get("field_model") or "").lower()
        params = phidesc.get("params", {}) or {}
        if fm in ("pde_cpu", "pde_1d_cpu"):
            return self.PDEAdapterCpu(params)
        if fm in ("pde_2d_cpu", "pde2d_cpu"):
            return self.PDEAdapter2DCpu(params)
        if fm in ("pde_gpu", "pde_2d_gpu"):
            # try to import GPU adapter module
            try:
                import importlib
                mod = None
                for candidate in ("examples.multi_agent.simulations.time_engine.adapters.pde_gpu", "examples.multi_agent.simulations.time_engine.adapters.pde_gpu", "adapters.pde_gpu"):
                    try:
                        mod = importlib.import_module(candidate)
                        break
                    except Exception:
                        continue
                if mod is None:
                    raise RuntimeError("PDE GPU adapter requested but GPU adapters module not found.")
                AdapterCls = getattr(mod, "PDEAdapterGpu", None)
                if AdapterCls is None:
                    raise RuntimeError("PDE GPU adapter class not found in module.")
                return AdapterCls(params)
            except Exception as e:
                raise RuntimeError("PDE GPU adapter requested but unavailable: %s" % (e,))
        if fm in ("surrogate_dummy", "surrogate_adapter", "surrogate"):
            return self.SurrogateAdapterDummy(params)
        if fm in ("qft_scalar", "qft_field"):
            return self.QFTFieldAdapter(params)
        # not supported: return None to fall back
        return None

    # ================== ASTT Phase-1: minimal stubs and utilities ==================
    def initialize_state(self, phidesc: Dict[str, Any], seed_config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Build an initial macro state x0 from a phidesc and seed_config.
        Deterministic when seed_config['noise_seed'] provided.
        """
        cfg = seed_config or phidesc.get("seed_config", {}) or {}
        t_init = float(cfg.get("t_init", 1e-36))
        a0 = float(cfg.get("a_0", 1.0))
        phi0 = float(cfg.get("phi_0", 0.0))
        rho0 = float(cfg.get("rho_0", 1.0))
        noise_seed = cfg.get("noise_seed", None)
        rng = np.random.default_rng(int(noise_seed)) if (np is not None and noise_seed is not None) else (np.random.default_rng() if np is not None else None)
        state: Dict[str, float] = {}
        # include cosmology defaults
        state["t"] = t_init
        state["a"] = a0
        state["phi"] = phi0
        state["phi_dot"] = float(cfg.get("phi_dot0", 0.0))
        state["rho"] = rho0
        # include default universe variables if present
        dcfg = self.get_default_universe_config()
        for k, v in dcfg.get("initial", {}).items():
            state.setdefault(k, float(v))
        
        # Phase 7: Quantum vacuum initialization
        if cfg.get("quantum_vacuum", False) and rng is not None:
            # Initialize field modes in vacuum state with zero-point fluctuations
            n_modes = int(cfg.get("n_qft_modes", 8))
            m = float(cfg.get("qft_mass", 1e-6))
            cutoff = float(cfg.get("qft_cutoff", 1e-2))
            
            for k in range(n_modes):
                k_val = float(k + 1) * cutoff / n_modes
                omega_k = math.sqrt(k_val * k_val + m * m) if np is not None else k_val
                # Vacuum fluctuation: Gaussian with variance 1/(2ω_k)
                variance = 1.0 / (2.0 * omega_k) if omega_k > 0 else 1e-12
                mode_amplitude = float(rng.normal(0, math.sqrt(variance)))
                state[f"phi_mode_{k}"] = float(mode_amplitude)
        
        # Phase 7: Cosmological perturbation seeds
        if cfg.get("perturbation_seed", None) is not None and rng is not None:
            # Generate initial power spectrum for density fluctuations
            # Scale-invariant spectrum: P(k) ∝ k^(n_s - 1) with n_s ≈ 1
            n_s = float(cfg.get("spectral_index", 0.96))  # nearly scale-invariant
            amplitude = float(cfg.get("perturbation_amplitude", 1e-5))
            
            # Generate Gaussian random field for density perturbation
            # Simplified: single mode amplitude
            k_pert = float(cfg.get("k_perturbation", 1e-3))
            power = amplitude * (k_pert ** (n_s - 1.0))
            state["delta_rho"] = float(rng.normal(0, math.sqrt(power)))
            state["delta_phi"] = float(rng.normal(0, math.sqrt(power * 0.1)))  # smaller for field
            
            # Tensor modes (gravitational waves)
            state["h_plus"] = float(rng.normal(0, math.sqrt(power * 0.01)))  # much smaller
            state["h_cross"] = float(rng.normal(0, math.sqrt(power * 0.01)))
        
        # small stochastic quantum/thermal noise if requested
        if rng is not None:
            for k in list(state.keys()):
                if not k.startswith("phi_mode_"):  # don't add noise to already initialized modes
                    state[k] = float(state[k] + float(rng.normal(0, max(abs(state[k]) * 1e-8, 1e-12))))
        return state

    def apply_boundary(self, state: Dict[str, float], boundary_model: Optional[Dict[str, Any]], t: int = 0) -> Dict[str, float]:
        """
        Apply boundary model to a macro state in-place and return it.
        Supported types: 'causal_sampling', 'reflective', 'absorptive'
        """
        if not boundary_model:
            return state
        typ = boundary_model.get("type", "causal_sampling")
        params = boundary_model.get("params", {})
        vars_to_affect = params.get("vars", list(state.keys()))
        sigma = float(params.get("sigma", 1e-6))
        rng = np.random.default_rng(int(params.get("seed", time.time() % (2 ** 31)))) if np is not None else None
        if typ == "causal_sampling":
            for v in vars_to_affect:
                add = float(rng.normal(0, sigma)) if rng is not None else 0.0
                state[v] = float(state.get(v, 0.0) + add)
        elif typ == "reflective":
            # simple clamp: keep values within [min_val, max_val] if provided
            minv = params.get("min", None)
            maxv = params.get("max", None)
            for v in vars_to_affect:
                val = state.get(v, 0.0)
                if minv is not None:
                    val = max(val, float(minv))
                if maxv is not None:
                    val = min(val, float(maxv))
                state[v] = float(val)
        elif typ == "absorptive":
            decay = float(params.get("decay", 0.01))
            for v in vars_to_affect:
                state[v] = float(state.get(v, 0.0) * (1.0 - decay))
        return state

    def coarse_grain(self, micro_state: Any, resolution_config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Minimal coarse-graining: accepts a list of micro states (dicts) and returns averaged macro variables.
        """
        out: Dict[str, float] = {}
        if not micro_state:
            return out
        if isinstance(micro_state, dict):
            # already macro
            return {k: float(v) for k, v in micro_state.items()}
        # assume iterable of dicts
        items = list(micro_state)
        keys = set()
        for m in items:
            if isinstance(m, dict):
                keys.update(m.keys())
        for k in keys:
            vals = []
            for m in items:
                if isinstance(m, dict):
                    vals.append(float(m.get(k, 0.0)))
            out[k] = float(np.mean(vals)) if (np is not None and vals) else (sum(vals) / max(1, len(vals)) if vals else 0.0)
        return out

    def evolve_state(self, state: Dict[str, float], dt: float = 1.0, method: str = "RK45", phidesc: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Dispatcher for time integration. Phase-1: simple linear ODE model x' = -lambda * x by default.
        If method == 'RK45' uses integrate_rk45; if 'Leapfrog' uses a simple leapfrog-like update.
        """
        # Adapter hook: if a field_model adapter is requested, use it
        if phidesc:
            field_model = (phidesc.get("field_model") or "").lower()
            if field_model:
                adapter = self.get_physics_adapter(phidesc)
                if adapter is not None:
                    # Phase 7: For QFT adapters, extract field modes from state
                    if field_model in ("qft_scalar", "qft_field"):
                        # Extract phi_mode_* keys from state
                        field_modes = []
                        mode_keys = sorted([k for k in state.keys() if k.startswith("phi_mode_")])
                        if mode_keys:
                            for mk in mode_keys:
                                field_modes.append(float(state.get(mk, 0.0)))
                        else:
                            # Fallback: use field_variables if no modes found
                            res_cfg = phidesc.get("resolution_config", {}) or {}
                            vars_list = res_cfg.get("field_variables", list(state.keys()))
                            field_modes = [float(state.get(v, 0.0)) for v in vars_list]
                        grid = field_modes
                    else:
                        # Build a simple local array from requested field_variables (if any)
                        res_cfg = phidesc.get("resolution_config", {}) or {}
                        vars_list = res_cfg.get("field_variables", list(state.keys()))
                        grid = [float(state.get(v, 0.0)) for v in vars_list]
                    # adapter.step or predict may accept (local_grid, dt, params)
                    try:
                        t0 = time.perf_counter()
                        if hasattr(adapter, "step"):
                            out_grid = adapter.step(grid, dt, phidesc.get("params", {}))
                        else:
                            out_grid = adapter.predict(grid, dt)
                        t1 = time.perf_counter()
                        # store last adapter duration in ms for diagnostics
                        self._last_adapter_ms = (t1 - t0) * 1000.0
                        # map back
                        out_state = dict(state)
                        if field_model in ("qft_scalar", "qft_field") and mode_keys:
                            # Map back to phi_mode_* keys
                            for i, mk in enumerate(mode_keys):
                                if i < len(out_grid):
                                    out_state[mk] = float(out_grid[i])
                        else:
                            # Map back to field_variables
                            res_cfg = phidesc.get("resolution_config", {}) or {}
                            vars_list = res_cfg.get("field_variables", list(state.keys()))
                            for i, v in enumerate(vars_list):
                                out_state[v] = float(out_grid[i] if i < len(out_grid) else out_state.get(v, 0.0))
                        return out_state
                    except Exception:
                        # fallback to existing evolution below
                        pass
        if np is None:
            # fall back to Euler step
            out = {k: float(v + dt * (-1e-3 * v)) for k, v in state.items()}
            return out
        # state vector ordering from default universe config
        var_list = list(self.get_default_universe_config().get("variables", list(state.keys())))
        x0 = np.array([float(state.get(v, 0.0)) for v in var_list], dtype=float)
        # define derivative function: simple relaxation with optional coupling from phidesc.params.alpha
        params = (phidesc or {}).get("params", {}) if phidesc else {}
        lam = float(params.get("lambda", 1e-3))
        def f_vec(x, tt):
            return -lam * x
        if method.upper() == "RK45":
            try:
                x_next = self.integrate_rk45(x0, f_vec, 0.0, dt, atol=float(params.get("atol", 1e-8)), rtol=float(params.get("rtol", 1e-6)))
            except Exception:
                # fallback to explicit Euler
                x_next = x0 + dt * f_vec(x0, 0.0)
        elif method.upper() == "LEAPFROG" or method.upper() == "LEAP":
            # Use leapfrog for second-order like systems. Map x->position, use simple momentum = -lambda*x
            # Build position and momentum vectors
            p0 = -1.0 * x0  # synthetic momentum for demo; in a real model this would come from state
            def force_fn(x_arr):
                return f_vec(x_arr, 0.0)
            try:
                x_next_arr, p_next_arr = self.leapfrog_step(x0, p0, force_fn, dt)
                x_next = x_next_arr
            except Exception:
                x_next = x0 + dt * f_vec(x0, 0.0)
        else:
            # Unknown method fallback: explicit Euler
            x_next = x0 + dt * f_vec(x0, 0.0)
        out = {v: float(x_next[i]) for i, v in enumerate(var_list)}
        # keep any other state keys unchanged
        for k in state.keys():
            if k not in out:
                out[k] = float(state[k])
        return out

    # ================== Phase 2-B: Tiles, OU boundary, multiscale run loop ==================
    def _ou_step(self, x_prev: float, theta: float, mu: float, sigma: float, dt: float, rng: Any) -> float:
        """Ornstein-Uhlenbeck step for correlated boundary process."""
        if rng is None:
            return x_prev
        dx = theta * (mu - x_prev) * dt + sigma * math.sqrt(max(dt, 1e-12)) * float(rng.normal(0, 1.0))
        return float(x_prev + dx)

    def run_multiscale_simulation(self, seed_state: Dict[str, float], horizon: int = 24, phidesc: Optional[Dict[str, Any]] = None, ensemble: int = 1, rng_seed: Optional[int] = None, dt: float = 1.0, progress_cb: Optional[Any] = None) -> Dict[str, Any]:
        """
        Lightweight multiscale simulation: create logical tiles and run a tiled timestep loop.
        Returns summary including per-tile diagnostics stored under 'tiles'.
        """
        rng = np.random.default_rng(rng_seed if rng_seed is not None else int(time.time() % (2 ** 31))) if np is not None else None
        ph = phidesc or {}
        tiles_cfg = ph.get("tiles", {}) or {}
        n_tiles = int(tiles_cfg.get("n_tiles", 4))
        tile_states = []
        for i in range(n_tiles):
            # initialize tile from seed_state with small perturbation
            ts = dict(seed_state)
            if rng is not None:
                for k in list(ts.keys()):
                    ts[k] = float(ts[k] + float(rng.normal(0, max(abs(ts[k]) * 1e-6, 1e-12))))
            tile_states.append({"id": i, "state": ts, "boundary": 0.0})

        # global spacetime state from seed_config or seed_state
        seed_cfg = ph.get("seed_config", {})
        global_state = {"a": seed_cfg.get("a_0", seed_state.get("a", 1.0)), "phi": seed_cfg.get("phi_0", seed_state.get("phi", 0.0)), "phi_dot": seed_cfg.get("phi_dot0", 0.0)}

        tile_history = {i: [] for i in range(n_tiles)}
        # boundary OU params
        bparams = (ph.get("boundary_model") or {}).get("params", {}) if ph.get("boundary_model") else {}
        theta = float(bparams.get("theta", 0.1))
        mu = float(bparams.get("mu", 0.0))
        sigma = float(bparams.get("sigma", 1e-6))

        # nested controller option
        if ph.get("controller_mode", "").lower() == "nested":
            return self.nested_multiscale_controller(tile_states, global_state, ph, rng, horizon, dt, progress_cb)

        # Phase 7: Check for quantum backreaction
        quantum_backreaction_enabled = ph.get("physics_modes", {}).get("quantum_backreaction", False)
        qft_adapter = None
        if quantum_backreaction_enabled:
            qft_adapter = self.get_physics_adapter(ph)

        for t in range(int(horizon)):
            # Phase 7: Compute quantum backreaction from tiles if enabled
            quantum_correction = {"delta_rho": 0.0, "delta_p": 0.0}
            if quantum_backreaction_enabled and qft_adapter is not None:
                # Aggregate quantum stress-energy from all tiles
                total_delta_rho = 0.0
                total_delta_p = 0.0
                for tile in tile_states:
                    # Extract field modes from tile state (if present)
                    field_modes = []
                    for k in range(8):  # default n_modes
                        mode_key = f"phi_mode_{k}"
                        if mode_key in tile["state"]:
                            field_modes.append(float(tile["state"][mode_key]))
                    if field_modes:
                        correction = self.compute_quantum_backreaction(qft_adapter, field_modes, ph.get("params", {}))
                        total_delta_rho += correction.get("delta_rho", 0.0)
                        total_delta_p += correction.get("delta_p", 0.0)
                # Average over tiles
                if n_tiles > 0:
                    quantum_correction = {
                        "delta_rho": float(total_delta_rho / n_tiles),
                        "delta_p": float(total_delta_p / n_tiles),
                    }
            
            # global update using evolve_spacetime_full if cosmology requested
            if ph.get("physics_modes", {}).get("gravity_model", "").upper() in ("FLRW", "GR_PERTURBATIONS"):
                x_vec = (float(global_state.get("a", 1.0)), float(global_state.get("phi", 0.0)), float(global_state.get("phi_dot", 0.0)))
                try:
                    # Apply quantum correction to energy density if enabled
                    params_corrected = dict(ph.get("params", {}))
                    if quantum_backreaction_enabled and quantum_correction["delta_rho"] != 0.0:
                        # Add quantum correction to Omega_L (dark energy) as approximation
                        params_corrected["Omega_L"] = float(params_corrected.get("Omega_L", 0.685)) + quantum_correction["delta_rho"]
                    
                    x_next_vec, diag = self.evolve_spacetime_full(x_vec, params_corrected, dt)
                    global_state["a"], global_state["phi"], global_state["phi_dot"] = float(x_next_vec[0]), float(x_next_vec[1]), float(x_next_vec[2])
                except Exception:
                    # fallback small drift
                    global_state["a"] = float(global_state.get("a", 1.0) * (1.0 + 1e-3 * dt))
            # per-tile updates
            for tile in tile_states:
                s = tile["state"]
                # Phase 7: Scale-dependent physics switching
                coherence_length = float(ph.get("physics_modes", {}).get("quantum_coherence_length", 1e-35))
                tile_scale = float(ph.get("tiles", {}).get("tile_scale", 1e-3))  # approximate tile scale in meters
                use_quantum = tile_scale < coherence_length
                
                # apply boundary correlated noise
                if ph.get("boundary_model", {}).get("type", "") == "causal_sampling":
                    prev = tile.get("boundary", 0.0)
                    newb = self._ou_step(prev, theta, mu, sigma, dt, rng)
                    tile["boundary"] = newb
                    # inject into selected vars
                    bvars = bparams.get("vars", list(s.keys()))
                    for bv in bvars:
                        s[bv] = float(s.get(bv, 0.0) + newb)
                
                # evolve local tile state using evolve_state
                method = (ph.get("evolution_solver") or {}).get("time_integrator", "RK45")
                # Phase 7: Switch to quantum field model if scale is small enough
                ph_local = dict(ph)
                if use_quantum and not ph_local.get("field_model"):
                    ph_local["field_model"] = "qft_scalar"
                try:
                    s_next = self.evolve_state(s, dt=dt, method=method, phidesc=ph_local)
                    tile["state"] = s_next
                except Exception:
                    # fallback small drift
                    for k in s.keys():
                        s[k] = float(s.get(k, 0.0) * (1.0 + 1e-3 * dt))
                # collect diagnostics
                diag = {"t": t, "mean": {k: float(v) for k, v in tile["state"].items()}, "adapter_ms": float(getattr(self, "_last_adapter_ms", 0.0))}
                tile_history[tile["id"]].append(diag)
                # push to progress callback if provided
                if progress_cb:
                    try:
                        payload = {"job_id": ph.get("_job_id") if ph else None, "tile_id": tile["id"], "t": t, "diagnostics": diag}
                        progress_cb(payload)
                    except Exception:
                        pass

        # assemble tile summaries
        tiles_out = []
        for tid, hist in tile_history.items():
            tiles_out.append({"id": tid, "steps": len(hist), "last": hist[-1] if hist else {}})

        # aggregate macro stats (use existing aggregate calculation)
        aggregate = {"tiles": len(tiles_out)}
        result = {"seed": seed_state, "horizon": horizon, "ensemble": 1, "tiles": tiles_out, "aggregate": aggregate, "created_ts": time.time()}
        sim_id = self._store_simulation_result("multiscale_run", seed_state, ph or {}, {"horizon": horizon}, result)
        return {"sim_id": sim_id, "summary": aggregate, "tiles": tiles_out}

    # ================== Laws registry ==================
    def register_law(self, name: str, phidesc: Dict[str, Any], description: str = "") -> int:
        """Register a phidesc (law/model) in the DB and return id. Delegates to extended registration when available."""
        if hasattr(self, "register_law_extended"):
            return self.register_law_extended(name, phidesc, description=description)
        cur = self._conn.cursor()
        cur.execute("INSERT INTO laws(name, description, phidesc_json, created_ts) VALUES (?, ?, ?, ?)", (name, description, json.dumps(phidesc), time.time()))
        lid = cur.lastrowid
        self._conn.commit()
        return lid

    def list_laws(self) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT id, name, description, phidesc_json, created_ts FROM laws ORDER BY created_ts DESC")
        out = []
        for lid, name, desc, pj, ts in cur.fetchall():
            try:
                p = json.loads(pj) if pj else {}
            except Exception:
                p = {}
            out.append({"id": lid, "name": name, "description": desc, "phidesc": p, "created_ts": float(ts)})
        return out

    def get_law(self, law_id: int) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT id, name, description, phidesc_json, created_ts FROM laws WHERE id = ?", (law_id,))
        row = cur.fetchone()
        if not row:
            return None
        lid, name, desc, pj, ts = row
        try:
            p = json.loads(pj) if pj else {}
        except Exception:
            p = {}
        return {"id": lid, "name": name, "description": desc, "phidesc": p, "created_ts": float(ts)}

    # ================== numerical integrators (adaptive RK4 via Richardson) ==================
    def rk4_step(self, x: np.ndarray, f: Any, t: float, dt: float) -> np.ndarray:
        """Single RK4 step for vector x with callable f(x,t)->dxdt."""
        k1 = f(x, t)
        k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(x + dt * k3, t + dt)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def adaptive_rk4_integrate(self, x0: np.ndarray, f: Any, t0: float, t1: float, atol: float = 1e-6, rtol: float = 1e-3, max_iter: int = 1000) -> np.ndarray:
        """
        Integrate from t0 to t1 with adaptive RK4 using Richardson extrapolation:
        compare full step vs two half steps to estimate error and adapt dt.
        """
        dt = t1 - t0
        x = x0.copy()
        t = t0
        iters = 0
        while t < t1 - 1e-12 and iters < max_iter:
            iters += 1
            h = min(dt, t1 - t)
            # one full step
            x_full = self.rk4_step(x, f, t, h)
            # two half steps
            x_half = self.rk4_step(x, f, t, h / 2.0)
            x_half = self.rk4_step(x_half, f, t + h / 2.0, h / 2.0)
            # estimate error
            err = np.linalg.norm(x_full - x_half) if np is not None else 0.0
            tol = atol + rtol * max(np.linalg.norm(x_full), np.linalg.norm(x_half)) if np is not None else atol
            if err <= tol or h <= 1e-8:
                # accept half-step estimate (higher order)
                x = x_half
                t += h
                # increase step for next iter
                dt = min(dt * 2.0, t1 - t)
            else:
                # reduce step
                dt = max(h / 2.0, 1e-8)
        return x

    def integrate_rk45(self, x0: np.ndarray, f: Any, t0: float, t1: float, atol: float = 1e-6, rtol: float = 1e-3, max_steps: int = 10000) -> np.ndarray:
        """
        Integrate using Dormand-Prince RK45 with adaptive step control from t0 to t1.
        f(x,t) -> dx/dt
        """
        if np is None:
            raise RuntimeError("numpy required for RK45")
        # Dormand-Prince coefficients
        c2 = 1/5
        c3 = 3/10
        c4 = 4/5
        c5 = 8/9
        c6 = 1.0
        c7 = 1.0

        a21 = 1/5
        a31 = 3/40; a32 = 9/40
        a41 = 44/45; a42 = -56/15; a43 = 32/9
        a51 = 19372/6561; a52 = -25360/2187; a53 = 64448/6561; a54 = -212/729
        a61 = 9017/3168; a62 = -355/33; a63 = 46732/5247; a64 = 49/176; a65 = -5103/18656
        a71 = 35/384; a72 = 0; a73 = 500/1113; a74 = 125/192; a75 = -2187/6784; a76 = 11/84

        # error coefficients
        e1 = 71/57600; e3 = -71/16695; e4 = 71/1920; e5 = -17253/339200; e6 = 22/525; e7 = -1/40

        t = t0
        x = x0.astype(float).copy()
        dt = (t1 - t0) / 10.0 if (t1 - t0) > 0 else 1.0
        steps = 0
        while t < t1 and steps < max_steps:
            steps += 1
            if t + dt > t1:
                dt = t1 - t
            k1 = f(x, t)
            k2 = f(x + dt * a21 * k1, t + c2 * dt)
            k3 = f(x + dt * (a31 * k1 + a32 * k2), t + c3 * dt)
            k4 = f(x + dt * (a41 * k1 + a42 * k2 + a43 * k3), t + c4 * dt)
            k5 = f(x + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), t + c5 * dt)
            k6 = f(x + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), t + c6 * dt)
            x_next = x + dt * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
            # 7th stage for error estimate
            k7 = f(x_next, t + dt)
            # error estimate
            err = dt * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7)
            err_norm = float(np.linalg.norm(err))
            x_norm = float(np.linalg.norm(x_next))
            tol = atol + rtol * max(x_norm, 1.0)
            if err_norm <= tol:
                # accept
                x = x_next
                t += dt
                # increase dt
                if err_norm == 0:
                    s = 2.0
                else:
                    s = 0.9 * (tol / err_norm) ** 0.2
                dt = min(dt * max(0.1, s), t1 - t)
            else:
                # reduce dt
                s = 0.9 * (tol / (err_norm + 1e-30)) ** 0.25
                dt = max(dt * max(0.1, s), 1e-16)
        return x

    # ================== DTW normalization utility ==================
    def normalize_series(self, series: List[float]) -> List[float]:
        """Z-score normalize a series; if constant, return zeros."""
        if np is None:
            return series
        arr = np.asarray(series, dtype=float)
        if arr.size == 0:
            return []
        mean = arr.mean()
        std = arr.std()
        if std < 1e-12:
            return (arr - mean).tolist()
        return ((arr - mean) / std).tolist()

    def extract_pattern_spectral(self, series: List[float]) -> Dict[str, Any]:
        """Compute a compact spectral fingerprint (magnitude of FFT) and return as proto."""
        if np is None:
            return {"error": "numpy required"}
        arr = np.asarray(series, dtype=float)
        N = len(arr)
        if N < 4:
            return {"error": "series too short"}
        freqs = np.fft.rfft(arr - arr.mean())
        mags = np.abs(freqs)
        proto = {"N": N, "top_magnitudes": mags[: min(10, len(mags))].tolist()}
        # store proto compressed
        blob = _compress_bytes(json.dumps(proto).encode("utf-8"))
        h = _sha256_hex(blob)
        cur = self._conn.cursor()
        cur.execute("INSERT INTO patterns(sim_id, name, proto_blob, created_ts) VALUES (?, ?, ?, ?)", (None, "spectral_proto", blob, time.time()))
        self._conn.commit()
        return proto

    # ================== DTW & pattern matching ==================
    def dtw_distance(self, a: List[float], b: List[float]) -> float:
        """Compute Dynamic Time Warping distance (classic DP implementation)."""
        if np is None:
            raise RuntimeError("numpy required for DTW")
        na = len(a)
        nb = len(b)
        if na == 0 or nb == 0:
            return float('inf')
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        dp = np.full((na + 1, nb + 1), np.inf)
        dp[0, 0] = 0.0
        for i in range(1, na + 1):
            for j in range(1, nb + 1):
                cost = abs(a[i - 1] - b[j - 1])
                dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        return float(dp[na, nb])

    def find_similar_patterns(self, series: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Compare a series against stored patterns (proto blobs) using DTW where possible."""
        cur = self._conn.cursor()
        cur.execute("SELECT id, name, proto_blob FROM patterns")
        candidates = []
        for pid, name, blob in cur.fetchall():
            try:
                # attempt to decompress and parse proto (we stored compressed proto in patterns earlier)
                data = zlib.decompress(blob)
                proto = json.loads(data.decode("utf-8"))
                # reconstruct a coarse series from proto (if spectral proto store magnitudes)
                proto_series = proto.get("top_magnitudes", [])
                dist = self.dtw_distance(series, proto_series) if proto_series else float('inf')
                candidates.append({"id": pid, "name": name, "dist": float(dist), "proto": proto})
            except Exception:
                continue
        candidates.sort(key=lambda x: x["dist"])
        return candidates[:top_k]

    # ================== Branch merging & pruning ==================
    def merge_branches(self, trajectories: List[Dict[str, Any]], threshold: float = 1e-2) -> List[Dict[str, Any]]:
        """
        Merge similar trajectory branches. Each trajectory expected to have 'final' state dict.
        Clustering by greedy agglomeration: merge if L2 distance between finals < threshold.
        """
        if np is None:
            return trajectories
        finals = [np.array([float(v) for v in tr["final"].values()]) for tr in trajectories]
        used = [False] * len(trajectories)
        merged = []
        for i in range(len(trajectories)):
            if used[i]:
                continue
            cluster = [i]
            for j in range(i + 1, len(trajectories)):
                if used[j]:
                    continue
                d = np.linalg.norm(finals[i] - finals[j])
                if d <= threshold:
                    cluster.append(j)
                    used[j] = True
            # aggregate cluster by averaging final vectors and concatenating checkpoints truncated
            agg_final = {}
            keys = list(trajectories[i]["final"].keys())
            for k in keys:
                vals = [trajectories[idx]["final"].get(k, 0.0) for idx in cluster]
                agg_final[k] = float(np.mean(vals))
            # assemble representative
            merged.append({"final": agg_final, "members": cluster})
        return merged

    # ================== Admin listing helpers ==================
    def list_simulations(self, limit: int = 50) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT id, created_ts, name, config_json FROM simulations ORDER BY created_ts DESC LIMIT ?", (limit,))
        out = []
        for sid, ts, name, cfg in cur.fetchall():
            try:
                cfgj = json.loads(cfg) if cfg else {}
            except Exception:
                cfgj = {}
            out.append({"id": sid, "ts": float(ts), "name": name, "config": cfgj})
        return out

    def get_simulation(self, sim_id: int) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT created_ts, name, x0_json, phidesc_json, config_json, result_blob FROM simulations WHERE id = ?", (sim_id,))
        row = cur.fetchone()
        if not row:
            return None
        created_ts, name, x0j, phidescj, cfgj, blob = row
        try:
            result = json.loads(zlib.decompress(blob).decode("utf-8"))
        except Exception:
            result = None
        out = {"id": sim_id, "created_ts": float(created_ts), "name": name}
        try:
            out["x0"] = json.loads(x0j) if x0j else {}
        except Exception:
            out["x0"] = {}
        try:
            out["phidesc"] = json.loads(phidescj) if phidescj else {}
        except Exception:
            out["phidesc"] = {}
        try:
            out["config"] = json.loads(cfgj) if cfgj else {}
        except Exception:
            out["config"] = {}
        out["result_preview"] = {"has_result": result is not None}
        return out

    def list_checkpoints(self, sim_id: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        if sim_id is None:
            cur.execute("SELECT id, sim_id, node_id, ts, blob_hash, resolution_tier FROM checkpoints ORDER BY ts DESC LIMIT ?", (limit,))
        else:
            cur.execute("SELECT id, sim_id, node_id, ts, blob_hash, resolution_tier FROM checkpoints WHERE sim_id = ? ORDER BY ts DESC LIMIT ?", (sim_id, limit))
        out = []
        for cid, sid, nid, ts, bh, rt in cur.fetchall():
            out.append({"id": cid, "sim_id": sid, "node_id": nid, "ts": float(ts), "blob_hash": bh, "resolution": rt})
        return out

    def get_checkpoint(self, checkpoint_id: int) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT blob_hash FROM checkpoints WHERE id = ?", (checkpoint_id,))
        row = cur.fetchone()
        if not row:
            return None
        h = row[0]
        b = self._get_blob(h)
        if b is None:
            return None
        try:
            return json.loads(b.decode("utf-8"))
        except Exception:
            return None

    def list_blobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT hash, size, created_ts FROM blobs ORDER BY created_ts DESC LIMIT ?", (limit,))
        return [{"hash": r[0], "size": int(r[1]), "created_ts": float(r[2])} for r in cur.fetchall()]

    def get_representatives(self, sim_id: int) -> Optional[List[Dict[str, Any]]]:
        """Return representatives stored in a simulation's result blob, if present."""
        cur = self._conn.cursor()
        cur.execute("SELECT result_blob FROM simulations WHERE id = ?", (sim_id,))
        row = cur.fetchone()
        if not row:
            return None
        try:
            result = json.loads(zlib.decompress(row[0]).decode("utf-8"))
        except Exception:
            return None
        reps = result.get("representatives")
        return reps

    def get_particle_results(self, sim_id: int) -> Optional[Dict[str, Any]]:
        """Return particle_filter result saved in a simulation record, if available."""
        cur = self._conn.cursor()
        cur.execute("SELECT result_blob FROM simulations WHERE id = ?", (sim_id,))
        row = cur.fetchone()
        if not row:
            return None
        try:
            result = json.loads(zlib.decompress(row[0]).decode("utf-8"))
        except Exception:
            return None
        # common key names
        if "particle_filter_result" in result:
            return result["particle_filter_result"]
        if "posterior" in result:
            return result["posterior"]
        return None

    # ================== Boundary model helpers ==================
    def sample_boundary(self, boundary_model: Optional[Dict[str, Any]], horizon: int, ensemble: int, rng: Any) -> List[Dict[int, Dict[str, float]]]:
        """
        Produce ensemble of boundary sample sequences. Each sample is a dict: t -> {var_influx: value}.
        boundary_model can specify type and params. Default: gaussian noise influx for ETH/BTC.
        """
        samples = []
        if not boundary_model:
            # default: no boundary influence (empty dictionaries)
            return [{t: {} for t in range(horizon)} for _ in range(ensemble)]

        typ = boundary_model.get("type", "gaussian_flux")
        params = boundary_model.get("params", {})
        if typ == "gaussian_flux":
            mu = float(params.get("mu", 0.0))
            sigma = float(params.get("sigma", 1.0))
            vars = params.get("vars", ["ETH_influx", "BTC_influx"])
            for _ in range(ensemble):
                seq = {}
                for t in range(horizon):
                    entry = {}
                    for v in vars:
                        entry[v] = float(rng.normal(mu, sigma))
                    seq[t] = entry
                samples.append(seq)
            return samples

        # fallback: empty
        return [{t: {} for t in range(horizon)} for _ in range(ensemble)]

    def incorporate_boundary(self, state: Dict[str, float], b_entry: Dict[str, float], phidesc: Dict[str, Any]) -> None:
        """
        Mutate state in-place to incorporate boundary influx values. Mapping rules from phidesc.params.
        """
        if not b_entry:
            return
        mapping = phidesc.get("params", {}).get("boundary_map", {"field1_influx": ("field1", 1.0)})
        # mapping: boundary_key -> (state_var, scale)
        for bk, val in b_entry.items():
            if bk in mapping:
                target, scale = mapping[bk]
                state[target] = float(state.get(target, 0.0) + float(val) * float(scale))

    # ================== merge threshold calibration ==================
    def calibrate_merge_threshold(self, phidesc: Optional[Dict[str, Any]] = None, horizon: int = 12, ensemble: int = 32, percentile: float = 10.0, rng_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a short calibration ensemble and compute a suggested merge_threshold as the given percentile
        of pairwise DTW distances among representative series (primary variable).
        """
        rng = np.random.default_rng(rng_seed if rng_seed is not None else int(time.time() % (2 ** 31)))
        # run lightweight sims using phi if provided otherwise default
        phidesc_local = phidesc or {"type": "econ_simple", "params": {}}
        # temporary run: use small ensemble with method phi
        samples = []
        primary_var = self.get_default_universe_config()["variables"][0]
        for _ in range(ensemble):
            state = self.get_default_universe_config()["initial"].copy()
            seq = []
            phi = self.get_domain_phi(phidesc_local)
            for t in range(horizon):
                state = phi(state, t, rng)
                seq.append(state.get(primary_var, 0.0))
            samples.append(self.normalize_series(seq))

        # compute pairwise DTW distances
        dists = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                try:
                    d = self.dtw_distance(samples[i], samples[j])
                except Exception:
                    d = float('inf')
                dists.append(d)
        if not dists:
            return {"suggested_threshold": 0.0, "percentile": percentile}
        arr = np.array(dists)
        sugg = float(np.percentile(arr, percentile))
        # clamp to nonzero minimum
        sugg = max(sugg, 1e-6)
        return {"suggested_threshold": sugg, "percentile": percentile, "sample_count": len(dists)}

    # ================== Particle filter assimilation ==================
    def _systematic_resample(self, weights: np.ndarray, rng: Any) -> np.ndarray:
        """Systematic resampling: returns indices array of length N."""
        N = len(weights)
        positions = (rng.random() + np.arange(N)) / N
        cumulative = np.cumsum(weights)
        indices = np.zeros(N, dtype=int)
        i, j = 0, 0
        while i < N and j < N:
            if positions[i] < cumulative[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return indices

    def particle_filter(
        self,
        initial_state: Dict[str, float],
        observations: Dict[int, Dict[str, float]],
        phidesc: Optional[Dict[str, Any]] = None,
        boundary_model: Optional[Dict[str, Any]] = None,
        num_particles: int = 128,
        horizon: Optional[int] = None,
        rng_seed: Optional[int] = None,
        obs_var: float = 1.0,
        progress_cb: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Particle filter assimilation over horizon steps.

        observations: dict mapping t -> {var: value} (observations at times)
        phidesc: domain phi descriptor
        boundary_model: optional boundary model descriptor
        Returns posterior particles and summary.
        """
        if np is None:
            raise RuntimeError("numpy required for particle_filter")
        rng = np.random.default_rng(rng_seed if rng_seed is not None else int(time.time() % (2 ** 31)))
        phidesc_local = phidesc or {"type": "econ_simple", "params": {}}
        phi = self.get_domain_phi(phidesc_local)
        H = horizon if horizon is not None else max(observations.keys()) + 1 if observations else 24

        # sample boundary sequences for each particle
        # clamp particle count to configured maximum to avoid OOM
        if num_particles > getattr(self, "_max_particles", 4096):
            LOG.warning("requested num_particles=%s exceeds MAX_PARTICLES=%s; clamping", num_particles, getattr(self, "_max_particles", 4096))
            num_particles = getattr(self, "_max_particles", 4096)
        boundary_samples = self.sample_boundary(boundary_model, H, num_particles, rng)

        # initialize particles: small perturbations around initial_state
        var_list = list(initial_state.keys())
        particles = []
        for i in range(num_particles):
            pstate = {k: float(initial_state[k] + rng.normal(0, max(abs(initial_state[k]) * 0.01, 1e-6))) for k in var_list}
            particles.append(pstate)

        weights = np.ones(num_particles, dtype=float) / num_particles
        ess_history = []
        particles_history = []

        for t in range(H):
            # propagate particles
            for i in range(num_particles):
                b_entry = boundary_samples[i].get(t, {})
                # incorporate boundary influx
                self.incorporate_boundary(particles[i], b_entry, phidesc_local)
                # propagate via phi
                particles[i] = phi(particles[i], t, rng)

            # weight update if observation exists at time t
            if t in observations:
                obs = observations[t]
                # compute likelihoods
                ws = np.zeros(num_particles, dtype=float)
                for i in range(num_particles):
                    # measurement function H is identity on observed vars
                    err_sq = 0.0
                    for k, v in obs.items():
                        pv = particles[i].get(k, 0.0)
                        err_sq += (float(pv) - float(v)) ** 2
                    # Gaussian likelihood with variance obs_var
                    ws[i] = np.exp(-0.5 * err_sq / max(obs_var, 1e-9))
                # avoid all-zero
                if ws.sum() <= 0.0 or not np.isfinite(ws.sum()):
                    ws = np.ones_like(ws)
                # multiply prior weights
                weights = weights * ws
                weights_sum = weights.sum()
                if weights_sum <= 0 or not np.isfinite(weights_sum):
                    weights = np.ones_like(weights) / len(weights)
                else:
                    weights = weights / weights_sum

            # compute ESS and resample if needed
            ess = 1.0 / np.sum(weights ** 2)
            ess_history.append(float(ess))
            if progress_cb:
                try:
                    progress_cb({"t": t, "ess": float(ess)})
                except Exception:
                    pass
            if ess < (num_particles / 2.0):
                indices = self._systematic_resample(weights, rng)
                particles = [dict(particles[idx]) for idx in indices]
                weights = np.ones(num_particles, dtype=float) / num_particles

            # store particle snapshot summary
            if t % max(1, H // 10) == 0 or t == H - 1:
                # compute weighted mean
                mean_state = {}
                for k in var_list:
                    vals = np.array([particles[i].get(k, 0.0) for i in range(num_particles)])
                    mean_state[k] = float(np.average(vals, weights=weights))
                particles_history.append({"t": t, "mean": mean_state, "ess": float(ess)})

        # final posterior summary
        posterior = {"num_particles": num_particles, "ess_history": ess_history, "particles_history": particles_history}
        return posterior

    # ================== Incremental particle filter primitives for streaming ==================
    def particle_filter_init(self, initial_state: Dict[str, float], phidesc: Optional[Dict[str, Any]] = None, boundary_model: Optional[Dict[str, Any]] = None, num_particles: int = 128, horizon: int = 100, rng_seed: Optional[int] = None) -> Dict[str, Any]:
        """Initialize PF state for streaming assimilation. Returns a state dict to keep in JobManager."""
        rng = np.random.default_rng(rng_seed if rng_seed is not None else int(time.time() % (2 ** 31)))
        var_list = list(initial_state.keys())
        particles = []
        # clamp particle count to configured maximum
        if num_particles > getattr(self, "_max_particles", 4096):
            LOG.warning("Initializing PF with num_particles %s > MAX_PARTICLES %s; clamping", num_particles, getattr(self, "_max_particles", 4096))
            num_particles = getattr(self, "_max_particles", 4096)
        for i in range(num_particles):
            pstate = {k: float(initial_state[k] + rng.normal(0, max(abs(initial_state[k]) * 0.01, 1e-6))) for k in var_list}
            particles.append(pstate)
        weights = np.ones(num_particles, dtype=float) / num_particles
        boundary_samples = self.sample_boundary(boundary_model, horizon, num_particles, rng)
        return {"particles": particles, "weights": weights.tolist(), "boundary_samples": boundary_samples, "rng_seed": int(rng_seed) if rng_seed is not None else None, "var_list": var_list, "t": 0}

    def particle_filter_step(self, pf_state: Dict[str, Any], observation: Optional[Dict[str, float]], phidesc: Optional[Dict[str, Any]] = None, obs_var: float = 1.0) -> Dict[str, Any]:
        """Perform one PF step given pf_state and optional observation at current t. Mutates pf_state and returns summary."""
        if np is None:
            raise RuntimeError("numpy required")
        rng = np.random.default_rng(pf_state.get("rng_seed", None))
        particles = pf_state["particles"]
        weights = np.array(pf_state["weights"], dtype=float)
        boundary_samples = pf_state.get("boundary_samples", [{} for _ in range(len(particles))])
        t = pf_state.get("t", 0)
        phidesc_local = phidesc or {"type": "econ_simple", "params": {}}
        phi = self.get_domain_phi(phidesc_local)
        num_particles = len(particles)
        # propagate
        for i in range(num_particles):
            b_entry = boundary_samples[i].get(t, {})
            self.incorporate_boundary(particles[i], b_entry, phidesc_local)
            particles[i] = phi(particles[i], t, rng)
        # incorporate observation
        if observation:
            ws = np.zeros(num_particles, dtype=float)
            for i in range(num_particles):
                err_sq = 0.0
                for k, v in observation.items():
                    pv = particles[i].get(k, 0.0)
                    err_sq += (float(pv) - float(v)) ** 2
                ws[i] = np.exp(-0.5 * err_sq / max(obs_var, 1e-9))
            if ws.sum() <= 0 or not np.isfinite(ws.sum()):
                ws = np.ones_like(ws)
            weights = weights * ws
            if weights.sum() <= 0 or not np.isfinite(weights.sum()):
                weights = np.ones_like(weights) / num_particles
            else:
                weights = weights / weights.sum()
        # resample if ESS low
        ess = 1.0 / np.sum(weights ** 2)
        if ess < (num_particles / 2.0):
            indices = self._systematic_resample(weights, rng)
            particles = [dict(particles[idx]) for idx in indices]
            weights = np.ones(num_particles, dtype=float) / num_particles
        # advance time
        pf_state["particles"] = particles
        pf_state["weights"] = weights.tolist()
        pf_state["t"] = t + 1
        # summary
        mean_state = {}
        var_list = pf_state.get("var_list", list(particles[0].keys()) if particles else [])
        for k in var_list:
            vals = np.array([particles[i].get(k, 0.0) for i in range(num_particles)])
            mean_state[k] = float(np.average(vals, weights=weights))
        return {"t": t, "ess": float(ess), "mean": mean_state}

    def run_reconstructability_test(self, horizon: int = 10, ensemble: int = 16) -> Dict[str, Any]:
        """
        Run a small simulation and attempt to reconstruct a sample trajectory from X0+phidesc+checkpoints.
        Returns reconstruction error summary.
        """
        cfg = self.get_default_universe_config()
        seed = cfg["initial"]
        out = self.run_simulation(seed_state=seed, horizon=horizon, ensemble=ensemble, name="recon_test", rng_seed=12345)
        sim_id = out.get("sim_id")
        if not sim_id:
            return {"error": "simulation failed"}
        # get the stored result blob and attempt to reconstruct one trajectory
        cur = self._conn.cursor()
        cur.execute("SELECT result_blob FROM simulations WHERE id = ?", (sim_id,))
        row = cur.fetchone()
        if not row:
            return {"error": "no result blob"}
        try:
            data = zlib.decompress(row[0])
            result = json.loads(data.decode("utf-8"))
        except Exception as e:
            return {"error": f"failed to read blob: {e}"}
        # pick first trajectory and compare to re-simulated simple model
        traj = result.get("results", [])
        if not traj:
            return {"error": "no trajectories"}
        sample = traj[0]
        # reconstruct: naive re-simulate single path with same rng seed
        recon = self.run_simulation(seed_state=seed, horizon=horizon, ensemble=1, name="recon_playback", rng_seed=12345)
        # load new sim
        cur.execute("SELECT result_blob FROM simulations WHERE id = ?", (recon.get("sim_id"),))
        row2 = cur.fetchone()
        if not row2:
            return {"error": "replay blob missing"}
        try:
            data2 = zlib.decompress(row2[0])
            result2 = json.loads(data2.decode("utf-8"))
        except Exception as e:
            return {"error": f"failed to read replay blob: {e}"}
        traj2 = result2.get("results", [])
        if not traj2:
            return {"error": "no replay trajectories"}
        sample2 = traj2[0]
        # compute simple L1 distance across shared keys
        keys = set(sample.keys()) & set(sample2.keys())
        if not keys:
            return {"error": "no matching keys"}
        dist = sum(abs(float(sample[k]) - float(sample2[k])) for k in keys) / len(keys)
        return {"sim_id": sim_id, "replay_sim_id": recon.get("sim_id"), "mean_l1_dist": float(dist)}

    def ingest(self, sample: Dict[str, float]) -> None:
        """Ingest a single sample into the short-term buffer (persist as macro event)."""
        ts = time.time()
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO events(sim_id, ts, parent_id, state_macro_json) VALUES (?, ?, ?, ?)",
            (None, float(ts), None, json.dumps(sample)),
        )
        self._conn.commit()

    def _store_simulation_result(self, name: str, x0: Dict[str, Any], phidesc: Dict[str, Any], config: Dict[str, Any], result: Dict[str, Any]) -> int:
        # attach provenance if missing
        if "provenance" not in result:
            try:
                prov = self._capture_provenance(x0, phidesc or {}, config or {})
                result["provenance"] = prov
            except Exception:
                result["provenance"] = {}
        blob = _compress_bytes(json.dumps(result).encode("utf-8"))
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO simulations(created_ts, name, x0_json, phidesc_json, config_json, result_blob) VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), name, json.dumps(x0), json.dumps(phidesc), json.dumps(config), blob),
        )
        sim_id = cur.lastrowid
        self._conn.commit()
        self._last_sim_id = sim_id
        return sim_id

    def _serialize_agent_graph(self) -> Optional[Dict[str, Any]]:
        """Serialize agent causal graph (nodes, edges with attributes) if available."""
        if not self.agent:
            return None
        try:
            # guard with lock
            with self._agent_lock:
                g = self.agent.causal_graph
                nodes = list(g.nodes())
                edges = []
                for u, v in g.edges():
                    attrs = dict(g[u][v])
                    edges.append({"u": u, "v": v, "attrs": attrs})
                return {"nodes": nodes, "edges": edges}
        except Exception:
            return None

    def _attach_agent_graph_to_sim(self, sim_id: int) -> None:
        """Attach serialized agent graph into an existing simulation result blob."""
        if not self.agent:
            return
        graph = self._serialize_agent_graph()
        if not graph:
            return
        cur = self._conn.cursor()
        cur.execute("SELECT result_blob FROM simulations WHERE id = ?", (sim_id,))
        row = cur.fetchone()
        if not row:
            return
        try:
            data = zlib.decompress(row[0])
            result = json.loads(data.decode("utf-8"))
        except Exception:
            return
        result["agent_graph"] = graph
        new_blob = _compress_bytes(json.dumps(result).encode("utf-8"))
        cur.execute("UPDATE simulations SET result_blob = ? WHERE id = ?", (new_blob, sim_id))
        self._conn.commit()

    def predict_with_agent(self, state: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Thread-safe wrapper to predict outcomes using the CRCAAgent."""
        if not self.agent:
            return None
        try:
            with self._agent_lock:
                # prefer cached predictor if available
                if hasattr(self.agent, "_predict_outcomes_cached"):
                    return self.agent._predict_outcomes_cached(state, {})
                elif hasattr(self.agent, "_predict_outcomes"):
                    return self.agent._predict_outcomes(state, {})
                elif hasattr(self.agent, "predict"):
                    return self.agent.predict(state)
                else:
                    return None
        except Exception:
            return None

    def query_graph(self) -> str:
        """Return a textual representation of the causal graph if available."""
        if self.agent is None:
            return "No CRCA agent available in this runtime."
        try:
            with self._agent_lock:
                return self.agent.get_causal_graph_visualization()
        except Exception as e:
            return f"crca agent error: {e}"

    def run_simulation(
        self,
        seed_state: Dict[str, float],
        horizon: int = 24,
        branching: int = 4,
        ensemble: int = 64,
        name: str = "sim_run",
        rng_seed: Optional[int] = None,
        method: str = "agent",
        max_materialize: int = 128,
        merge_threshold: float = 1.0,
        phidesc: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a Monte Carlo ensemble simulation forward from seed_state.

        Returns a summary and persists the full compressed result to the local DB.
        """
        # deterministic rng for reproducibility
        if np is None:
            raise RuntimeError("numpy is required for simulation")
        rng = np.random.default_rng(rng_seed if rng_seed is not None else int(time.time() % (2 ** 31)))

        # If phidesc requests tiles, use multiscale path
        if phidesc and (phidesc.get("tiles") or {}).get("n_tiles"):
            return self.run_multiscale_simulation(seed_state=seed_state, horizon=horizon, phidesc=phidesc, ensemble=ensemble, rng_seed=rng_seed, dt=float((phidesc.get("evolution_solver") or {}).get("dt", 1.0)))

        results: List[Dict[str, Any]] = []
        # choose model: CRCAAgent prediction or fallback simple stochastic dynamics
        use_agent = self.agent is not None
        # normalize phidesc for safe access in code below
        phidesc_local = phidesc or {}
        # phi_func may be set when using domain phidesc
        phi_func = None
        # Precompute linear dynamics matrix if needed (from default universe config)
        default_cfg = self.get_default_universe_config()
        var_list = list(default_cfg.get("variables", list(seed_state.keys())))
        var_index = {v: i for i, v in enumerate(var_list)}
        nvars = len(var_list)
        A = None
        if method in ("rk4", "sde"):
            # build small adjacency-weighted matrix
            A = np.zeros((nvars, nvars))
            for (u, v) in default_cfg.get("edges", []):
                if u in var_index and v in var_index:
                    A[var_index[v], var_index[u]] = 1e-3  # small coupling

        # per-trajectory simulation with optional checkpoint collection
        all_checkpoints: List[Tuple[int, List[Dict[str, Any]]]] = []  # (traj_idx, [ (t,state) ])
        # representatives for online merging
        reps: List[Dict[str, Any]] = []
        pruned_count = 0
        for s in range(int(ensemble)):
            # initialize
            state = dict(seed_state)
            traj_checkpoints: List[Dict[str, Any]] = []
            trajectory_states: List[Dict[str, float]] = []
            last_state_vec = np.array([float(state.get(v, 0.0)) for v in var_list]) if np is not None else None

            for t in range(int(horizon)):
                # domain Phi usage
                if phidesc is not None:
                    # domain-specific discrete step function
                    if phi_func is None:
                        phi_func = self.get_domain_phi(phidesc_local)
                    state = phi_func(state, t, rng)
                elif method == "agent" and use_agent:
                    try:
                        pred = self.predict_with_agent(state)
                        if pred:
                            state = {k: float(v) for k, v in pred.items()}
                        else:
                            raise RuntimeError("agent returned no prediction")
                    except Exception:
                        # fallback drift
                        for k, v in list(state.items()):
                            noise = float(rng.normal(0, abs(v) * 0.01 + 1e-6))
                            state[k] = float(v + 0.01 * v + noise)
                elif method == "rk4" and A is not None:
                    # linear ODE: x' = A x
                    x = np.array([float(state.get(v, 0.0)) for v in var_list])
                    dt = 1.0
                    k1 = A @ x
                    k2 = A @ (x + 0.5 * dt * k1)
                    k3 = A @ (x + 0.5 * dt * k2)
                    k4 = A @ (x + dt * k3)
                    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                    for i, v in enumerate(var_list):
                        state[v] = float(x_next[i])
                elif method == "sde" and A is not None:
                    x = np.array([float(state.get(v, 0.0)) for v in var_list])
                    dt = 1.0
                    drift = A @ x
                    noise_vec = rng.normal(0, 1.0, size=nvars)
                    x_next = x + drift * dt + 0.1 * np.sqrt(dt) * noise_vec
                    for i, v in enumerate(var_list):
                        state[v] = float(x_next[i])
                else:
                    # default multiplicative drift
                    for k, v in list(state.items()):
                        drift = 0.01 * float(v)
                        noise = float(rng.normal(0, max(abs(v) * 0.01, 1e-6)))
                        state[k] = float(v + drift + noise)

                # store macro state and sampling checkpoints heuristics
                trajectory_states.append(dict(state))
                if last_state_vec is not None:
                    cur_vec = np.array([float(state.get(v, 0.0)) for v in var_list])
                    delta = float(np.linalg.norm(cur_vec - last_state_vec))
                    # checkpoint if change above threshold or periodic
                    if delta > 1e-2 * max(1.0, np.linalg.norm(last_state_vec)) or (t % max(1, horizon // 4) == 0):
                        traj_checkpoints.append({"t": t, "state": dict(state)})
                    last_state_vec = cur_vec
                else:
                    traj_checkpoints.append({"t": t, "state": dict(state)})

                # === online merging check (mid-simulation) ===
                # perform merging after a checkpoint is created
                if traj_checkpoints:
                    primary_var = var_list[0] if var_list else None
                    if primary_var is not None:
                        series_partial = [cp["state"].get(primary_var, 0.0) for cp in traj_checkpoints]
                        series_norm = self.normalize_series(series_partial)
                        merged_early = False
                        for ridx, rep in enumerate(reps):
                            # compare to rep prefix
                            rep_series = rep.get("series", [])
                            # take prefix of rep to match length
                            if len(rep_series) >= len(series_norm) and len(series_norm) > 0:
                                try:
                                    rep_prefix = rep_series[: len(series_norm)]
                                    rep_prefix_norm = self.normalize_series(rep_prefix)
                                    dist = self.dtw_distance(series_norm, rep_prefix_norm)
                                except Exception:
                                    dist = float('inf')
                            else:
                                # if rep shorter, pad rep and compare
                                try:
                                    rep_prefix_norm = self.normalize_series(rep.get("series", []))
                                    dist = self.dtw_distance(series_norm, rep_prefix_norm)
                                except Exception:
                                    dist = float('inf')
                            if dist <= merge_threshold:
                                # merge and prune trajectory early
                                rep_count = rep.get("count", 1)
                                for k in final_state.keys():
                                    rep_val = rep["final"].get(k, 0.0)
                                    rep["final"][k] = float((rep_val * rep_count + state.get(k, 0.0)) / (rep_count + 1))
                                rep["count"] = rep_count + 1
                                # update rep series by padding/averaging
                                a = np.array(rep.get("series", []), dtype=float)
                                b = np.array(series_norm, dtype=float)
                                if a.size < b.size:
                                    a = np.pad(a, (0, b.size - a.size), 'edge') if a.size>0 else np.zeros_like(b)
                                elif b.size < a.size:
                                    b = np.pad(b, (0, a.size - b.size), 'edge')
                                rep["series"] = ((a * rep_count + b) / (rep_count + 1)).tolist()
                                rep["members"].append(s)
                                pruned_count += 1
                                merged_early = True
                                break
                        if merged_early:
                            # stop simulating this trajectory
                            break

            final_state = dict(state)
            # online merging based on DTW of checkpoint series for primary variable
            primary_var = var_list[0] if var_list else None
            series = []
            if primary_var is not None:
                # prefer checkpoint samples; fallback to sampled trajectory states
                if traj_checkpoints:
                    series = [cp["state"].get(primary_var, 0.0) for cp in traj_checkpoints]
                else:
                    series = [st.get(primary_var, 0.0) for st in trajectory_states]

            merged_into = None
            if series and len(reps) > 0:
                for ridx, rep in enumerate(reps):
                    try:
                        dist = self.dtw_distance(series, rep["series"])
                    except Exception:
                        dist = float('inf')
                    if dist <= float(phidesc_local.get("merge_threshold", merge_threshold)):
                        # merge: update representative aggregate
                        rep_count = rep.get("count", 1)
                        # weighted average of final vectors
                        for k in final_state.keys():
                            rep_val = rep["final"].get(k, 0.0)
                            rep["final"][k] = float((rep_val * rep_count + final_state.get(k, 0.0)) / (rep_count + 1))
                        rep["count"] = rep_count + 1
                        # update series by averaging (pad)
                        a = np.array(rep["series"], dtype=float)
                        b = np.array(series, dtype=float)
                        if a.size < b.size:
                            a = np.pad(a, (0, b.size - a.size), 'edge')
                        elif b.size < a.size:
                            b = np.pad(b, (0, a.size - b.size), 'edge')
                        rep["series"] = ((a * rep_count + b) / (rep_count + 1)).tolist()
                        rep["members"].append(s)
                        merged_into = ridx
                        pruned_count += 1
                        break

            if merged_into is None:
                # create new representative
                reps.append({"final": final_state, "series": series, "members": [s], "count": 1, "checkpoints": traj_checkpoints})
                results.append({"final": final_state, "checkpoints": traj_checkpoints})
            else:
                # store only minimal info for pruned trajectory
                all_checkpoints.append((s, traj_checkpoints))

        # aggregate macro statistics from final states (results store dicts with 'final')
        if results:
            keys = set().union(*[set(r["final"].keys()) for r in results])
        else:
            keys = set(seed_state.keys())
        aggregate = {}
        arr_by_key = {}
        for k in keys:
            vals = []
            for r in results:
                f = r.get("final", r)
                v = f.get(k, 0.0) if isinstance(f, dict) else 0.0
                vals.append(float(v))
            arr = np.array(vals, dtype=float) if vals else np.array([], dtype=float)
            arr_by_key[k] = arr
            aggregate[k] = {"mean": float(np.mean(arr)) if arr.size > 0 else 0.0, "std": float(np.std(arr)) if arr.size > 0 else 0.0, "samples": int(arr.size)}

        # persist compressed full result
        phidesc = {"type": "hybrid_default", "branching": branching}
        config = {"horizon": horizon, "branching": branching, "ensemble": ensemble, "rng_seed": int(rng_seed) if rng_seed is not None else None}
        x0 = seed_state
        # prune materialization: keep only top-K full trajectories if ensemble large
        materialized = results
        if ensemble > max_materialize:
            # rank by L2 norm of final vector (largest magnitude)
            scored = []
            for idx, r in enumerate(results):
                vec = np.array([r["final"].get(v, 0.0) for v in var_list])
                scored.append((idx, float(np.linalg.norm(vec))))
            scored.sort(key=lambda x: x[1], reverse=True)
            keep_idx = set(idx for idx, _ in scored[:max_materialize])
            materialized = [results[i] for i in sorted(keep_idx)]
            # replace results with materialized + summary indicating many pruned
            result = {"seed": seed_state, "horizon": horizon, "ensemble": ensemble, "materialized": materialized, "pruned_count": len(results) - len(materialized), "aggregate": aggregate, "created_ts": time.time()}
        else:
            result = {"seed": seed_state, "horizon": horizon, "ensemble": ensemble, "results": results, "aggregate": aggregate, "created_ts": time.time()}

        sim_id = self._store_simulation_result(name, x0, phidesc, config, result)
        LOG.info("Simulation stored id=%s", sim_id)
        # attach agent graph metadata if agent present
        try:
            self._attach_agent_graph_to_sim(sim_id)
        except Exception:
            pass
        # persist checkpoints now that sim_id is known
        for traj_idx, cps in all_checkpoints:
            for cp in cps:
                try:
                    self.create_checkpoint(sim_id, node_id=traj_idx, state=cp["state"], resolution_tier="micro")
                except Exception:
                    pass
        return {"sim_id": sim_id, "summary": aggregate}

    def snapshot(self, name: str = "manual_snapshot") -> int:
        """Create a small snapshot record (macro-level) and persist it as a simulation with empty result."""
        x0 = {"snapshot_ts": time.time()}
        phidesc = {"type": "snapshot"}
        config = {}
        result = {"note": "snapshot created", "ts": time.time()}
        return self._store_simulation_result(name, x0, phidesc, config, result)

    def close(self) -> None:
        try:
            self._conn.commit()
            self._conn.close()
        except Exception:
            pass


