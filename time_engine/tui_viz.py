#!/usr/bin/env python3
"""
Minimal TUI visualizer for Time Engine PF streaming.

Usage:
  python tui_viz.py --host 127.0.0.1 --port 8765 --job 1 [--api-key KEY] [--primary-var field1]

Keys:
  q - quit
  u - unsubscribe
  s - fetch and show representatives
  p - fetch and show particles (counts)
"""
import argparse
import threading
import socket
import curses
import json
import time
from collections import deque


class TUIClient:
    def __init__(self, host, port, job_id, api_key=None, primary_var=None):
        self.host = host
        self.port = port
        self.job_id = job_id
        self.tile_id = None
        self.api_key = api_key
        self.primary_var = primary_var
        self.history = deque(maxlen=80)
        self.ess = None
        self.t = None
        self.running = True

    def connect(self):
        # use blocking socket in a reader thread to avoid asyncio/curses interaction
        self.sock = socket.create_connection((self.host, int(self.port)))
        self.sock_file = self.sock.makefile("rb")
        subs = {"cmd": "subscribe_pf", "job_id": int(self.job_id)}
        if self.api_key:
            subs["api_key"] = self.api_key
        self.sock.sendall(json.dumps(subs).encode() + b"\n")
        # start reader thread
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _reader_loop(self):
        try:
            while self.running:
                line = self.sock_file.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode())
                except Exception:
                    continue
                # tile push payloads contain 'tile_id' and 'diagnostics'
                if isinstance(msg, dict) and "tile_id" in msg and msg.get("job_id") == (int(self.job_id) if self.job_id is not None else None):
                    tile = int(msg.get("tile_id"))
                    if self.tile_id is None or tile == int(self.tile_id):
                        diag = msg.get("diagnostics", {})
                        mean = diag.get("mean", {})
                        pv = self.primary_var or (list(mean.keys())[0] if mean else None)
                        if pv and pv in mean:
                            try:
                                self.history.append(float(mean[pv]))
                            except Exception:
                                pass
                        self.t = diag.get("t", self.t)
                else:
                    summary = msg.get("summary") or msg.get("pf_observe") or msg.get("particle_filter_result") or msg
                    if isinstance(summary, dict) and "mean" in summary:
                        m = summary["mean"]
                        pv = self.primary_var or (list(m.keys())[0] if m else None)
                        if pv and pv in m:
                            try:
                                self.history.append(float(m[pv]))
                            except Exception:
                                pass
                        self.ess = summary.get("ess", self.ess)
                        self.t = summary.get("t", self.t)
                # handle branches listing responses
                if isinstance(msg, dict) and "branches" in msg:
                    try:
                        branches = msg.get("branches") or []
                        lines = []
                        for b in branches:
                            lines.append(f"BRANCH id={b.get('id')} tile={b.get('tile_id')} t={b.get('t')} status={b.get('status')} depth={b.get('depth')}")
                        if lines:
                            # append joined lines to history for display
                            self.history.append(lines[0])
                            for ln in lines[1:]:
                                self.history.append(ln)
                    except Exception:
                        pass
        except Exception:
            pass

    # reader functionality implemented in _reader_loop thread

    def draw_sparkline(self, win, y, x, width):
        vals = list(self.history)
        if not vals:
            win.addstr(y, x, "(no data)")
            return
        lo = min(vals)
        hi = max(vals)
        rng = hi - lo if hi != lo else 1.0
        chars = "▁▂▃▄▅▆▇█"
        s = ""
        for v in vals[-width:]:
            idx = int((v - lo) / rng * (len(chars) - 1))
            s += chars[max(0, min(idx, len(chars) - 1))]
        win.addstr(y, x, s)

    def tui_loop(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(200)
        while self.running:
            stdscr.erase()
            stdscr.addstr(0, 0, f"PF TUI - job {self.job_id} - primary var: {self.primary_var}")
            stdscr.addstr(1, 0, f"t: {self.t}  ESS: {self.ess}")
            self.draw_sparkline(stdscr, 3, 0, 60)
            stdscr.addstr(5, 0, "q=quit u=unsubscribe s=reps p=particles")
            stdscr.refresh()
            try:
                ch = stdscr.getch()
                if ch == ord("q"):
                    self.running = False
                    break
                if ch == ord("u"):
                    try:
                        self.sock.sendall(json.dumps({"cmd": "unsubscribe_pf", "job_id": int(self.job_id)}).encode() + b"\n")
                    except Exception:
                        pass
                if ch == ord("s"):
                    try:
                        self.sock.sendall(json.dumps({"cmd": "get_representatives", "sim_id": int(self.job_id)}).encode() + b"\n")
                    except Exception:
                        pass
                if ch == ord("p"):
                    try:
                        self.sock.sendall(json.dumps({"cmd": "get_particles", "sim_id": int(self.job_id)}).encode() + b"\n")
                    except Exception:
                        pass
            except Exception:
                pass
            time.sleep(0.1)

    def run(self):
        # connect and start reader thread
        self.connect()
        try:
            curses.wrapper(self.tui_loop)
        except Exception:
            pass
        self.running = False
        try:
            if hasattr(self, "sock_file"):
                self.sock_file.close()
            if hasattr(self, "sock"):
                self.sock.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--job", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--primary-var", default=None)
    args = parser.parse_args()
    client = TUIClient(args.host, args.port, args.job, api_key=args.api_key, primary_var=args.primary_var)
    client.run()


if __name__ == "__main__":
    main()


