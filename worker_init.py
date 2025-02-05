# worker_init.py
import signal
import os
import json
import glob
import time
import sys
import ast


def snapshot_signal_handler(signum, frame):
    # (Your existing snapshot_signal_handler code)
    snapshot_vars = {}
    frames = sys._current_frames()
    for tid, frm in frames.items():
        current = frm
        while current:
            try:
                frame_file = os.path.abspath(current.f_code.co_filename)
            except Exception:
                frame_file = None
            # Skip frames from our analyzer (adjust if needed)
            if frame_file and frame_file.endswith("codeanalyser.py"):
                current = current.f_back
                continue
            func_name = current.f_code.co_name
            for var, value in current.f_locals.items():
                if var in ["args", "snapshot_vars"]:
                    continue
                try:
                    example = value if isinstance(
                        value, (int, float, str, list, dict)) else repr(value)
                except Exception:
                    example = repr(value)
                if var in snapshot_vars:
                    snapshot_vars[var]["functions"].add(func_name)
                else:
                    snapshot_vars[var] = {
                        "example": example, "functions": set([func_name])
                    }
            current = current.f_back
    for var in snapshot_vars:
        snapshot_vars[var]["functions"] = list(snapshot_vars[var]["functions"])
    filename = f"snapshot_vars_worker_{os.getpid()}.json"
    try:
        with open(filename, "w") as f:
            json.dump(snapshot_vars, f, indent=4)
        print(f"[Worker {os.getpid()}] Snapshot saved to {filename}.")
    except Exception as e:
        print(f"[Worker {os.getpid()}] Error writing snapshot: {e}")


def process_pool_initializer():
    # Use SIGUSR1 if available, otherwise use SIGBREAK (available on Windows)
    sig = getattr(signal, "SIGUSR1", None) or getattr(signal, "SIGBREAK", None)
    if sig is None:
        print("No suitable signal found for snapshot capturing.")
        return
    signal.signal(sig, snapshot_signal_handler)
