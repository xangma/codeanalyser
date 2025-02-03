#!/usr/bin/env python3
"""
codeanalyser.py
This script maps out an entire Python codebase (all .py files in the target script’s directory,
excluding files matching patterns in .gitignore) and runs the target script to capture example
variable values across functions (from both the main process and any ProcessPool workers). The final
JSON output contains two keys:
  - "codemap": A list of code maps (one per Python file), each including module-level functions,
               classes and their methods. (Optionally, source code snippets can be included.)
  - "variable_examples": A consolidated mapping from variable names to a sample value and a list
                         of function names where the variable appears.

Usage:
    ./analyze_and_snapshot.py path/to/target_script.py output.json [--timeout 10] [--include-source]
         [--filter-file substring [substring ...]] [--filter-name substring [substring ...]]

The --include-source flag (default off) controls whether to include source code snippets for each function/method.
The --filter-file and --filter-name options limit the output codemap and variable examples to only those files
and function/class names that match the given substrings.
"""

import os
import sys
import ast
import json
import argparse
import runpy
import threading
import time
import glob
import signal
import psutil  # pip install psutil
import concurrent.futures

# Try to import pathspec for .gitignore support. If not installed, no filtering is performed.
try:
    import pathspec
except ImportError:
    pathspec = None

# Limit for one-level conversion of large objects (e.g. only the first few elements)
LIMIT = 10

# Define the absolute path of this analyzer script so it can be skipped in analyses.
SELF_PATH = os.path.abspath(__file__)

# =============================================================================
# Code mapping functions
# =============================================================================

def extract_function_source(filepath, target_lineno, target_name):
    """
    Given a file path and the starting line number for a function,
    re-parse the file to locate the function definition and return its source snippet.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

    try:
        tree = ast.parse(source, filename=filepath)
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_name and node.lineno == target_lineno:
            if hasattr(node, "end_lineno") and node.end_lineno is not None:
                lines = source.splitlines()
                snippet = "\n".join(lines[node.lineno - 1: node.end_lineno])
                return snippet
            else:
                # Fallback heuristic: capture until indent decreases.
                lines = source.splitlines()
                start_line = node.lineno - 1
                base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
                snippet_lines = [lines[start_line]]
                for l in lines[start_line+1:]:
                    if l.strip() == "":
                        snippet_lines.append(l)
                        continue
                    current_indent = len(l) - len(l.lstrip())
                    if current_indent <= base_indent:
                        break
                    snippet_lines.append(l)
                return "\n".join(snippet_lines)
    return None

def get_function_signature(node):
    """
    Build a simple signature string for a function from its AST node.
    """
    args = []
    for arg in node.args.args:
        args.append(arg.arg)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    return "(" + ", ".join(args) + ")"

class CallCollector(ast.NodeVisitor):
    """
    Walks through a function/method body and collects function calls.
    """
    def __init__(self):
        self.calls = []
    def visit_Call(self, node):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            curr = node.func
            while isinstance(curr, ast.Attribute):
                parts.insert(0, curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                parts.insert(0, curr.id)
            func_name = ".".join(parts)
        if func_name:
            self.calls.append(func_name)
            self.calls = list(set(self.calls))
        self.generic_visit(node)

def build_codemap(file_path, include_source=False):
    """
    Build a code map from the given Python file. The result is a dictionary with:
      - file: the file path
      - functions: list of module-level function info
      - classes: list of classes (each with methods)
    If include_source is True, a source snippet is added for each function/method.
    """
    codemap = {"file": file_path, "functions": [], "classes": []}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return codemap

    try:
        tree = ast.parse(source, filename=file_path)
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return codemap

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_info = {
                "name": node.name,
                "lineno": node.lineno,
                "signature": get_function_signature(node),
                "doc": ast.get_docstring(node),
            }
            collector = CallCollector()
            collector.visit(node)
            func_info["calls"] = collector.calls
            if include_source:
                func_info["source_code"] = extract_function_source(file_path, node.lineno, node.name)
            codemap["functions"].append(func_info)
        elif isinstance(node, ast.ClassDef):
            class_info = {
                "name": node.name,
                "lineno": node.lineno,
                "doc": ast.get_docstring(node),
                "methods": []
            }
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_info = {
                        "name": item.name,
                        "lineno": item.lineno,
                        "signature": get_function_signature(item),
                        "doc": ast.get_docstring(item),
                    }
                    collector = CallCollector()
                    collector.visit(item)
                    method_info["calls"] = collector.calls
                    if include_source:
                        method_info["source_code"] = extract_function_source(file_path, item.lineno, item.name)
                    class_info["methods"].append(method_info)
            codemap["classes"].append(class_info)
    return codemap

def build_directory_codemap(target_script, include_source=False):
    """
    Walk through the directory (and subdirectories) of the target script and build a list of code maps
    for every Python file found, excluding files that match the patterns in the repository's .gitignore.
    Also, skip this analyzer’s own file.
    """
    directory = os.path.dirname(os.path.abspath(target_script))
    
    # If pathspec is available, try to load the .gitignore from the repository root (assumed to be 'directory').
    gitignore_spec = None
    if pathspec is not None:
        gitignore_file = os.path.join(directory, ".gitignore")
        if os.path.exists(gitignore_file):
            try:
                with open(gitignore_file, "r", encoding="utf-8") as f:
                    gitignore_lines = f.read().splitlines()
                gitignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", gitignore_lines)
            except Exception as e:
                print(f"Error reading .gitignore: {e}")

    codemaps = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                # Skip this analyzer's own file.
                if os.path.abspath(full_path) == SELF_PATH:
                    print(f"Skipping self file: {full_path}")
                    continue
                # Compute the relative path from the repository root.
                relative_path = os.path.relpath(full_path, directory)
                if gitignore_spec and gitignore_spec.match_file(relative_path):
                    print(f"Skipping {relative_path} (matches .gitignore)")
                    continue
                print(f"Mapping file: {full_path}")
                codemap = build_codemap(full_path, include_source=include_source)
                codemaps.append(codemap)
    return codemaps

# =============================================================================
# Snapshot functions (consolidated variable examples)
# =============================================================================

def sanitize_for_json(obj, seen=None):
    """
    Recursively sanitize an object to be JSON-serializable.
    """
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return repr(obj)
    seen.add(obj_id)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    elif isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            try:
                new_key = str(key)
            except Exception:
                new_key = repr(key)
            new_dict[new_key] = sanitize_for_json(value, seen)
        return new_dict
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item, seen) for item in obj]
    elif isinstance(obj, set):
        return [sanitize_for_json(item, seen) for item in obj]
    else:
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return repr(obj)

def safe_snapshot_value(value):
    """
    Try to convert a value for snapshotting; if not possible, return its repr().
    """
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)

def snapshot_main_process():
    """
    Traverse all threads in the main process and build a consolidated mapping:
      variable name -> {"example": <example value>, "functions": [list of function names where it appears]}.
    Frames whose code originates from this analyzer script (SELF_PATH) are skipped.
    """
    snapshot_vars = {}
    frames = sys._current_frames()
    for tid, frm in frames.items():
        current = frm
        while current:
            try:
                frame_file = os.path.abspath(current.f_code.co_filename)
            except Exception:
                frame_file = None
            if frame_file == SELF_PATH:
                current = current.f_back
                continue
            func_name = current.f_code.co_name
            for var, value in current.f_locals.items():
                if var in ["args", "snapshot_vars"]:
                    continue
                try:
                    example = safe_snapshot_value(value[:LIMIT])
                except Exception:
                    try:
                        example = safe_snapshot_value(value[0][:LIMIT])
                    except Exception:
                        example = safe_snapshot_value(value)
                if var in snapshot_vars:
                    snapshot_vars[var]["functions"].add(func_name)
                else:
                    snapshot_vars[var] = {"example": example, "functions": set([func_name])}
            current = current.f_back
    # Convert sets to lists for JSON serialization.
    for var in snapshot_vars:
        snapshot_vars[var]["functions"] = list(snapshot_vars[var]["functions"])
    return snapshot_vars

def snapshot_signal_handler(signum, frame):
    """
    Signal handler for worker processes. It traverses local variables in all frames and writes a snapshot.
    The snapshot maps variable names to {"example": <example>, "functions": [list of function names]}.
    Frames whose code originates from this analyzer script (SELF_PATH) are skipped.
    """
    snapshot_vars = {}
    frames = sys._current_frames()
    for tid, frm in frames.items():
        current = frm
        while current:
            try:
                frame_file = os.path.abspath(current.f_code.co_filename)
            except Exception:
                frame_file = None
            if frame_file == SELF_PATH:
                current = current.f_back
                continue
            func_name = current.f_code.co_name
            for var, value in current.f_locals.items():
                if var in ["args", "snapshot_vars"]:
                    continue
                try:
                    example = safe_snapshot_value(value[:LIMIT])
                except Exception:
                    try:
                        example = safe_snapshot_value(value[0][:LIMIT])
                    except Exception:
                        example = safe_snapshot_value(value)
                if var in snapshot_vars:
                    snapshot_vars[var]["functions"].add(func_name)
                else:
                    snapshot_vars[var] = {"example": example, "functions": set([func_name])}
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

# --- Monkey-patch ProcessPoolExecutor so workers install our snapshot initializer ---
OriginalProcessPoolExecutor = concurrent.futures.ProcessPoolExecutor

def process_pool_initializer():
    signal.signal(signal.SIGUSR1, snapshot_signal_handler)

class PatchedProcessPoolExecutor(OriginalProcessPoolExecutor):
    def __init__(self, *args, **kwargs):
        if 'initializer' not in kwargs or kwargs['initializer'] is None:
            kwargs['initializer'] = process_pool_initializer
        super().__init__(*args, **kwargs)

concurrent.futures.ProcessPoolExecutor = PatchedProcessPoolExecutor

def trigger_worker_snapshots():
    """
    Use psutil to send SIGUSR1 to all child processes (workers).
    """
    current_process = psutil.Process(os.getpid())
    children = current_process.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGUSR1)
            print(f"Sent SIGUSR1 to worker process {child.pid}")
        except Exception as e:
            print(f"Failed to send SIGUSR1 to process {child.pid}: {e}")

def collect_worker_snapshots():
    """
    Look for snapshot files from workers, load them, and return a dictionary mapping
    each worker’s PID to its snapshot.
    """
    time.sleep(2)  # Give workers time to write snapshots.
    worker_snapshots = {}
    for filename in glob.glob("snapshot_vars_worker_*.json"):
        try:
            with open(filename, "r") as f:
                snapshot = json.load(f)
            base = os.path.basename(filename)
            parts = base.split("_")
            pid_part = parts[-1]
            pid_str = pid_part.split(".")[0]
            worker_snapshots[pid_str] = snapshot
            os.remove(filename)
        except Exception as e:
            print(f"Error reading worker snapshot from {filename}: {e}")
    return worker_snapshots

def merge_snapshots(main_snapshot, worker_snapshots):
    """
    Merge the main process snapshot with all worker snapshots.
    For each variable, union the lists of functions.
    """
    consolidated = dict(main_snapshot)
    for worker, snapshot in worker_snapshots.items():
        for var, info in snapshot.items():
            if var in consolidated:
                consolidated[var]["functions"] = list(set(consolidated[var]["functions"]).union(set(info["functions"])))
                # Optionally, keep the main snapshot's example.
            else:
                consolidated[var] = info
    return consolidated

def run_target_program(file_path):
    """
    Run the target Python file. (Any ProcessPoolExecutors created therein will use our patched initializer.)
    """
    target_dir = os.path.dirname(os.path.abspath(file_path))
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)
    runpy.run_path(file_path, run_name="__main__")

def run_with_snapshot(file_path, timeout=10):
    """
    Run the target program in a separate thread and, after a delay, take snapshots of the main process
    and any worker processes. Then merge them into a consolidated snapshot mapping variable names to
    an example value and a list of function names where the variable appears.
    """
    
    target_thread = threading.Thread(target=run_target_program, args=(file_path,))
    target_thread.start()

    print(f"Waiting {timeout} seconds before taking a snapshot...")
    time.sleep(timeout)

    main_snapshot = snapshot_main_process()
    print("Main process snapshot taken.")

    trigger_worker_snapshots()
    worker_snapshots = collect_worker_snapshots()

    consolidated_snapshot = merge_snapshots(main_snapshot, worker_snapshots)
    target_thread.join()
    return consolidated_snapshot

# =============================================================================
# Filtering functions for codemap and variable examples
# =============================================================================

def filter_codemap(codemaps, file_filters=None, name_filters=None):
    """
    Given a list of codemap entries, filter them according to:
      - file_filters: only include a file if its 'file' path contains one of the specified substrings.
      - name_filters: for each file, only include functions and classes (and their methods) whose names
                      contain one of the specified substrings.
    A file is included if either its filename matches a file filter or it contains at least one matching
    function or class.
    """
    filtered_codemap = []
    for entry in codemaps:
        pwd = os.getcwd()
        filtered_functions = []
        filtered_classes = []
        file_filters = [os.path.abspath(os.path.join(pwd, filt)) for filt in file_filters] if file_filters else []
        file_path = entry.get("file", "")
        # Determine if the file name matches any file filter (if provided)
        file_matches = False
        if file_filters:
            file_matches = any(filt.lower() in file_path.lower() for filt in file_filters)

        # If name_filters is provided, filter functions and classes accordingly.
        if name_filters:
            filtered_functions = [f for f in entry.get("functions", [])
                                  if any(nf.lower() in f.get("name", "").lower() for nf in name_filters)]
            filtered_classes = []
            for cls in entry.get("classes", []):
                class_matches = any(nf.lower() in cls.get("name", "").lower() for nf in name_filters)
                methods = cls.get("methods", [])
                filtered_methods = [m for m in methods
                                    if any(nf.lower() in m.get("name", "").lower() for nf in name_filters)]
                if class_matches or filtered_methods:
                    new_cls = cls.copy()
                    new_cls["methods"] = filtered_methods
                    filtered_classes.append(new_cls)
        else:
            if file_matches:
                filtered_functions = entry.get("functions", [])
                filtered_classes = entry.get("classes", [])

        # Decide whether to include this file:
        # Include if the file matches OR if any functions/classes matched the name filter.
        if file_matches or filtered_functions or filtered_classes:
            new_entry = entry.copy()
            new_entry["functions"] = filtered_functions
            new_entry["classes"] = filtered_classes
            filtered_codemap.append(new_entry)
    return filtered_codemap

def filter_variable_examples(variable_examples, name_filters=None):
    """
    Given a mapping of variable names to their snapshot info (which includes a list of function names),
    filter the mapping to only include variables that appear in at least one function whose name contains
    one of the specified substrings (if name_filters is provided).
    """
    if not name_filters:
        return variable_examples
    filtered = {}
    for var, info in variable_examples.items():
        funcs = info.get("functions", [])
        filtered_funcs = [fn for fn in funcs if any(nf.lower() in fn.lower() for nf in name_filters)]
        if filtered_funcs:
            filtered[var] = {"example": info["example"], "functions": filtered_funcs}
    return filtered

# =============================================================================
# Main: combine directory codemap and consolidated variable snapshots into one JSON output
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Map out an entire Python codebase (from the target script’s directory) and run the target script to capture example variable values."
    )
    parser.add_argument("script", help="Path to the Python script to analyze and run")
    parser.add_argument("output", help="Output JSON file for the codemap and variable examples")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Timeout in seconds before taking the snapshot (default: 10)")
    parser.add_argument("--include-source", action="store_true", default=False,
                        help="Include source code snippet for each function/method (default: off)")
    parser.add_argument("--filter-file", nargs='+',
                        help="Only include files whose path contains one or more of the given substrings")
    parser.add_argument("--filter-name", nargs='+',
                        help="Only include functions, classes, or methods whose name contains one or more of the given substrings")
    args = parser.parse_args()

    script_path = os.path.abspath(args.script)

    if not os.path.exists(script_path):
        print(f"Script {script_path} does not exist.")
        sys.exit(1)

    print("Building directory code map...")
    directory_codemap = build_directory_codemap(script_path, include_source=args.include_source)

    # Apply filtering to the codemap if requested.
    if args.filter_file or args.filter_name:
        directory_codemap = filter_codemap(directory_codemap,
                                           file_filters=args.filter_file,
                                           name_filters=args.filter_name)

    print("Running target script to capture variable examples...")
    variable_snapshot = run_with_snapshot(script_path, args.timeout)

    # Apply filtering to variable examples if a name filter is provided.
    if args.filter_name:
        variable_snapshot = filter_variable_examples(variable_snapshot, name_filters=args.filter_name)

    output_data = {
        "codemap": directory_codemap,
        "variable_examples": variable_snapshot
    }

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"Output written to {args.output}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()