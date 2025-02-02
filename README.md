# Code Analyzer and Snapshot Tool

## Overview

**Code Analyzer and Snapshot Tool** is a Python utility that performs two primary tasks:

1. **Code Mapping**:  
   Recursively scans the directory (and subdirectories) of a given target script to generate a "codemap" of all Python files. The codemap includes details of module-level functions, classes, and their methods. Optionally, it can also include source code snippets for each function or method.

2. **Variable Snapshotting**:  
   Runs the target Python script to capture example variable values from active stack frames in both the main process and any worker processes (spawned via `ProcessPoolExecutor`). The snapshot collects a sample value and records the function names where each variable appears.

The final output is a JSON file containing two keys:
- **codemap**: A list of code maps (one per Python file).
- **variable_examples**: A consolidated mapping of variable names to an example value and a list of function names where the variable is used.

## Features

- **AST-Based Code Mapping**:  
  Uses Python's `ast` module to extract function signatures, docstrings, and function calls from each Python file.

- **Optional Source Code Extraction**:  
  Include the actual source code snippets for functions/methods with the `--include-source` flag.

- **.gitignore Support**:  
  Skips files that match patterns defined in a `.gitignore` file (if the optional `pathspec` module is installed).

- **Variable Snapshotting**:  
  Captures runtime snapshots of variable values from the main process and from worker processes via signal handling.

- **Worker Process Integration**:  
  Automatically patches `ProcessPoolExecutor` so that worker processes install a snapshot signal handler.

- **Filtering Options**:  
  Filter the codemap and variable snapshots by file paths (`--filter-file`) or by function/class/method names (`--filter-name`).

## Prerequisites

- **Python 3**  
  Ensure you are running Python 3.

- **psutil**  
  Required for process management. Install with:
  ```bash
  pip install psutil
  ```

- **pathspec (Optional)**  
  Enables `.gitignore` support. Install with:
  ```bash
  pip install pathspec
  ```

## Installation

1. Clone or download the repository containing the `codeanalyser.py` script.
2. Install the required dependencies:
   ```bash
   pip install psutil
   # Optionally, if you want .gitignore support:
   pip install pathspec
   ```

## Usage

Run the script from the command line using the following syntax:

```bash
./codeanalyser.py path/to/target_script.py output.json [--timeout 10] [--include-source] [--filter-file substring [substring ...]] [--filter-name substring [substring ...]]
```

### Arguments

- **script**:  
  Path to the Python script you want to analyze and run.

- **output**:  
  Path to the JSON file that will store the codemap and variable examples.

- **--timeout**:  
  *(Optional)* Time in seconds to wait before taking the variable snapshot (default is 10 seconds).

- **--include-source**:  
  *(Optional)* If specified, includes source code snippets for each function/method in the codemap.

- **--filter-file**:  
  *(Optional)* Only include files whose paths contain one or more specified substrings.

- **--filter-name**:  
  *(Optional)* Only include functions, classes, or methods whose names contain one or more specified substrings.

## How It Works

1. **Building the Codemap**:
   - The script recursively walks through the target script’s directory.
   - It processes each Python file (excluding the analyzer’s own file and any that match `.gitignore` patterns, if available).
   - For each file, it uses the `ast` module to extract details like function names, signatures, docstrings, and function calls.
   - Optionally, it extracts the source code snippet for each function/method.

2. **Capturing Variable Snapshots**:
   - The target script is executed in a separate thread.
   - After the specified timeout, the tool traverses all active stack frames in the main process to capture local variable states.
   - It sends a signal (`SIGUSR1`) to any worker processes (spawned via `ProcessPoolExecutor`), prompting them to write their own snapshots.
   - The main and worker snapshots are merged into a consolidated mapping.

3. **Output**:
   - The final JSON output includes:
     - **codemap**: The detailed code structure of the scanned Python files.
     - **variable_examples**: A mapping of variable names to an example value and the list of functions where they appear.

## Customization and Extension

- **Source Code Inclusion**:  
  Use the `--include-source` flag if you want to include function/method source snippets in the codemap.

- **Filtering**:  
  Apply `--filter-file` and/or `--filter-name` to restrict the analysis to specific files or code elements.

- **Timeout Adjustment**:  
  Adjust the `--timeout` parameter to allow enough time for the target script and its worker processes to run and generate snapshots.

## Troubleshooting

- **Parsing Errors**:  
  If you encounter issues reading or parsing files, check the console output for error messages.

- **Snapshot Issues**:  
  If worker snapshots are not generated, ensure that worker processes are correctly spawned and that the process has permission to send signals.

- **Missing Modules**:  
  The script will work without `pathspec`, but without it, `.gitignore` filtering will not be applied.

<!-- ## License

Will do in a bit. -->

## Acknowledgements

- Utilizes Python’s built-in `ast` module for static code analysis.
- Leverages `psutil` for process management.
- Inspired by various tools and scripts designed for code introspection and debugging.

---

*This tool is designed to help developers gain insights into the structure and runtime behavior of Python codebases, facilitating debugging, analysis, and documentation efforts.*