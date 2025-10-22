"""Entry point for the Morse solver CLI.

This small script proxies command-line arguments to the Typer app defined in
`morse_solver` (or `morse_solver.compute` / `morse_solver.main_morse_solver`).

Usage examples:
    python run_morse.py stretches
    python run_morse.py compute --help

The script locates the Typer app by importing `morse_solver` and looking for a
module-level `app` variable. If not found, it will try `morse_solver.compute`.
"""
import sys
from importlib import import_module
import os

MODULE_NAMES = [
    "morse_solver.cli",
    "morse_solver.main_morse_solver",
    "morse_solver.derivatives_dipole_moment",
    "morse_solver.run_morse_model",
    "morse_solver.high_precision_arithmetic",
    "morse_solver.pyscf_monkeypatch",
    "morse_solver.normalize_bonds"
]

# If this script is executed with CWD inside the `morse_solver` package
# directory (i.e. CWD contains __init__.py), Python won't find the
# top-level package `morse_solver` unless the parent directory is on
# sys.path. Add the parent directory in that case so import_module
# can import `morse_solver.*` successfully.
cwd = os.getcwd()
if os.path.exists(os.path.join(cwd, "__init__.py")):
    parent = os.path.dirname(cwd)
    if parent not in sys.path:
        sys.path.insert(0, parent)

app = None
for mod_name in MODULE_NAMES:
    try:
        mod = import_module(mod_name)
        if hasattr(mod, "app"):
            app = getattr(mod, "app")
            break
    except Exception:
        continue

if app is None:
    print("Could not find a Typer `app` in morse_solver modules. Please ensure `morse_solver` exposes `app`.")
    sys.exit(2)

if __name__ == "__main__":
    # Forward all args to the Typer app in cli.py 
    # (which has a compute function that runs the whole stack)
    app()
