"""morse_solver package initializer.

Expose a top-level ``app`` variable when any of the package submodules
defines a Typer application called ``app``. Import failures are swallowed so
importing the package won't fail when optional runtime dependencies (e.g.
`typer` or `pyscf`) are not installed; callers can still inspect ``morse_solver.app``
to see if a Typer app is available.
"""
from importlib import import_module
from typing import Optional

MODULE_CANDIDATES = (
    "morse_solver.cli",
    "morse_solver.main_morse_solver",
    "morse_solver.derivatives_dipole_moment",
)

app: Optional[object] = None
for _mod in MODULE_CANDIDATES:
    try:
        m = import_module(_mod)
        if hasattr(m, "app"):
            app = getattr(m, "app")
            break
    except Exception:
        # intentionally ignore import errors from heavy optional deps
        continue

__all__ = ["app"]
