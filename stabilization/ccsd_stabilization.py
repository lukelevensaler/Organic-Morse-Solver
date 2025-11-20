"""Single-shot CCSD stabilization helpers.

Provides a utility for running one relaxed CCSD attempt with
user-controllable convergence parameters. This keeps the main workflow
simple while still exposing knobs for difficult systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from tqdm import tqdm

from pyscf import cc


@dataclass
class CCSDRunResult:
    """Container for the outcome of a stabilized CCSD run."""

    solver: Optional[Any]
    converged: bool
    error: Optional[Exception]
    iterations: Optional[int]
    residual: Optional[float]

    def log_diagnostics(self) -> None:
        """Print basic diagnostics when convergence fails."""
        if self.converged or self.solver is None:
            return
        if self.iterations is not None:
            print(f"    Iterations completed before abort: {self.iterations}")
        if self.residual is not None:
            print(f"    Final RMS residual: {self.residual}")
        if self.error is not None:
            print(f"    Reported CCSD error: {self.error}")


def run_stabilized_ccsd(
    mf,
    *,
    conv_tol: float = 1e-3,
    conv_tol_normt_scale: float = 0.1,
    max_cycle: int = 200,
    diis_space: int = 12,
    diis_start: int = 1,
    level_shift: float = 0.2,
    direct: bool = True,
    desc: str = "Stabilized CCSD Attempt",
    colour: str = "green",
) -> CCSDRunResult:
    """Run a single stabilized CCSD calculation with relaxed settings.

    Parameters are intentionally forgiving so that difficult systems get a
    fair chance to converge without needing a ladder of progressively
    tighter runs.
    """

    mycc = cc.CCSD(mf)
    mycc.conv_tol = conv_tol
    if hasattr(mycc, "conv_tol_normt"):
        mycc.conv_tol_normt = conv_tol * conv_tol_normt_scale
    mycc.max_cycle = max_cycle
    if hasattr(mycc, "diis_space"):
        mycc.diis_space = diis_space
    if hasattr(mycc, "diis_start_cycle"):
        mycc.diis_start_cycle = diis_start
    if hasattr(mycc, "direct"):
        mycc.direct = bool(direct)
    setattr(mycc, "level_shift", level_shift)

    try:
        with tqdm(desc=desc, unit="step", colour=colour) as pbar:
            pbar.set_postfix(tol=f"{conv_tol:.0e}", max_cycle=max_cycle)
            mycc.run()
            pbar.update(1)
            corr_energy = getattr(mycc, "e_corr", float("nan"))
            pbar.set_postfix(converged=mycc.converged, corr_energy=f"{corr_energy:.6f}")
    except Exception as exc:
        return CCSDRunResult(
            solver=None,
            converged=False,
            error=exc,
            iterations=None,
            residual=None,
        )

    iterations = getattr(mycc, "iteration", None)
    residual = getattr(mycc, "rms", None)
    return CCSDRunResult(
        solver=mycc,
        converged=bool(mycc.converged),
        error=None,
        iterations=iterations,
        residual=residual,
    )
