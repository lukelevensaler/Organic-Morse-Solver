"""Helpers for CCSD(T) triples corrections with progress reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tqdm import tqdm


@dataclass
class CCSDTResult:
    """Container describing the outcome of a CCSD(T) triples correction."""

    available: bool
    energy: Optional[float]
    error: Optional[Exception]

    def log_status(self) -> None:
        """Emit human-readable diagnostics when triples are unavailable."""
        if self.error is not None:
            print(f"⚠️  CCSD(T) triples correction failed: {self.error}")
        elif not self.available:
            print("⚠️  CCSD(T) triples correction not available - using CCSD dipole")


def compute_triples_correction(
    mycc,
    *,
    desc: str = "CCSD(T) Triples Correction",
    colour: str = "red",
) -> CCSDTResult:
    """Attempt a CCSD(T) triples correction with a progress indicator."""

    with tqdm(desc=desc, unit="step", colour=colour) as pbar:
        try:
            if hasattr(mycc, "ccsd_t") and getattr(mycc, "converged", False):
                pbar.set_postfix(status="computing")
                energy = mycc.ccsd_t()
                pbar.update(1)
                pbar.set_postfix(E_T=f"{energy:.8f}")
                print(f"✅ CCSD(T) triples correction: {energy:.8f} Hartree")
                return CCSDTResult(available=True, energy=energy, error=None)

            pbar.update(1)
            pbar.set_postfix(status="skipped")
            return CCSDTResult(available=False, energy=None, error=None)

        except Exception as exc:
            pbar.update(1)
            pbar.set_postfix(status="failed")
            return CCSDTResult(available=False, energy=None, error=exc)
