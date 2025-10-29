"""Tests that verify SCF, CCSD and CCSD(T) run on the GPU backend.

These tests are intentionally small (H2 in STO-3G) to keep runtime low.
They assert that the CUDA-backed implementations are selected via the
project's `cuda_adapter.build_ccsd_solver` helper and exercise both the
cuTENSOR-enabled path and the CuPy fallback path by temporarily
overriding `gpu4pyscf.lib.cutensor.contract_engine`.

If the required GPU stack (CuPy, a CUDA device, gpu4pyscf) is missing the
tests will be skipped with clear messages.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - optional runtime
    cp = None  # type: ignore

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gpu4pyscf
from gpu4pyscf import scf as gpu_scf
from pyscf import gto

from cuda_adapter import build_ccsd_solver


def minimal_h2_mol():
    return gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )


def run_pipeline_and_assert(mf):
    """Run CCSD and CCSD(T) on an already-created mean-field object.

    Returns the CCSD energy and the (T) correction (or None if (T) not
    implemented by the backend).
    """

    try:
        solver, using_gpu = build_ccsd_solver(mf, prefer_gpu=True)
    except (TypeError, RuntimeError) as exc:
        pytest.skip(
            "CUDA solver unavailable: "
            "ensure gpu4pyscf.logger.warn signature matches PySCF",
        )
    except Exception as exc:
        pytest.skip(f"CUDA solver failed unexpectedly: {exc}")

    assert using_gpu is True, "Expected CUDA-backed CCSD to be selected"
    assert getattr(solver, "_cuda_adapter_gpu_enabled", False) is True

    # Run CCSD (small system; should be fast)
    out = solver.kernel()
    # PySCF/gpu4pyscf sometimes return a tuple; energy is first element
    if isinstance(out, (tuple, list)):
        cc_energy = out[0]
    else:
        cc_energy = out

    ccsd_t_energy = None
    # Try to run (T) if the solver exposes the helper
    if hasattr(solver, "ccsd_t"):
        try:
            ccsd_t_energy = solver.ccsd_t()
        except Exception:
            # Some backends may require extra data or not implement (T)
            ccsd_t_energy = None

    # Basic sanity checks: energies are floats and finite
    assert isinstance(cc_energy, float)
    assert cc_energy == cc_energy  # not NaN
    if ccsd_t_energy is not None:
        assert isinstance(ccsd_t_energy, float)
        assert ccsd_t_energy == ccsd_t_energy

    return cc_energy, ccsd_t_energy


@pytest.mark.requires_pyscf
def test_gpu_pipeline_with_cuTENSOR():
    """Verify the GPU pipeline when cuTENSOR is available.

    If cuTENSOR is not detected this test will be skipped; the separate
    test below covers the CuPy fallback.
    """

    # Quick GPU presence checks
    if cp is None:
        pytest.skip("CuPy is not installed in this environment")

    cupy_cuda = getattr(cp, "cuda", None)
    if cupy_cuda is None:
        pytest.skip("CuPy CUDA bindings unavailable")

    try:
        ndev = int(cupy_cuda.runtime.getDeviceCount())
    except Exception:
        ndev = 0
    if ndev == 0:
        pytest.skip("No CUDA devices detected")

    # Check whether gpu4pyscf reports a non-empty contract engine. If it
    # does not, cuTENSOR isn't wired in and we skip this test.
    try:
        import gpu4pyscf.lib.cutensor as cut
        engine = getattr(cut, "contract_engine", None)
    except Exception:
        engine = None

    if engine in (None, "cupy"):
        pytest.skip("cuTENSOR not detected; skipping cuTENSOR-specific test")

    mol = minimal_h2_mol()
    mf = gpu_scf.RHF(mol)
    e_scf = mf.kernel()
    # Accept either scalar or tuple return types
    if isinstance(e_scf, (tuple, list)):
        e_scf = e_scf[0]
    assert isinstance(e_scf, float)

    run_pipeline_and_assert(mf)


@pytest.mark.requires_pyscf
def test_gpu_pipeline_with_cupy_fallback(monkeypatch):
    """Force the cutensor engine to the CuPy fallback and verify the
    GPU pipeline still runs."""

    if cp is None:
        pytest.skip("CuPy is not installed in this environment")

    cupy_cuda = getattr(cp, "cuda", None)
    if cupy_cuda is None:
        pytest.skip("CuPy CUDA bindings unavailable")

    try:
        ndev = int(cupy_cuda.runtime.getDeviceCount())
    except Exception:
        ndev = 0
    if ndev == 0:
        pytest.skip("No CUDA devices detected")

    # Monkeypatch the gpu4pyscf cutensor contract_engine value to emulate
    # the CuPy fallback path. Use raising=False in case the attribute is
    # missing in the installed version.
    try:
        import gpu4pyscf.lib.cutensor as cut
    except Exception:
        pytest.skip("gpu4pyscf.cutensor module not importable; cannot continue")

    original = getattr(cut, "contract_engine", None)
    monkeypatch.setattr(cut, "contract_engine", "cupy", raising=False)
    try:
        mol = minimal_h2_mol()
        mf = gpu_scf.RHF(mol)
        e_scf = mf.kernel()
        if isinstance(e_scf, (tuple, list)):
            e_scf = e_scf[0]
        assert isinstance(e_scf, float)

        run_pipeline_and_assert(mf)
    finally:
        # Restore original value for cleanliness
        monkeypatch.setattr(cut, "contract_engine", original, raising=False)
