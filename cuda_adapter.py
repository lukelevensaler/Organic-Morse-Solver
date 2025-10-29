"""Utilities for routing high-cost computations through a CUDA backend.

This module centralises all optional CUDA logic so that call sites can
keep their PySCF-facing code simple while still taking advantage of a
GPU when available.  Each helper returns clear fallbacks when the CUDA
stack is not present, allowing the rest of the codebase to degrade
gracefully on CPU-only machines.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

logger = logging.getLogger(__name__)


try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from gpu4pyscf import cc as gpu_cc  # type: ignore
    from gpu4pyscf.cc import ccsd_incore as gpu_ccsd_incore  # type: ignore
except Exception:  # pragma: no cover
    gpu_cc = None  # type: ignore[assignment]
    gpu_ccsd_incore = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from gpu4pyscf import scf as gpu_scf  # type: ignore
except Exception:  # pragma: no cover
    gpu_scf = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from gpu4pyscf import grad as gpu_grad  # type: ignore
except Exception:  # pragma: no cover
    gpu_grad = None  # type: ignore[assignment]

try:
    from pyscf import cc as cpu_cc
except Exception:  # pragma: no cover - PySCF is a hard dependency elsewhere
    cpu_cc = None  # type: ignore[assignment]

try:
    from pyscf import scf as cpu_scf
except Exception:  # pragma: no cover - PySCF is a hard dependency elsewhere
    cpu_scf = None  # type: ignore[assignment]

try:
    from pyscf import grad as pyscf_grad
except Exception:  # pragma: no cover - PySCF is a hard dependency elsewhere
    pyscf_grad = None  # type: ignore[assignment]

# Initialize cuTENSOR if available
try:
    import gpu4pyscf.lib.cutensor as cutensor_lib
    if hasattr(cutensor_lib, 'cutensor_backend') and cutensor_lib.cutensor_backend is not None:
        cutensor_lib.contract_engine = 'cutensor'
        logger.info("cuTENSOR enabled for gpu4pyscf tensor operations")
    else:
        logger.info("cuTENSOR backend not available, using CuPy fallback")
except Exception as e:
    logger.info(f"cuTENSOR initialization failed: {e}")


def cupy_device_count() -> int:
    """Return detected CUDA device count, treating any failure as zero."""

    if cp is None:
        return 0
    try:
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def cuda_available() -> bool:
    """Return True when CuPy sees at least one CUDA-capable device."""

    return cupy_device_count() > 0


def build_ccsd_solver(
    mf: Any,
    *,
    prefer_gpu: bool = True,
    cpu_module: Any | None = None,
) -> Tuple[Any, bool]:
    """Create a CCSD solver, preferring a CUDA-backed implementation.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        Mean-field object produced by the SCF step.
    prefer_gpu : bool
        Toggle that allows callers to opt out of GPU use while retaining
        a single code path for instantiation.
    cpu_module : module | None
        Optional PySCF ``cc`` module override.  When omitted, the helper
        imports ``pyscf.cc`` itself.

    Returns
    -------
    tuple[Any, bool]
        The CCSD solver instance and a flag indicating whether it is
        using the CUDA backend.
    """

    solver = None
    using_gpu = False

    if prefer_gpu and gpu_cc is not None and cuda_available():
        try:
            cc_factory = getattr(gpu_cc, "CCSD", None)
            if cc_factory is None and gpu_ccsd_incore is not None:
                cc_factory = getattr(gpu_ccsd_incore, "CCSD", None)
            if cc_factory is None:
                raise AttributeError("gpu4pyscf.cc does not expose CCSD")
            solver = cc_factory(mf)
            using_gpu = True
            setattr(solver, "_cuda_adapter_backend", "gpu")
            try:
                device = cp.cuda.Device() if cp is not None else None
                setattr(solver, "_cuda_adapter_device", getattr(device, "id", None))
            except Exception:
                setattr(solver, "_cuda_adapter_device", None)
        except Exception as e:
            solver = None
            using_gpu = False
            logger.debug("Falling back to CPU CCSD after GPU failure: %s", e)

    if solver is None:
        module = cpu_module or cpu_cc
        if module is None:  # pragma: no cover - PySCF import guard
            raise RuntimeError(
                "PySCF CC module is unavailable; cannot construct a CCSD solver."
            )
        cc_factory = getattr(module, "CCSD", None)
        if cc_factory is None:
            raise AttributeError("PySCF CC module does not expose CCSD")
        solver = cc_factory(mf)
        setattr(solver, "_cuda_adapter_backend", "cpu")
        setattr(solver, "_cuda_adapter_device", None)

    setattr(solver, "_cuda_adapter_gpu_enabled", using_gpu)
    return solver, using_gpu


def build_scf_solver(
    mol: Any,
    *,
    spin: int = 0,
    prefer_gpu: bool = True,
) -> Tuple[Any, bool]:
    """Create a SCF solver, preferring a CUDA-backed implementation.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object for the SCF calculation.
    spin : int
        Spin multiplicity (0 for RHF, non-zero for UHF).
    prefer_gpu : bool
        Toggle that allows callers to opt out of GPU use while retaining
        a single code path for instantiation.

    Returns
    -------
    tuple[Any, bool]
        The SCF solver instance and a flag indicating whether it is
        using the CUDA backend.
    """

    solver = None
    using_gpu = False

    if prefer_gpu and gpu_scf is not None and cuda_available():
        try:
            if spin == 0:
                scf_factory = getattr(gpu_scf, "RHF", None)
            else:
                scf_factory = getattr(gpu_scf, "UHF", None)
            if scf_factory is None:
                raise AttributeError(f"gpu4pyscf.scf does not expose {'RHF' if spin == 0 else 'UHF'}")
            solver = scf_factory(mol)
            using_gpu = True
            setattr(solver, "_cuda_adapter_backend", "gpu")
            try:
                device = cp.cuda.Device() if cp is not None else None
                setattr(solver, "_cuda_adapter_device", getattr(device, "id", None))
            except Exception:
                setattr(solver, "_cuda_adapter_device", None)
        except Exception as e:
            solver = None
            using_gpu = False
            logger.debug("Falling back to CPU SCF after GPU failure: %s", e)

    if solver is None:
        if spin == 0:
            scf_factory = getattr(cpu_scf, "RHF", None)
        else:
            scf_factory = getattr(cpu_scf, "UHF", None)
        if scf_factory is None:
            raise AttributeError(f"PySCF SCF module does not expose {'RHF' if spin == 0 else 'UHF'}")
        solver = scf_factory(mol)
        setattr(solver, "_cuda_adapter_backend", "cpu")
        setattr(solver, "_cuda_adapter_device", None)

    setattr(solver, "_cuda_adapter_gpu_enabled", using_gpu)
    return solver, using_gpu


def build_grad_solver(
    mf: Any,
    *,
    spin: int = 0,
    prefer_gpu: bool = True,
) -> Any:
    """Create a gradient solver, preferring a CUDA-backed implementation.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        Mean-field object produced by the SCF step.
    spin : int
        Spin multiplicity (0 for RHF, non-zero for UHF).
    prefer_gpu : bool
        Toggle that allows callers to opt out of GPU use while retaining
        a single code path for instantiation.

    Returns
    -------
    Any
        The gradient solver instance.
    """

    solver = None

    if prefer_gpu and gpu_grad is not None:
        try:
            if spin == 0:
                grad_factory = getattr(gpu_grad, "RHF", None)
            else:
                grad_factory = getattr(gpu_grad, "UHF", None)
            if grad_factory is None:
                raise AttributeError(f"gpu4pyscf.grad does not expose {'RHF' if spin == 0 else 'UHF'}")
            solver = grad_factory(mf)
            setattr(solver, "_cuda_adapter_backend", "gpu")
            try:
                device = cp.cuda.Device() if cp is not None else None
                setattr(solver, "_cuda_adapter_device", getattr(device, "id", None))
            except Exception:
                setattr(solver, "_cuda_adapter_device", None)
        except Exception as e:
            solver = None
            logger.debug("Falling back to CPU grad after GPU failure: %s", e)

    if solver is None:
        if pyscf_grad is None:  # pragma: no cover - PySCF import guard
            raise RuntimeError("PySCF grad module is unavailable; cannot construct a gradient solver.")
        if spin == 0:
            grad_factory = getattr(pyscf_grad, "RHF", None)
        else:
            grad_factory = getattr(pyscf_grad, "UHF", None)
        if grad_factory is None:
            raise AttributeError(f"PySCF grad module does not expose {'RHF' if spin == 0 else 'UHF'}")
        solver = pyscf_grad.RHF(mf) if spin == 0 else pyscf_grad.UHF(mf)
        setattr(solver, "_cuda_adapter_backend", "cpu")
        setattr(solver, "_cuda_adapter_device", None)

    return solver


def describe_cuda_backend() -> str:
    """Return a short human-readable description of the CUDA status."""

    if not cuda_available():
        return "CUDA backend unavailable"
    if cp is None:
        return "CuPy runtime missing"
    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        name_obj = props.get("name", b"CUDA device")
        if isinstance(name_obj, bytes):
            name = name_obj.decode(errors="ignore")
        else:
            name = str(name_obj)
        return f"CUDA backend active on device {device.id} ({name})"
    except Exception:
        return "CUDA backend active"


def run_cuda_computation(data: Any) -> Any:
    """Demo helper kept for tests: run a CuPy FFT when possible."""

    if cp is None:
        raise RuntimeError("CuPy is not available in the current environment")
    return cp.fft.fft(data)