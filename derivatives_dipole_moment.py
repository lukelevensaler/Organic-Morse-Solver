import warnings
# Suppress pkg_resources deprecation warning from pyberny
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

from pyscf import gto, scf
import numpy as np
import os
from typing import Any
try:
    # geomopt Berny solver for correlated gradients
    from pyscf.geomopt import berny_solver
except Exception:
    berny_solver = None

# lazily import CC and grad inside functions to avoid static analysis issues
cc = None
grad = None
import threading
import time
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ===== INITIAL DIPOLE DERIVATIVE SOLVING WITH PySCF =======

def dipole_for_geometry(atom_string: str, spin: int, basis: str = "aug-cc-pV5Z ") -> np.ndarray:
	"""Return the molecular dipole vector (Debye) computed at mean-field level.

	This function lazily requires PySCF. It uses RHF for closed-shell (spin==0)
	and UHF for open-shell cases (spin>0).
	"""
	mol = gto.M(atom=atom_string, basis=basis, spin=spin)
	if spin == 0:
		mf = scf.RHF(mol).run()
	else:
		mf = scf.UHF(mol).run()
	# PySCF returns dipole in Debye
	return np.array(mf.dip_moment())


def compute_mu_derivatives(coords_string: str, specified_spin: int, delta: float = 0.01, basis: str = "aug-cc-pV5Z ", atom_index: int = 0, axis: int = 2, bond_pair: tuple[int, int] | None = None) -> tuple[float, float]:
    """
    Compute first and second derivatives of the dipole using finite differences.

    coords_string: multiline string of fully numeric coordinates (Element x y z)
    specified_spin: spin state for dipole calculation
    delta: displacement magnitude (Å)
    basis: quantum chemistry basis set
    atom_index: which atom to displace (0-based)
    axis: which Cartesian axis to displace (0=x, 1=y, 2=z)
    
    Returns (mu1, mu2) in SI units: mu1 in C·m/m, mu2 in C·m/m^2
    """
    # read coords
    if os.path.isfile(coords_string):
        with open(coords_string, 'r') as fh:
            coord_text = fh.read()
    else:
        coord_text = coords_string

    # parse lines into numpy array
    lines = [ln.strip() for ln in coord_text.splitlines() if ln.strip()]
    atoms = []
    positions = []
    for ln in lines:
        parts = ln.split()
        if len(parts) != 4:
            raise ValueError(f"Invalid coordinate line: '{ln}'. Expected 'Element x y z'.")
        atoms.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    positions = np.array(positions, dtype=float)

    # create displaced geometries along chosen axis
    pos0 = positions.copy()
    pos_plus = positions.copy()
    pos_minus = positions.copy()

    # If a bond_pair is provided, perform a symmetric bond stretch along the
    # normalized bond vector: move atom i by +delta/2 and atom j by -delta/2
    # (and vice-versa for the minus geometry). This produces a pure bond
    # stretching displacement independent of an explicit atom_index.
    if bond_pair is not None:
        i, j = bond_pair
        if not (0 <= i < positions.shape[0] and 0 <= j < positions.shape[0]):
            raise IndexError("bond_pair indices out of range of atom positions")
        bond_vec = positions[j] - positions[i]
        norm = np.linalg.norm(bond_vec)
        if norm == 0:
            raise ValueError("bond_pair atoms are at identical positions; cannot compute bond vector")
        unit = bond_vec / norm
        half = float(delta) / 2.0
        # stretch the bond: +half on i and -half on j for the + geometry
        pos_plus[i] += unit * half
        pos_plus[j] -= unit * half
        # inverse for the - geometry
        pos_minus[i] -= unit * half
        pos_minus[j] += unit * half
    else:
        # Cartesian axis displacement (default behavior)
        pos_plus[atom_index, axis] += delta
        pos_minus[atom_index, axis] -= delta

    # helper to build XYZ block string
    def block_from_positions(pos_array):
        return "\n".join(f"{atom} {x:.6f} {y:.6f} {z:.6f}" 
                         for atom, (x, y, z) in zip(atoms, pos_array))

    atom0 = block_from_positions(pos0)
    atom_plus = block_from_positions(pos_plus)
    atom_minus = block_from_positions(pos_minus)

    # compute dipoles
    mu0 = dipole_for_geometry(atom0, specified_spin, basis=basis)
    mu_plus = dipole_for_geometry(atom_plus, specified_spin, basis=basis)
    mu_minus = dipole_for_geometry(atom_minus, specified_spin, basis=basis)

    # Debug output for dipole moments
    print(f"Debug: Dipole at equilibrium: {mu0} Debye")
    print(f"Debug: Dipole at +δ: {mu_plus} Debye") 
    print(f"Debug: Dipole at -δ: {mu_minus} Debye")
    print(f"Debug: Displacement δ = {delta} Å")
    
    # finite-difference derivatives
    mu_prime_vec = (mu_plus - mu_minus) / (2.0 * delta)
    mu_double_vec = (mu_plus - 2.0 * mu0 + mu_minus) / (delta ** 2)
    
    print(f"Debug: First derivative vector (Debye/Å): {mu_prime_vec}")
    print(f"Debug: Second derivative vector (Debye/Å²): {mu_double_vec}")

    # convert from Debye/(Å^n) to C·m/(m^n)
    D_TO_CM = 3.33564e-30
    mu1_si = np.linalg.norm(mu_prime_vec) * (D_TO_CM / 1e-10)
    mu2_si = np.linalg.norm(mu_double_vec) * (D_TO_CM / 1e-20)
    
    print(f"Debug: |μ'(0)| = {mu1_si:.6e} C·m/m")
    print(f"Debug: |μ''(0)| = {mu2_si:.6e} C·m/m²")

    return float(mu1_si), float(mu2_si)


def optimize_geometry_ccsd(coords_string: str, specified_spin: int, basis: str = "aug-cc-pV5Z ", maxsteps: int = 50, use_hf_fallback: bool = True) -> str:
    """
    Run a CCSD geometry optimization (via CCSD gradients + Berny) where available.

    coords_string: either a path to a file or a multiline XYZ-style string (Element x y z)
    specified_spin: spin multiplicity value
    basis: basis set name
    maxsteps: maximum Berny steps

    Returns an XYZ-style block string with optimized coordinates (same format as input).

    If PySCF CCSD gradients or berny_solver are unavailable, falls back to a HF
    optimization (RHF/UHF) if possible, or returns the original geometry.
    """
    # read coords
    if os.path.isfile(coords_string):
        with open(coords_string, 'r') as fh:
            coord_text = fh.read()
    else:
        coord_text = coords_string

    # parse as in compute_mu_derivatives
    lines = [ln.strip() for ln in coord_text.splitlines() if ln.strip()]
    atoms = []
    positions = []
    for ln in lines:
        parts = ln.split()
        if len(parts) != 4:
            raise ValueError(f"Invalid coordinate line: '{ln}'. Expected 'Element x y z'.")
        atoms.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    positions = np.array(positions, dtype=float)

    # build molecule
    mol = gto.M(atom="\n".join(f"{a} {x} {y} {z}" for a, (x, y, z) in zip(atoms, positions)), basis=basis, spin=specified_spin, unit='Angstrom')

    # run initial mean-field
    try:
        if specified_spin == 0:
            mf = scf.RHF(mol).run()
        else:
            mf = scf.UHF(mol).run()
    except Exception as e:
        raise RuntimeError(f"SCF failed while preparing for CCSD optimization: {e}")

    # lazily import cc and grad to check for analytic gradient support
    try:
        from pyscf import cc as _cc, grad as _grad
    except Exception:
        _cc = None
        _grad = None

    # Require CCSD gradients + berny_solver to be available; otherwise fail loudly
    missing = []
    if berny_solver is None:
        missing.append('berny_solver (geometry optimizer)')
    if _cc is None:
        missing.append('pyscf.cc (CCSD module)')
    if _grad is None:
        missing.append('pyscf.grad (analytic gradients)')
    if missing:
        msg = (
            "CCSD gradients and/or berny_solver are not available in this PySCF installation; cannot perform CCSD geometry optimization.\n"
            "Missing: " + ", ".join(missing) + ".\n"
            "Recommended fixes:\n"
            " - Install PySCF and berny from conda-forge (recommended):\n"
            "     conda create -n pyscf-env -c conda-forge python=3.11 pyscf berny scipy -y\n"
            " - Or install via pip and build from source if wheel lacks compiled extensions:\n"
            "     python -m pip install pyscf berny\n"
            "   If pip wheel lacks CCSD gradient code, build PySCF from source (see INSTALL_PySCF.md in project root).\n"
            "After installing, re-run the script in the same Python environment.\n"
        )
        raise RuntimeError(msg)

    # Run CCSD optimization and raise if it fails
    try:
        # ensure static analysers know these are present
        assert _cc is not None, 'pyscf.cc not available'
        assert _grad is not None, 'pyscf.grad not available'
        # use the lazily imported cc
        mycc = _cc.CCSD(mf)

        # Progress indication with optional tqdm progress bar
        def run_with_message(fn, *fargs, desc: str = "working"):
            print(f"{desc}...")
            start_time = time.time()
            
            # Try to use tqdm for indeterminate progress if available
            if tqdm is not None:
                # Use a separate thread for the actual calculation
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(fn, *fargs)
                    
                    # Show progress bar while calculation runs
                    with tqdm(desc=desc, bar_format="{desc}: {elapsed}", leave=False) as pbar:
                        while not future.done():
                            time.sleep(0.5)
                            pbar.update(0)  # Just update the timer
                    
                    # Get the result
                    result = future.result()
            else:
                result = fn(*fargs)
            
            elapsed = time.time() - start_time
            print(f"{desc} completed in {elapsed:.1f}s")
            return result

        # run CCSD with simple progress messages
        run_with_message(mycc.run, desc="Running CCSD")

        # Run CCSD(T) triples correction if available (essential for CCSD(T) level accuracy)
        ccsd_t_energy = None
        try:
            if hasattr(mycc, 'ccsd_t'):
                ccsd_t_energy = run_with_message(lambda: mycc.ccsd_t(), desc="Computing CCSD(T) triples correction")
                print(f"CCSD(T) correction energy: {ccsd_t_energy:.8f} Hartree")
            else:
                print("Warning: CCSD(T) method not available, using CCSD gradients only")
        except Exception as e:
            print(f"Warning: CCSD(T) triples correction failed: {e}")
            print("Continuing with CCSD gradients...")
        
        try:
            # build gradients and optimize geometry (may be long)
            print("Building CCSD gradients...")
            g = _grad.ccsd.Gradients(mycc)
            
            # Test the gradient calculation first
            print("Testing gradient calculation...")
            try:
                test_grad = g.kernel()
                print(f"Gradient test successful, shape: {test_grad.shape}")
            except Exception as grad_e:
                print(f"Gradient calculation failed: {grad_e}")
                # The gradient failed, this is likely the source of our issue
                raise RuntimeError(f"CCSD gradient calculation failed: {grad_e}")
            
            # Narrow types for static analysis
            assert berny_solver is not None
            print("Starting berny geometry optimization...")
            
            # Use berny_solver.optimize with explicit error handling
            try:
                mol_opt: Any = run_with_message(lambda: getattr(berny_solver, 'optimize')(g, maxsteps=int(maxsteps)), desc="Optimizing geometry with CCSD gradients")
            except Exception as berny_e:
                print(f"Berny optimization failed: {berny_e}")
                print("Trying direct berny call...")
                mol_opt = getattr(berny_solver, 'optimize')(g, maxsteps=int(maxsteps))
            
            # ensure mol_opt exposes expected API
            if not hasattr(mol_opt, 'atom_coords') or not hasattr(mol_opt, 'atom_symbol'):
                # Try different ways to access the optimized geometry
                if hasattr(mol_opt, 'atom_coord'):
                    coords_opt = mol_opt.atom_coord()
                    symbols = mol_opt.atom_symbol() if hasattr(mol_opt, 'atom_symbol') else [mol_opt.atom_pure_symbol(i) for i in range(len(coords_opt))]
                elif hasattr(mol_opt, 'atom'):
                    # mol_opt might be a PySCF molecule object
                    coords_opt = mol_opt.atom_coords()
                    symbols = [mol_opt.atom_symbol(i) for i in range(mol_opt.natm)]
                else:
                    raise RuntimeError("Cannot extract optimized coordinates from berny result")
            else:
                coords_opt = mol_opt.atom_coords()
                symbols = [mol_opt.atom_symbol(i) for i in range(len(coords_opt))]
            
            optimized_coords = "\n".join(f"{symbols[i]} {x:.6f} {y:.6f} {z:.6f}" for i, (x, y, z) in enumerate(coords_opt))
            print("Geometry optimization completed successfully!")
            return optimized_coords
        except Exception as e:
            # CCSD gradients or berny optimization failed — no HF fallback requested
            print(f"Detailed error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"CCSD geometry optimization failed: {e}")
    except Exception as e_outer:
        # Top-level CCSD run failed, try HF optimization as fallback if enabled
        if use_hf_fallback:
            print(f"CCSD optimization failed ({e_outer}), falling back to HF optimization...")
            try:
                return optimize_geometry_hf(coords_string, specified_spin, basis, maxsteps)
            except Exception as hf_e:
                print(f"HF fallback also failed: {hf_e}")
                print("Returning original geometry (no optimization performed)")
                return coords_string.strip()
        else:
            raise RuntimeError(f"CCSD optimization failed: {e_outer}")


def optimize_geometry_hf(coords_string: str, specified_spin: int, basis: str = "aug-cc-pV5Z ", maxsteps: int = 50) -> str:
    """
    Fallback HF geometry optimization using Hartree-Fock gradients.
    """
    # read coords
    if os.path.isfile(coords_string):
        with open(coords_string, 'r') as fh:
            coord_text = fh.read()
    else:
        coord_text = coords_string

    # parse as in compute_mu_derivatives
    lines = [ln.strip() for ln in coord_text.splitlines() if ln.strip()]
    atoms = []
    positions = []
    for ln in lines:
        parts = ln.split()
        if len(parts) != 4:
            raise ValueError(f"Invalid coordinate line: '{ln}'. Expected 'Element x y z'.")
        atoms.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    positions = np.array(positions, dtype=float)

    # build molecule
    mol = gto.M(atom="\n".join(f"{a} {x} {y} {z}" for a, (x, y, z) in zip(atoms, positions)), basis=basis, spin=specified_spin, unit='Angstrom')

    # run initial mean-field
    try:
        if specified_spin == 0:
            mf = scf.RHF(mol).run()
        else:
            mf = scf.UHF(mol).run()
    except Exception as e:
        raise RuntimeError(f"SCF failed while preparing for HF optimization: {e}")

    # Check if berny_solver is available for HF optimization
    if berny_solver is None:
        print("berny_solver not available, returning original geometry")
        return coord_text.strip()
    
    try:
        # Use HF gradients for optimization
        print("Optimizing geometry with HF gradients...")
        mol_opt = getattr(berny_solver, 'optimize')(mf, maxsteps=int(maxsteps))
        
        # Extract coordinates
        if hasattr(mol_opt, 'atom_coords') and hasattr(mol_opt, 'atom_symbol'):
            coords_opt = mol_opt.atom_coords()
            symbols = [mol_opt.atom_symbol(i) for i in range(len(coords_opt))]
        elif hasattr(mol_opt, 'atom_coord'):
            coords_opt = mol_opt.atom_coord()
            symbols = [mol_opt.atom_symbol(i) for i in range(len(coords_opt))]
        else:
            # mol_opt might be a PySCF molecule object
            coords_opt = mol_opt.atom_coords()
            symbols = [mol_opt.atom_symbol(i) for i in range(mol_opt.natm)]
        
        optimized_coords = "\n".join(f"{symbols[i]} {x:.6f} {y:.6f} {z:.6f}" for i, (x, y, z) in enumerate(coords_opt))
        print("HF geometry optimization completed successfully!")
        return optimized_coords
        
    except Exception as e:
        print(f"HF optimization failed: {e}")
        print("Returning original geometry")
        return coord_text.strip()
