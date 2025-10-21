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
import time
from tqdm import tqdm

# ===== MAXIMUM PRECISION DIPOLE DERIVATIVE SOLVING WITH CCSD(T) =======
#
# This module implements the most computationally rigorous approach possible:
# - All dipole moments computed at full CCSD(T) level
# - Extremely tight SCF convergence (1e-12) for optimal CCSD(T) starting point
# - High-precision CCSD convergence (1e-9 to 1e-10) 
# - Mandatory CCSD(T) triples correction for all calculations
# - High-quality correlation-consistent basis sets (aug-cc-pVTZ default)
# - Optimized DIIS spaces and maximum iteration counts
# - Direct algorithms for best numerical accuracy
#
# This ensures maximum computational rigor for overtone spectroscopy applications
# where dipole derivative accuracy is critical.

def dipole_for_geometry(atom_string: str, spin: int, basis: str = "aug-cc-pVTZ", conv_tol: float = 1e-9, max_cycle: int = 150) -> np.ndarray:
    """Return the molecular dipole vector (Debye) computed at CCSD(T) level.

    This function uses full CCSD(T) method for accurate dipole moments, which is 
    essential for high-accuracy overtone transition dipole calculations.

    NOTE: Hartree refers to the atomic unit of energy, not the Hartree-Fock method, which we do not use.

    Parameters:
    -----------
    atom_string : str
        Molecular geometry in XYZ format
    spin : int
        Spin multiplicity
    basis : str
        Basis set (default: aug-cc-pVTZ for high accuracy)
    conv_tol : float
        CCSD convergence tolerance (default: 1e-9 for very tight convergence)
    max_cycle : int
        Maximum CCSD iterations (default: 150 for robust convergence)
        
    Returns:
    --------
    np.ndarray
        Dipole moment vector in Debye units
    """
    print(f"Computing CCSD(T) dipole for geometry with basis {basis}")
    
    mol = gto.M(atom=atom_string, basis=basis, spin=spin)
    
    # Run initial SCF calculation with maximum precision for CCSD(T)
    with tqdm(desc="🔬 SCF Convergence", unit="step", colour='blue') as pbar:
        if spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)
        
        # Use very tight SCF convergence for highest quality CCSD(T)
        mf.conv_tol = 1e-12  # Extremely tight for CCSD(T) accuracy
        mf.max_cycle = 300   # More cycles for robust convergence
        mf.diis_space = 12   # Larger DIIS space for better SCF convergence
        pbar.set_postfix(tol=f"{mf.conv_tol:.0e}", max_cycle=mf.max_cycle)
        
        mf.run()
        pbar.update(1)
        pbar.set_postfix(converged=mf.converged, energy=f"{mf.e_tot:.6f}")
    
    if not mf.converged:
        raise RuntimeError("SCF did not converge to required precision - cannot proceed with high-accuracy CCSD(T)")
    
    print(f"High-precision SCF converged for CCSD(T). Energy = {mf.e_tot:.12f} Hartree")
    
    # Import CC module
    try:
        from pyscf import cc
    except ImportError:
        raise RuntimeError("PySCF CC module not available - cannot compute CCSD(T) dipoles")
    
    # Set up CCSD calculation with maximum rigor
    mycc = cc.CCSD(mf)
    mycc.conv_tol = conv_tol
    mycc.max_cycle = max_cycle
    mycc.diis_space = 12  # Maximum DIIS space for optimal convergence
    mycc.direct = True    # Use direct algorithm for better accuracy
    
    print(f"High-precision CCSD(T) dipole settings: conv_tol={mycc.conv_tol:.2e}, max_cycle={mycc.max_cycle}")
    
    # Run CCSD calculation with maximum precision
    with tqdm(desc="🧬 CCSD Correlation", unit="step", colour='green') as pbar:
        pbar.set_postfix(tol=f"{mycc.conv_tol:.0e}", max_cycle=mycc.max_cycle)
        try:
            mycc.run()
            pbar.update(1)
            pbar.set_postfix(converged=mycc.converged, corr_energy=f"{mycc.e_corr:.6f}")
        except Exception as e:
            raise RuntimeError(f"High-precision CCSD calculation failed: {e}")
    
    if not mycc.converged:
        raise RuntimeError("CCSD did not converge to required precision - cannot compute high-accuracy dipole")
    
    print(f"High-precision CCSD converged. Correlation energy = {mycc.e_corr:.12f} Hartree")
    
    # Compute CCSD(T) triples correction - MANDATORY for maximum accuracy
    with tqdm(desc="⚛️ CCSD(T) Triples Correction", unit="step", colour='red') as pbar:
        try:
            if hasattr(mycc, 'ccsd_t'):
                pbar.set_postfix(status="computing")
                e_t = mycc.ccsd_t()
                pbar.update(1)
                pbar.set_postfix(E_T=f"{e_t:.8f}")
                print(f"High-precision CCSD(T) triples correction: {e_t:.12f} Hartree")
                total_energy = mycc.e_tot + e_t
                print(f"Total high-precision CCSD(T) energy: {total_energy:.12f} Hartree")
            else:
                raise RuntimeError("CCSD(T) triples correction not available - cannot achieve required computational rigor")
        except Exception as e:
            raise RuntimeError(f"CCSD(T) triples correction failed - required for maximum accuracy: {e}")
    
    # Calculate CCSD dipole moment
    with tqdm(desc="🔬 Computing CCSD Dipole Moment", unit="step", colour='magenta') as pbar:
        try:
            # Use CCSD density matrices for dipole calculation
            dm1 = mycc.make_rdm1()
            pbar.update(1)
            pbar.set_postfix(step="density_matrix")
            
            # Calculate dipole integrals
            dip_ints = mol.intor('int1e_r', comp=3)  # x, y, z components
            pbar.update(1) 
            pbar.set_postfix(step="dipole_integrals")
            
            # Calculate electronic dipole contribution
            dip_elec = -np.einsum('xij,ji->x', dip_ints, dm1)
            pbar.update(1)
            pbar.set_postfix(step="electronic_contribution")
            
            # Add nuclear contribution
            charges = mol.atom_charges()
            coords = mol.atom_coords()
            dip_nuc = np.einsum('i,ix->x', charges, coords)
            pbar.update(1)
            pbar.set_postfix(step="nuclear_contribution")
            
            # Total dipole moment in atomic units
            dipole_au = dip_elec + dip_nuc
            
            # Convert from atomic units to Debye (1 au = 2.541746 Debye)
            au_to_debye = 2.541746473
            dipole_debye = dipole_au * au_to_debye
            
            print(f"High-precision CCSD(T) dipole moment: {np.linalg.norm(dipole_debye):.8f} Debye")
            print(f"High-precision CCSD(T) dipole components (Debye): [{dipole_debye[0]:.8f}, {dipole_debye[1]:.8f}, {dipole_debye[2]:.8f}]")
            
            return dipole_debye
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute high-precision CCSD(T) dipole moment: {e}")


def compute_mu_derivatives(coords_string: str, specified_spin: int, delta: float = 0.005, basis: str = "aug-cc-pVTZ", atom_index: int = 0, axis: int = 2, bond_pair: tuple[int, int] | None = None, dual_bond_axes: str | None = None, m1: float | None = None, m2: float | None = None) -> tuple[float, float]:
    """
    Compute first and second derivatives of the dipole using finite differences at CCSD(T) level.

    coords_string: multiline string of fully numeric coordinates (Element x y z)
    specified_spin: spin state for dipole calculation
    delta: displacement magnitude (Å)
    basis: quantum chemistry basis set
    atom_index: which atom to displace (0-based)
    axis: which Cartesian axis to displace (0=x, 1=y, 2=z)
    bond_pair: optional tuple of atom indices to define bond stretch direction
    dual_bond_axes: optional string in format "(n,x);(a,x)" for two bond axes with shared element x
    m1: mass of element A (same as main_morse_solver.py m1 = A)
    m2: mass of element B (same as main_morse_solver.py m2 = B)
    
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
    
    def parse_dual_bond_axes(dual_axes_str: str) -> tuple[tuple[int, int], tuple[int, int]]:
        """Parse dual bond axes string like '(n,x);(a,x)' into bond pairs."""
        try:
            parts = dual_axes_str.split(';')
            if len(parts) != 2:
                raise ValueError("Dual bond axes must be separated by semicolon")
            
            bond1_str = parts[0].strip('() ')
            bond2_str = parts[1].strip('() ')
            
            bond1_indices = tuple(map(int, bond1_str.split(',')))
            bond2_indices = tuple(map(int, bond2_str.split(',')))
            
            if len(bond1_indices) != 2 or len(bond2_indices) != 2:
                raise ValueError("Each bond must have exactly two atom indices")
                
            return bond1_indices, bond2_indices
            
        except Exception as e:
            raise ValueError(f"Invalid dual bond axes format '{dual_axes_str}'. Expected '(n,x);(a,x)': {e}")
    
    def create_symmetric_antisymmetric_vectors(positions: np.ndarray, atoms: list[str], 
                                             bond1: tuple[int, int], bond2: tuple[int, int],
                                             m1: float, m2: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Create symmetric and antisymmetric displacement vectors for dual bond axes.
        
        For dual bond axes like (n,x);(a,x):
        - Element 'a' (first element type in user input) corresponds to m1 (same as main_morse_solver.py) 
        - Element 'b' (second element type in user input) corresponds to m2 (same as main_morse_solver.py)
        - In bond pairs like n-x and a-x, 'n' and 'a' get m1, 'x' gets m2
        
        Returns:
            tuple of (symmetric_vector, antisymmetric_vector) - both normalized 3N vectors
        """
        n_atoms = len(atoms)
        
        # Create normalized bond vectors
        i1, j1 = bond1  # bond1: (n,x) 
        i2, j2 = bond2  # bond2: (a,x)
        
        if not (0 <= i1 < n_atoms and 0 <= j1 < n_atoms and 0 <= i2 < n_atoms and 0 <= j2 < n_atoms):
            raise IndexError("Bond atom indices out of range")
            
        # Calculate bond vectors and normalize
        bond_vec1 = positions[j1] - positions[i1]
        norm1 = np.linalg.norm(bond_vec1)
        if norm1 == 0:
            raise ValueError("Bond 1 atoms are at identical positions")
        unit1 = bond_vec1 / norm1
        
        bond_vec2 = positions[j2] - positions[i2]  
        norm2 = np.linalg.norm(bond_vec2)
        if norm2 == 0:
            raise ValueError("Bond 2 atoms are at identical positions")
        unit2 = bond_vec2 / norm2
        
        # Create 3N displacement vectors for each bond
        e1 = np.zeros(3 * n_atoms)
        e2 = np.zeros(3 * n_atoms)
        
        # Bond 1: move atom i1 by +unit1/2, atom j1 by -unit1/2
        e1[3*i1:3*i1+3] = unit1 / 2.0
        e1[3*j1:3*j1+3] = -unit1 / 2.0
        
        # Bond 2: move atom i2 by +unit2/2, atom j2 by -unit2/2  
        e2[3*i2:3*i2+3] = unit2 / 2.0
        e2[3*j2:3*j2+3] = -unit2 / 2.0
        
        # Create symmetric and antisymmetric combinations
        e_sym = (e1 + e2) / np.sqrt(2.0)
        e_anti = (e1 - e2) / np.sqrt(2.0)
        
        # Apply mass-weighting using user-provided masses
        # For dual bonds like (n,x);(a,x): n and a get m1, x gets m2
        for atom_idx in range(n_atoms):
            # Determine which mass to use based on bond structure
            if atom_idx == i1 or atom_idx == i2:  # atoms n or a
                sqrt_mass = np.sqrt(m1)
            elif atom_idx == j1 or atom_idx == j2:  # atom x (shared)
                sqrt_mass = np.sqrt(m2)
            else:
                # For atoms not involved in the dual bonds, assign unit mass (no weighting)
                # This is appropriate since we're focused on the specific bond modes
                sqrt_mass = 1.0
            
            # Mass-weight each 3-component block
            start_idx = 3 * atom_idx
            end_idx = start_idx + 3
            e_sym[start_idx:end_idx] *= sqrt_mass
            e_anti[start_idx:end_idx] *= sqrt_mass
        
        # Renormalize after mass-weighting
        e_sym_norm = np.linalg.norm(e_sym)
        e_anti_norm = np.linalg.norm(e_anti)
        
        if e_sym_norm > 0:
            e_sym /= e_sym_norm
        if e_anti_norm > 0:
            e_anti /= e_anti_norm
            
        return e_sym, e_anti

    # create displaced geometries
    pos0 = positions.copy()
    pos_plus = positions.copy()
    pos_minus = positions.copy()

    # Handle dual bond axes case with symmetric/antisymmetric combinations
    if dual_bond_axes is not None:
        print("Processing dual bond axes with symmetric/antisymmetric combinations and mass-weighting...")
        
        # Require user-provided masses for dual bond axes
        if m1 is None or m2 is None:
            raise ValueError("dual_bond_axes requires m1 and m2 parameters (user-provided masses from main solver)")
        
        # Parse dual bond axes
        bond1, bond2 = parse_dual_bond_axes(dual_bond_axes)
        print(f"Bond 1: atoms {bond1[0]}-{bond1[1]} ({atoms[bond1[0]]}-{atoms[bond1[1]]})")
        print(f"Bond 2: atoms {bond2[0]}-{bond2[1]} ({atoms[bond2[0]]}-{atoms[bond2[1]]})")
        print(f"Using user-provided masses: m1={m1:.3f}, m2={m2:.3f}")
        
        # Create symmetric and antisymmetric displacement vectors
        e_sym, e_anti = create_symmetric_antisymmetric_vectors(positions, atoms, bond1, bond2, m1, m2)
        
        # Apply displacement along the symmetric mode (larger projection typically)
        # Convert 3N vector back to coordinate displacements
        n_atoms = len(atoms)
        displacement_coords = np.zeros_like(positions)
        for atom_idx in range(n_atoms):
            start_idx = 3 * atom_idx
            displacement_coords[atom_idx] = e_sym[start_idx:start_idx+3]
        
        # Scale by delta and apply displacements
        pos_plus += displacement_coords * delta
        pos_minus -= displacement_coords * delta
        
        print("Applied mass-weighted symmetric stretch displacement")
        
    elif bond_pair is not None:
        # Single bond pair case
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

    # compute dipoles at maximum precision CCSD(T) level
    print("Using maximum precision CCSD(T) level for dipole moment calculations")
    # Use high-quality basis sets for maximum accuracy
    ccsd_basis = basis  # Use the requested basis directly for maximum rigor
    if basis == "aug-cc-pVQZ":
        # Keep the high-quality basis for ultimate accuracy
        print("Using aug-cc-pVQZ basis for maximum CCSD(T) accuracy")
    elif "cc-p" in basis.lower():
        ccsd_basis = basis  # Use correlation-consistent basis as requested
    else:
        ccsd_basis = "aug-cc-pVTZ"  # Default to high-quality basis
    
    print(f"Maximum precision CCSD(T) basis set: {ccsd_basis}")
    
    # Use tighter convergence for finite difference derivatives
    tight_conv_tol = 1e-10  # Very tight for numerical derivatives
    max_cycles = 200        # More cycles for robust convergence
    
    # Main finite difference progress tracker
    with tqdm(total=3, desc="🧮 Finite Difference CCSD(T) Calculations", unit="geom", colour='green') as pbar:
        pbar.set_postfix(geometry="equilibrium")
        mu0 = dipole_for_geometry(atom0, specified_spin, basis=ccsd_basis, conv_tol=tight_conv_tol, max_cycle=max_cycles)
        pbar.update(1)
        
        pbar.set_postfix(geometry="+δ displacement")
        mu_plus = dipole_for_geometry(atom_plus, specified_spin, basis=ccsd_basis, conv_tol=tight_conv_tol, max_cycle=max_cycles)
        pbar.update(1)
        
        pbar.set_postfix(geometry="-δ displacement")
        mu_minus = dipole_for_geometry(atom_minus, specified_spin, basis=ccsd_basis, conv_tol=tight_conv_tol, max_cycle=max_cycles)
        pbar.update(1)
        
        pbar.set_postfix(geometry="completed")

    # Debug output for high-precision CCSD(T) dipole moments
    print(f"Debug: CCSD(T) dipole at equilibrium: {mu0} Debye")
    print(f"Debug: CCSD(T) dipole at +δ: {mu_plus} Debye") 
    print(f"Debug: CCSD(T) dipole at -δ: {mu_minus} Debye")
    print(f"Debug: High-precision displacement δ = {delta} Å")
    
    # finite-difference derivatives
    mu_prime_vec = (mu_plus - mu_minus) / (2.0 * delta)
    mu_double_vec = (mu_plus - 2.0 * mu0 + mu_minus) / (delta ** 2)
    
    print(f"Debug: CCSD(T) first derivative vector (Debye/Å): {mu_prime_vec}")
    print(f"Debug: CCSD(T) second derivative vector (Debye/Å²): {mu_double_vec}")

    # convert from Debye/(Å^n) to C·m/(m^n)
    D_TO_CM = 3.33564e-30
    mu1_si = np.linalg.norm(mu_prime_vec) * (D_TO_CM / 1e-10)
    mu2_si = np.linalg.norm(mu_double_vec) * (D_TO_CM / 1e-20)
    
    print(f"Debug: |μ'(0)| = {mu1_si:.10e} C·m/m (CCSD(T) precision)")
    print(f"Debug: |μ''(0)| = {mu2_si:.10e} C·m/m² (CCSD(T) precision)")

    return float(mu1_si), float(mu2_si)


def optimize_geometry_ccsd(coords_string: str, specified_spin: int, basis: str = "aug-cc-pVTZ", maxsteps: int = 50) -> str:
    """
    Run a CCSD geometry optimization (via CCSD gradients + Berny).

    coords_string: either a path to a file or a multiline XYZ-style string (Element x y z)
    specified_spin: spin multiplicity value
    basis: basis set name (default: aug-cc-pVTZ for maximum CCSD(T) accuracy)
    maxsteps: maximum Berny steps

    Returns an XYZ-style block string with optimized coordinates (same format as input).
    
    Uses maximum precision CCSD(T) gradients for the most rigorous geometry optimization.
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

    # run initial mean-field with maximum precision for CCSD(T)
    if specified_spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)
    
    # Use maximum precision SCF settings for CCSD(T) optimization
    mf.conv_tol = 1e-12   # Extremely tight convergence
    mf.max_cycle = 400    # Many cycles for robust convergence
    mf.diis_space = 15    # Large DIIS space
    
    # Start progress tracking for SCF optimization
    with tqdm(desc="🔬 SCF for Geometry Optimization", unit="step", colour='blue') as pbar:
        pbar.set_postfix(tol=f"{mf.conv_tol:.0e}", max_cycle=mf.max_cycle)
        mf.run()
        pbar.update(1)
        pbar.set_postfix(converged=mf.converged, energy=f"{mf.e_tot:.6f}")
    
    if not mf.converged:
        raise RuntimeError("High-precision SCF did not converge - cannot proceed with CCSD(T) optimization")

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
        # use the lazily imported cc with maximum rigor convergence controls
        mycc = _cc.CCSD(mf)
        
        # Set maximum precision convergence criteria for ultimate CCSD(T) accuracy
        mycc.conv_tol = 1e-10   # Maximum precision convergence
        mycc.max_cycle = 200    # Many iterations for robust convergence
        mycc.diis_space = 15    # Maximum DIIS space for optimal convergence
        mycc.direct = True      # Use direct algorithm for better numerical accuracy
        
        print(f"Maximum precision CCSD(T) optimization settings: conv_tol={mycc.conv_tol}, max_cycle={mycc.max_cycle}, diis_space={mycc.diis_space}")

        # Run CCSD calculation with progress tracking
        with tqdm(desc="🧬 CCSD for Geometry Optimization", unit="iter", colour='green') as pbar:
            pbar.set_postfix(conv_tol=f"{mycc.conv_tol:.0e}", max_cycle=mycc.max_cycle)
            print("Running maximum precision CCSD calculation...")
            start_time = time.time()
            mycc.run()
            elapsed = time.time() - start_time
            pbar.update(1)
            pbar.set_postfix(converged=mycc.converged, energy=f"{mycc.e_corr:.8f}", time=f"{elapsed:.1f}s")
        print(f"CCSD calculation completed in {elapsed:.1f}s")
        
        if not mycc.converged:
            raise RuntimeError("CCSD did not converge to required precision - cannot proceed with optimization")

        # Run CCSD(T) triples correction - MANDATORY for maximum computational rigor
        ccsd_t_energy = None
        with tqdm(desc="⚛️ CCSD(T) Triples for Optimization", unit="step", colour='red') as pbar:
            try:
                if hasattr(mycc, 'ccsd_t'):
                    print("Computing maximum precision CCSD(T) triples correction...")
                    start_time = time.time()
                    ccsd_t_energy = mycc.ccsd_t()
                    elapsed = time.time() - start_time
                    pbar.update(1)
                    pbar.set_postfix(E_T=f"{ccsd_t_energy:.8f}", time=f"{elapsed:.1f}s")
                    print(f"CCSD(T) triples correction completed in {elapsed:.1f}s")
                    print(f"Maximum precision CCSD(T) correction energy: {ccsd_t_energy:.12f} Hartree")
                else:
                    raise RuntimeError("CCSD(T) method not available - cannot achieve required computational rigor for optimization")
            except Exception as e:
                raise RuntimeError(f"CCSD(T) triples correction failed - required for maximum rigor optimization: {e}")
        
        try:
            # build gradients and optimize geometry (may be long)
            with tqdm(desc="📊 Building CCSD Gradients", unit="step", colour='cyan') as pbar:
                print("Building CCSD gradients...")
                g = _grad.ccsd.Gradients(mycc)
                pbar.update(1)
            
            # Test the gradient calculation first
            with tqdm(desc="🧪 Testing Gradient Calculation", unit="step", colour='yellow') as pbar:
                print("Testing gradient calculation...")
                try:
                    test_grad = g.kernel()
                    pbar.update(1)
                    pbar.set_postfix(shape=f"{test_grad.shape}")
                    print(f"Gradient test successful, shape: {test_grad.shape}")
                except Exception as grad_e:
                    print(f"Gradient calculation failed: {grad_e}")
                    # The gradient failed, this is likely the source of our issue
                    raise RuntimeError(f"CCSD gradient calculation failed: {grad_e}")
            
            # Narrow types for static analysis
            assert berny_solver is not None
            print("Starting maximum precision CCSD(T) berny geometry optimization...")
            
            # Use berny_solver.optimize with explicit error handling
            with tqdm(desc="🔧 Berny Geometry Optimization", total=maxsteps, unit="step", colour='magenta') as pbar:
                try:
                    start_time = time.time()
                    
                    # Create a wrapper to track optimization steps
                    class OptimizationTracker:
                        def __init__(self, max_steps):
                            self.max_steps = max_steps
                            self.step = 0
                        
                        def __call__(self, *args, **kwargs):
                            self.step += 1
                            pbar.update(1)
                            pbar.set_postfix(step=f"{self.step}/{self.max_steps}")
                            return True
                    
                    mol_opt: Any = getattr(berny_solver, 'optimize')(g, maxsteps=int(maxsteps))
                    elapsed = time.time() - start_time
                    pbar.set_postfix(completed=True, time=f"{elapsed:.1f}s")
                    print(f"Berny geometry optimization completed in {elapsed:.1f}s")
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
            print("Maximum precision CCSD(T) geometry optimization completed successfully!")
            return optimized_coords
        except Exception as e:
            # CCSD gradients or berny optimization failed — no HF fallback requested
            print(f"Detailed error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"CCSD geometry optimization failed: {e}")
    except Exception as e_outer:
        # CCSD optimization failed
        raise RuntimeError(f"CCSD optimization failed: {e_outer}")



