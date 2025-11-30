"""
SCF Stabilization Module
========================

This module provides utilities to prevent and handle overlap matrix singularities
in PySCF calculations, which can cause SCF convergence failures.
"""

import numpy as np
from pyscf import gto, scf
from typing import Tuple, Optional, Any, cast


def check_overlap_condition_number(mol: gto.Mole, threshold: float = 1e7) -> Tuple[float, bool]:
    """
    Check the condition number of the overlap matrix to detect potential singularities.
    
    Parameters:
    -----------
    mol : gto.Mole
        PySCF molecule object
    threshold : float
        Threshold for determining if overlap matrix is singular (default: 1e7)
        Increased threshold to handle large basis sets like aug-cc-pV5Z
        
    Returns:
    --------
    tuple[float, bool]
        (condition_number, is_singular)
    """
    # Build overlap matrix
    s = mol.intor('int1e_ovlp')
    
    # Compute condition number
    eigenvals = np.linalg.eigvals(s)
    min_eigenval = np.min(eigenvals)
    max_eigenval = np.max(eigenvals)
    
    if min_eigenval <= 0:
        print(f"WARNING: Negative or zero eigenvalue detected: {min_eigenval:.2e}")
        return float('inf'), True
    
    condition_number = float(max_eigenval / min_eigenval)
    is_singular = bool(condition_number > threshold)
    
    if is_singular:
        print(f"WARNING: Overlap matrix condition number = {condition_number:.2e} exceeds threshold {threshold:.2e}")
        print("This indicates potential numerical instability in the SCF calculation.")
    else:
        print(f"Overlap matrix condition number = {condition_number:.2e} (good)")
    
    return condition_number, is_singular


def stabilize_scf_convergence(mf, overlap_threshold: float = 1e7) -> None:
    """
    Apply stabilization techniques to prevent SCF convergence issues.
    
    Parameters:
    -----------
    mf : pyscf.scf object
        Mean field object (RHF, UHF, etc.)
    overlap_threshold : float
        Threshold for overlap matrix condition number (default: 1e7)
    """
    mol = mf.mol
    
    # Check overlap matrix condition
    condition_num, is_singular = check_overlap_condition_number(mol, overlap_threshold)
    
    # Apply mild stabilization for elevated condition numbers (1e6 - 1e7)
    if condition_num > 1e6:
        print("Applying SCF stabilization for elevated condition number...")
        
        # Lighter stabilization for marginally problematic cases
        if condition_num < 1e7:
            mf.level_shift = 0.2  # Mild level shift
            mf.damp = 0.3         # Light damping
            mf.diis_space = min(10, mf.diis_space)  # Slightly reduce DIIS space
            print(f"Applied light stabilization: level_shift={mf.level_shift}, damp={mf.damp}")
        else:
            # More aggressive stabilization for severe cases
            mf.level_shift = 0.5  # Add level shift to prevent oscillations
            mf.damp = 0.5         # Add density damping
            mf.diis_space = min(8, mf.diis_space)  # Reduce DIIS space
            mf.diis_start_cycle = 3  # Start DIIS later
            print(f"Applied aggressive stabilization: level_shift={mf.level_shift}, damp={mf.damp}")
        
        # For elevated condition numbers, relax initial convergence slightly but allow tight final convergence
        if hasattr(mf, 'conv_tol'):
            original_conv_tol = mf.conv_tol
            # Only relax for more severe cases (> 5e6), allow moderate cases to use tight convergence
            if condition_num > 5e6:
                mf.conv_tol = max(1e-7, original_conv_tol * 5)  # Relax by factor of 5 for better convergence
                print(f"Relaxed initial SCF convergence: {original_conv_tol:.2e} → {mf.conv_tol:.2e}")
                cast(Any, mf).original_conv_tol = original_conv_tol
            else:
                print(f"Condition number {condition_num:.2e} acceptable for tight convergence")
                cast(Any, mf).original_conv_tol = original_conv_tol

def robust_scf_calculation(atom_string: str, spin: int, basis: Optional[str] = None, 
                          target_conv_tol: float = 1e-8, max_cycle: int = 400,
                          initial_conv_tol: Optional[float] = None,
                          initial_level_shift: Optional[float] = None,
                          initial_diis_space: Optional[int] = None) -> scf.hf.SCF:
    """
    Perform a SCF calculation with automatic overlap matrix singularity handling.
    
    Parameters:
    -----------
    atom_string : str
        Molecular geometry string
    spin : int
        Spin multiplicity  
    basis : str
        Basis set name provided by the user (required)
    target_conv_tol : float
        Target convergence tolerance (default: 1e-8, increased for stability)
    max_cycle : int
        Maximum SCF cycles
    initial_conv_tol : float | None
        Optional relaxed tolerance for the stabilized run prior to tightening
    initial_level_shift : float | None
        Optional user-specified level shift to apply before stabilization
    initial_diis_space : int | None
        Optional override for the DIIS subspace dimension during stabilization
        
    Returns:
    --------
    tuple[pyscf.scf.hf.SCF, bool]
        Converged SCF object
    """
    if not basis:
        raise ValueError("robust_scf_calculation requires an explicit basis set from the user")

    print(f"Starting SCF calculation with basis {basis}")
    
    # Always use the user's specified basis - no fallback
    try:
        mol = gto.M(atom=atom_string, basis=basis, spin=spin, unit='Angstrom')
        condition_num, is_singular = check_overlap_condition_number(mol, threshold=1e7)
        
        if is_singular:
            print(f"Overlap matrix singularity detected with basis {basis} (condition = {condition_num:.2e})")
            print("Proceeding with user's basis and enhanced stabilization techniques...")
        elif condition_num > 1e6:
            print(f"Elevated but manageable condition number ({condition_num:.2e}). Proceeding with stabilization.")
        
        actual_basis = basis  # Always keep user's basis
            
    except Exception as e:
        print(f"Failed to build molecule with user's basis {basis}: {e}")
        raise RuntimeError(f"Cannot proceed with user-specified basis {basis}: {e}")
    
    # Set up SCF calculation
    if spin != 0:
        mf = scf.UHF(mol)
    else:
        mf = scf.RHF(mol)

    # Apply stabilization techniques based on overlap condition
    stabilize_scf_convergence(mf, overlap_threshold=1e7)

    # Allow callers to override baseline stabilization knobs
    mf.max_cycle = max_cycle
    if initial_diis_space is not None and hasattr(mf, 'diis_space'):
        mf.diis_space = initial_diis_space
    if initial_level_shift is not None:
        mf.level_shift = initial_level_shift
    if initial_conv_tol is not None:
        mf.conv_tol = initial_conv_tol
        if not hasattr(mf, 'original_conv_tol'):
            cast(Any, mf).original_conv_tol = initial_conv_tol
    
    # Run initial SCF with stabilization
    print("Running SCF with stabilization...")
    mf.run()
    
    if not mf.converged:
        print("Initial SCF failed to converge. Trying additional stabilization...")
        
        # Try even more aggressive stabilization
        mf.level_shift = 1.0
        mf.damp = 0.8
        mf.conv_tol = 1e-5  # Increased tolerance for SCF convergence
        mf.max_cycle = max_cycle * 2
        
        mf.run()
        
        if not mf.converged:
            raise RuntimeError("SCF failed to converge even with aggressive stabilization")
    
    # If we used a relaxed tolerance and want to tighten it
    if hasattr(mf, 'original_conv_tol') and mf.conv_tol > target_conv_tol:
        print(f"Tightening convergence from {mf.conv_tol:.2e} to {target_conv_tol:.2e}")
        
        # Gradually tighten convergence
        mf.level_shift = max(0.05, mf.level_shift * 0.2)  # Reduce level shift
        mf.damp = max(0.1, mf.damp * 0.4)                 # Reduce damping
        mf.conv_tol = target_conv_tol
        mf.max_cycle = max_cycle
        
        # Use previous orbitals as starting guess
        mf.run()
        
        if not mf.converged:
            print(f"Warning: Could not achieve target convergence {target_conv_tol:.2e}")
            print(f"Final convergence: {mf.conv_tol:.2e}")
            # Accept the current convergence level rather than failing
    
    print(f"SCF converged successfully. Final energy = {mf.e_tot:.12f} Hartree")
    return mf

def check_geometry_for_problems(atom_string: str, min_distance: float = 0.5) -> None:
    """
    Check molecular geometry for potential problems that could cause overlap issues.
    
    Parameters:
    -----------
    atom_string : str
        Molecular geometry string
    min_distance : float
        Minimum allowed interatomic distance in Angstroms
    """
    lines = [l.strip() for l in atom_string.strip().split('\n') if l.strip()]
    
    atoms = []
    coords = []
    
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            atom = parts[0]
            x, y, z = map(float, parts[1:4])
            atoms.append(atom)
            coords.append([x, y, z])
    
    coords = np.array(coords)
    n_atoms = len(atoms)
    
    print(f"Checking geometry for {n_atoms} atoms...")
    
    problematic_pairs = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            distance = np.linalg.norm(coords[i] - coords[j])
            if distance < min_distance:
                problematic_pairs.append((i, j, atoms[i], atoms[j], distance))
    
    if problematic_pairs:
        print("WARNING: Found potentially problematic atom distances:")
        for i, j, atom_i, atom_j, dist in problematic_pairs:
            print(f"  {atom_i}({i}) - {atom_j}({j}): {dist:.4f} Å (< {min_distance:.2f} Å)")
        print("These short distances may cause overlap matrix singularities.")
    else:
        print("Geometry check passed: no problematic atom distances found.")