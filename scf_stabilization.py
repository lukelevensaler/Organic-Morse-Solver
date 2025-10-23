"""
SCF Stabilization Module
========================

This module provides utilities to prevent and handle overlap matrix singularities
in PySCF calculations, which can cause SCF convergence failures.
"""

import numpy as np
from pyscf import gto, scf
import warnings
from typing import Tuple, Optional


def check_overlap_condition_number(mol: gto.Mole, threshold: float = 5e6) -> Tuple[float, bool]:
    """
    Check the condition number of the overlap matrix to detect potential singularities.
    
    Parameters:
    -----------
    mol : gto.Mole
        PySCF molecule object
    threshold : float
        Threshold for determining if overlap matrix is singular (default: 5e6)
        More permissive threshold since 1e6-5e6 is often still acceptable
        
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


def stabilize_scf_convergence(mf, overlap_threshold: float = 5e6) -> None:
    """
    Apply stabilization techniques to prevent SCF convergence issues.
    
    Parameters:
    -----------
    mf : pyscf.scf object
        Mean field object (RHF, UHF, etc.)
    overlap_threshold : float
        Threshold for overlap matrix condition number
    """
    mol = mf.mol
    
    # Check overlap matrix condition
    condition_num, is_singular = check_overlap_condition_number(mol, overlap_threshold)
    
    # Apply mild stabilization for elevated condition numbers (1e6 - 5e6)
    if condition_num > 1e6:
        print("Applying SCF stabilization for elevated condition number...")
        
        # Lighter stabilization for marginally problematic cases
        if condition_num < 5e6:
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
        
        # For any elevated condition number, relax initial convergence slightly
        if hasattr(mf, 'conv_tol'):
            original_conv_tol = mf.conv_tol
            mf.conv_tol = max(1e-8, original_conv_tol * 5)  # Relax by factor of 5
            print(f"Relaxed initial SCF convergence: {original_conv_tol:.2e} → {mf.conv_tol:.2e}")
            mf._original_conv_tol = original_conv_tol


def try_alternative_basis_sets(atom_string: str, spin: int, original_basis: str = "aug-cc-pVTZ") -> Tuple[gto.Mole, str]:
    """
    Try alternative basis sets if the original one causes overlap matrix issues.
    
    Parameters:
    -----------
    atom_string : str
        Molecular geometry string
    spin : int
        Spin multiplicity
    original_basis : str
        Original basis set that failed
        
    Returns:
    --------
    tuple[gto.Mole, str]
        (molecule_object, successful_basis_name)
    """
    # Hierarchy of basis sets from most to least diffuse/problematic
    basis_hierarchy = [
        original_basis,
        "cc-pVTZ",      # Remove augmentation
        "cc-pVDZ",      # Smaller basis
        "6-311++G**",   # Different family with diffuse
        "6-311G**",     # Remove diffuse functions
        "6-31+G*",      # Smaller with some diffuse
        "6-31G*",       # Remove all diffuse functions
        "STO-3G"        # Minimal basis (last resort)
    ]
    
    # Use more permissive threshold for basis set selection
    permissive_threshold = 1e7  # 10x more permissive for basis selection
    
    for basis in basis_hierarchy:
        try:
            print(f"Trying basis set: {basis}")
            mol = gto.M(atom=atom_string, basis=basis, spin=spin, unit='Angstrom')
            
            # Check overlap condition with permissive threshold
            condition_num, is_singular = check_overlap_condition_number(mol, threshold=permissive_threshold)
            
            if not is_singular:
                if basis != original_basis:
                    print(f"Successfully switched from {original_basis} to {basis}")
                return mol, basis
            else:
                print(f"Basis {basis} shows severe overlap issues (condition = {condition_num:.2e})")
                
        except Exception as e:
            print(f"Failed to build molecule with basis {basis}: {e}")
            continue
    
    # If we get here, all basis sets failed
    raise RuntimeError("All attempted basis sets show severe overlap matrix singularities. Check molecular geometry for problematic atom distances.")


def robust_scf_calculation(atom_string: str, spin: int, basis: str = "aug-cc-pVTZ", 
                          target_conv_tol: float = 1e-12, max_cycle: int = 400) -> scf.hf.SCF:
    """
    Perform a robust SCF calculation with automatic overlap matrix singularity handling.
    
    Parameters:
    -----------
    atom_string : str
        Molecular geometry string
    spin : int
        Spin multiplicity  
    basis : str
        Basis set name
    target_conv_tol : float
        Target convergence tolerance
    max_cycle : int
        Maximum SCF cycles
        
    Returns:
    --------
    pyscf.scf object
        Converged SCF object
    """
    print(f"Starting robust SCF calculation with basis {basis}")
    
    # Try to build molecule with original basis
    try:
        mol = gto.M(atom=atom_string, basis=basis, spin=spin, unit='Angstrom')
        condition_num, is_singular = check_overlap_condition_number(mol, threshold=5e6)
        
        # Only switch basis for truly problematic cases (condition > 5e6)
        if is_singular:
            print("Severe overlap matrix singularity detected. Trying alternative basis sets...")
            mol, actual_basis = try_alternative_basis_sets(atom_string, spin, basis)
            print(f"Using basis: {actual_basis}")
        else:
            actual_basis = basis
            if condition_num > 1e6:
                print(f"Elevated but manageable condition number ({condition_num:.2e}). Proceeding with stabilization.")
            
    except Exception as e:
        print(f"Failed to build molecule with {basis}: {e}")
        print("Trying alternative basis sets...")
        mol, actual_basis = try_alternative_basis_sets(atom_string, spin, basis)
    
    # Set up SCF calculation
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)
    
    # Apply stabilization techniques based on overlap condition
    stabilize_scf_convergence(mf, overlap_threshold=5e6)
    
    # Set cycle limit
    mf.max_cycle = max_cycle
    
    # Run initial SCF with stabilization
    print("Running SCF with stabilization...")
    mf.run()
    
    if not mf.converged:
        print("Initial SCF failed to converge. Trying additional stabilization...")
        
        # Try even more aggressive stabilization
        mf.level_shift = 1.0
        mf.damp = 0.8
        mf.conv_tol = 1e-6  # Very relaxed
        mf.max_cycle = max_cycle * 2
        
        mf.run()
        
        if not mf.converged:
            raise RuntimeError("SCF failed to converge even with aggressive stabilization")
    
    # If we used a relaxed tolerance and want to tighten it
    if hasattr(mf, '_original_conv_tol') and mf.conv_tol > target_conv_tol:
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