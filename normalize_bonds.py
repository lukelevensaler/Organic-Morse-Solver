import numpy as np

"""
This file normalizes and displaces the computed bond axis vectors for the
provided molecular geometry, so as to allow proper ab initio structure optimization
for dipole calculations. For a bond axis pair, we double normalize and mass weight
to deal with stretch symmetry/asymmetry.
"""

def parse_dual_bond_axes(dual_axes_str: str) -> tuple[tuple[int, int], tuple[int, int]]:
        """Parse dual bond axes string like '(n,x);(a,x)' into bond pairs.
        
        Note: Converts from 1-based indexing (user input) to 0-based indexing (internal use).
        """
        try:
            parts = dual_axes_str.split(';')
            if len(parts) != 2:
                raise ValueError("Dual bond axes must be separated by semicolon")
            
            bond1_str = parts[0].strip('() ')
            bond2_str = parts[1].strip('() ')
            
            # Parse indices and convert from 1-based to 0-based
            bond1_raw = tuple(map(int, bond1_str.split(',')))
            bond2_raw = tuple(map(int, bond2_str.split(',')))
            
            if len(bond1_raw) != 2 or len(bond2_raw) != 2:
                raise ValueError("Each bond must have exactly two atom indices")
            
            # Convert from 1-based to 0-based indexing
            bond1_indices = (bond1_raw[0] - 1, bond1_raw[1] - 1)
            bond2_indices = (bond2_raw[0] - 1, bond2_raw[1] - 1)
                
            return bond1_indices, bond2_indices
            
        except Exception as e:
            raise ValueError(f"Invalid dual bond axes format '{dual_axes_str}'. Expected '(n,x);(a,x)' with 1-based indices: {e}")
    
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


def process_bond_displacements(positions, atoms, dual_bond_axes=None, bond_pair=None, 
                              delta=0.01, m1=None, m2=None, atom_index=0, axis=2):
    """Process bond displacements for various bond configurations."""
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
    
    return pos_plus, pos_minus, block_from_positions
