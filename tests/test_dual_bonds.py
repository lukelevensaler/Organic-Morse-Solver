#!/usr/bin/env python3
"""
Test script for dual bond axes functionality in derivatives_dipole_moment.py
"""

import sys
import os
import numpy as np

# Add the current directory to the path so we can import our module
sys.path.insert(0, '/Users/lukelevensaler/morse_solver')

try:
    from derivatives_dipole_moment import compute_mu_derivatives
    print("Successfully imported derivatives_dipole_moment module")
except ImportError as e:
    print(f"Failed to import module: {e}")
    sys.exit(1)

# Test molecule: simple H2O for testing
test_coords = """O 0.000000 0.000000 0.000000
H 0.757000 0.586000 0.000000
H -0.757000 0.586000 0.000000"""

print("Test coordinates:")
print(test_coords)

# Test parsing dual bond axes string
try:
    print("\nTesting dual bond axes parsing...")
    
    # Test case: both O-H bonds in H2O
    dual_axes = "(0,1);(0,2)"  # O-H1 and O-H2 bonds
    print(f"Dual axes string: {dual_axes}")
    
    # This should work if our parsing is correct
    result = compute_mu_derivatives(
        coords_string=test_coords,
        specified_spin=0,  # Singlet H2O
        delta=0.01,  # Small displacement
        basis="sto-3g",  # Fast basis for testing
        dual_bond_axes=dual_axes,
        m1=16.0,  # Mass of O (same as main_morse_solver.py m1 = A)
        m2=1.008  # Mass of H (same as main_morse_solver.py m2 = B)
    )
    
    print(f"Derivatives computed successfully: mu1={result[0]:.6e}, mu2={result[1]:.6e}")
    
except Exception as e:
    print(f"Error during computation: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed.")