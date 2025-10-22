#!/usr/bin/env python3
"""
Test script for dual bond axes functionality in derivatives_dipole_moment.py
"""

import sys
import os

# Add the parent directory to the path so we can import our module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_dual_bond_axes():
    """Test dual bond axes functionality with H2O molecule."""
    try:
        from derivatives_dipole_moment import compute_mu_derivatives
        print("Successfully imported derivatives_dipole_moment module")
    except ImportError as e:
        print(f"Failed to import module: {e}")
        raise

    # Test molecule: simple H2O for testing
    test_coords = """O 0.000000 0.000000 0.000000
H 0.757000 0.586000 0.000000
H -0.757000 0.586000 0.000000"""

    print("Test coordinates:")
    print(test_coords)

    # Test parsing dual bond axes string
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
    
    # Assertions for pytest - result is a tuple of (mu1, mu2)
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
    
    print("✅ Dual bond axes test completed successfully!")

if __name__ == "__main__":
    print("Dual Bond Axes Test for Morse Solver")
    print("="*50)
    
    try:
        test_dual_bond_axes()
        print("\n✅ Dual bond axes test passed!")
        print("Your environment is properly configured.")
    except Exception:
        print("\n❌ Test failed!")
        print("There may be issues with your PySCF installation.")
        raise