"""
Quick test script for debugging CCSD hanging issues.
This script runs a minimal calculation to verify the environment works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from derivatives_dipole_moment import optimize_geometry_ccsd

def test_quick_optimization():
    """Test with a simple H2 molecule using CCSD(T) optimization."""
    
    # Simple H2 molecule
    h2_coords = """H 0.0 0.0 0.0
H 0.0 0.0 0.74"""
    
    print("Testing CCSD optimization with H2 molecule...")
    print("Initial coordinates:")
    print(h2_coords)
    print()
    
    try:
        # Test CCSD optimization
        result = optimize_geometry_ccsd(
            coords_string=h2_coords,
            specified_spin=0,  # singlet
            basis="STO-3G",    # smallest basis set
            maxsteps=10        # few optimization steps
        )
        
        print("HF optimization completed successfully!")
        print("Optimized coordinates:")
        print(result)
        return True
        
    except Exception as e:
        print(f"HF optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Quick Test for Morse Solver Environment")
    print("="*50)
    
    success = test_quick_optimization()
    
    if success:
        print("\n✅ Basic optimization works!")
        print("Your environment is properly configured.")
    else:
        print("\n❌ Basic optimization failed!")
        print("There may be issues with your PySCF installation.")
    