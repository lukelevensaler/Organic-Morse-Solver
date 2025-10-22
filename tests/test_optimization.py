"""
Quick test script for debugging CCSD hanging issues.
This script runs a minimal calculation to verify the environment works.
"""

import sys
import os

# Add the parent directory to the path so we can import our module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_quick_optimization():
    """Test with a simple H2 molecule using CCSD(T) optimization."""
    
    try:
        from derivatives_dipole_moment import optimize_geometry_ccsd
        print("Successfully imported derivatives_dipole_moment module")
    except ImportError as e:
        print(f"Failed to import module: {e}")
        raise
    
    # Simple H2 molecule
    h2_coords = """H 0.0 0.0 0.0
H 0.0 0.0 0.74"""
 
    print("Testing CCSD optimization with H2 molecule...")
    print("Initial coordinates:")
    print(h2_coords)
    print()
    
    # Test CCSD optimization
    result = optimize_geometry_ccsd(
        coords_string=h2_coords,
        specified_spin=0,  # singlet
        basis="STO-3G",    # smallest basis set
        maxsteps=10        # few optimization steps
    )
    
    print("CCSD optimization completed successfully!")
    print("Optimized coordinates:")
    print(result)
    
    # Add proper assertions for pytest
    assert result is not None
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    assert "H" in result
    assert result.count("H") == 2  # Should have 2 hydrogen atoms
    
    print("✅ Basic optimization test passed!")

if __name__ == "__main__":
    print("Quick Test for Morse Solver Environment")
    print("="*50)
    
    try:
        test_quick_optimization()
        print("\n✅ Basic optimization works!")
        print("Your environment is properly configured.")
    except Exception as e:
        print(f"\n❌ Basic optimization failed: {e}")
        print("There may be issues with your PySCF installation.")
        import traceback
        traceback.print_exc()
        raise
    