"""
Quick test script for debugging CCSD hanging issues.
This script runs a minimal calculation to verify the environment works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from derivatives_dipole_moment import optimize_geometry_hf

def test_quick_optimization():
    """Test with a simple H2 molecule using HF optimization."""
    
    # Simple H2 molecule
    h2_coords = """H 0.0 0.0 0.0
H 0.0 0.0 0.74"""
    
    print("Testing HF optimization with H2 molecule...")
    print("Initial coordinates:")
    print(h2_coords)
    print()
    
    try:
        # Test HF optimization first (much faster)
        result = optimize_geometry_hf(
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

def print_recommendations():
    """Print recommendations for avoiding CCSD hanging."""
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS TO AVOID CCSD HANGING:")
    print("="*60)
    print()
    print("1. USE SMALLER BASIS SETS:")
    print("   - Instead of aug-cc-pV5Z, try: aug-cc-pVDZ or cc-pVDZ")
    print("   - For initial testing: 6-31G* or STO-3G")
    print()
    print("2. START WITH HF OPTIMIZATION:")
    print("   - Test with HF first: --disable-ccsd flag")
    print("   - HF is much faster and more stable")
    print()
    print("3. SYSTEM SIZE LIMITS:")
    print("   - CCSD scales as O(N^6) with system size")
    print("   - For molecules >10 atoms, consider DFT instead")
    print()
    print("4. MEMORY REQUIREMENTS:")
    print("   - CCSD needs significant RAM (>4GB for medium systems)")
    print("   - Close other applications during calculation")
    print()
    print("5. CONVERGENCE ISSUES:")
    print("   - If SCF doesn't converge, CCSD will fail")
    print("   - Try different initial geometries")
    print()
    print("6. ALTERNATIVE METHODS:")
    print("   - MP2: faster than CCSD, similar accuracy")
    print("   - DFT (B3LYP, etc.): much faster for large systems")
    print("="*60)

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
    
    print_recommendations()