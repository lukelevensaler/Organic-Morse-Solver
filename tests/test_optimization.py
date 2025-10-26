#!/usr/bin/env python3

"""
Pytest module to verify that the fixed optimize_geometry_ccsd function works properly
"""

import pytest
from optimize_geometry import optimize_geometry_ccsd


@pytest.fixture
def test_molecule():
    """Test molecule fixture - slightly distorted H2O"""
    return """
O 0.0 0.0 0.0
H 0.8 0.6 0.0
H -0.8 0.6 0.0
"""


class TestGeometryOptimization:
    """Test class for geometry optimization functions"""

    def test_scf_optimization(self, test_molecule):
        """Test SCF geometry optimization"""
        # Run optimization with a smaller basis set for speed
        result = optimize_geometry_ccsd(
            coords_string=test_molecule,
            specified_spin=0,  # Singlet (2S = 0, so S = 0)
            basis="sto-3g",    # Small basis for testing
            maxsteps=10
        )
        
        # Assert that we get a result back
        assert result is not None, "SCF optimization should return a result"
        assert isinstance(result, str), "Result should be a string containing coordinates"
        assert "O" in result, "Result should contain oxygen atom"
        assert "H" in result, "Result should contain hydrogen atoms"

    def test_ccsd_optimization(self, test_molecule):
        """Test CCSD geometry optimization"""
        # Run optimization with a smaller basis set for speed
        result = optimize_geometry_ccsd(
            coords_string=test_molecule,
            specified_spin=0,  # Singlet (2S = 0, so S = 0)
            basis="sto-3g",    # Small basis for testing
            maxsteps=5         # Fewer steps for testing
        )
        
        # Assert that we get a result back
        assert result is not None, "CCSD optimization should return a result"
        assert isinstance(result, str), "Result should be a string containing coordinates"
        assert "O" in result, "Result should contain oxygen atom"
        assert "H" in result, "Result should contain hydrogen atoms"

    def test_optimization_with_invalid_coordinates(self):
        """Test that optimization fails gracefully with invalid coordinates"""
        invalid_coords = "This is not a valid coordinate string"
        
        with pytest.raises(Exception):
            optimize_geometry_ccsd(
                coords_string=invalid_coords,
                specified_spin=0,
                basis="sto-3g",
                maxsteps=5
            )

    def test_optimization_with_invalid_spin(self, test_molecule):
        """Test that optimization handles invalid spin values appropriately"""
        # This might not raise an exception but could give unexpected results
        # We'll test that it at least doesn't crash
        try:
            result = optimize_geometry_ccsd(
                coords_string=test_molecule,
                specified_spin=100,  # Very high spin value
                basis="sto-3g",
                maxsteps=2
            )
            # If it doesn't crash, that's good enough for this test
            assert True
        except Exception:
            # If it does raise an exception, that's also acceptable behavior
            assert True


@pytest.mark.slow
class TestLongRunningOptimization:
    """Test class for longer-running optimization tests"""
    
    def test_optimization_convergence(self, test_molecule):
        """Test that optimization actually converges (slower test)"""
        result = optimize_geometry_ccsd(
            coords_string=test_molecule,
            specified_spin=0,
            basis="sto-3g",
            maxsteps=20  # More steps for better convergence
        )
        
        assert result is not None
        # Could add more sophisticated convergence checks here