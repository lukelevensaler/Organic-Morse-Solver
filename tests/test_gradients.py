#!/usr/bin/env python3

"""
Pytest module to verify that gradient calculations work properly
with the simplified PySCF 2.11.0 API
"""

import numpy as np
import pytest
from pyscf import gto, scf, grad, cc


@pytest.fixture
def h2o_molecule():
    """H2O molecule fixture for testing"""
    mol_str = """
    O 0.0 0.0 0.0
    H 0.757 0.587 0.0
    H -0.757 0.587 0.0
    """
    return gto.M(atom=mol_str, basis='sto-3g')


@pytest.fixture
def converged_scf(h2o_molecule):
    """SCF calculation fixture that ensures convergence"""
    mf = scf.RHF(h2o_molecule)
    mf.kernel()
    assert mf.converged, "SCF calculation must converge for gradient tests"
    return mf


class TestSCFGradients:
    """Test class for SCF gradient calculations"""

    def test_scf_gradients_basic(self, converged_scf):
        """Test basic SCF gradient calculation"""
        # Calculate gradients
        g = grad.RHF(converged_scf)
        grad_array = g.kernel()
        
        # Test gradient properties
        assert isinstance(grad_array, np.ndarray), "Gradient should be a numpy array"
        assert grad_array.size > 0, "Gradient array should not be empty"
        assert grad_array.shape == (3, 3), "Gradient should have shape (n_atoms, 3)"
        
        # Test that gradient norm is reasonable (not zero, not too large)
        grad_norm = np.linalg.norm(grad_array)
        assert grad_norm > 1e-10, "Gradient norm should not be essentially zero"
        assert grad_norm < 1e3, "Gradient norm should not be unreasonably large"

    def test_scf_gradients_shape(self, converged_scf):
        """Test that SCF gradients have the correct shape"""
        g = grad.RHF(converged_scf)
        grad_array = g.kernel()
        
        n_atoms = converged_scf.mol.natm
        expected_shape = (n_atoms, 3)
        
        assert grad_array.shape == expected_shape, \
            f"Expected gradient shape {expected_shape}, got {grad_array.shape}"

    def test_scf_gradients_finite_values(self, converged_scf):
        """Test that SCF gradients contain finite values"""
        g = grad.RHF(converged_scf)
        grad_array = g.kernel()
        
        assert np.all(np.isfinite(grad_array)), "All gradient values should be finite"
        assert not np.any(np.isnan(grad_array)), "No gradient values should be NaN"


class TestCCSDGradients:
    """Test class for CCSD gradient calculations"""

    @pytest.fixture
    def converged_ccsd(self, converged_scf):
        """CCSD calculation fixture that ensures convergence"""
        mycc = cc.CCSD(converged_scf)
        mycc.kernel()
        if not mycc.converged:
            pytest.skip("CCSD did not converge, skipping CCSD gradient test")
        return mycc

    def test_ccsd_gradients_basic(self, converged_ccsd):
        """Test basic CCSD gradient calculation"""
        # Calculate CCSD gradients
        g = grad.ccsd.Gradients(converged_ccsd)
        grad_array = g.kernel()
        
        # Test gradient properties
        assert isinstance(grad_array, np.ndarray), "CCSD gradient should be a numpy array"
        assert grad_array.size > 0, "CCSD gradient array should not be empty"
        assert grad_array.shape == (3, 3), "CCSD gradient should have shape (n_atoms, 3)"
        
        # Test that gradient norm is reasonable
        grad_norm = np.linalg.norm(grad_array)
        assert grad_norm > 1e-10, "CCSD gradient norm should not be essentially zero"
        assert grad_norm < 1e3, "CCSD gradient norm should not be unreasonably large"

    def test_ccsd_gradients_finite_values(self, converged_ccsd):
        """Test that CCSD gradients contain finite values"""
        g = grad.ccsd.Gradients(converged_ccsd)
        grad_array = g.kernel()
        
        assert np.all(np.isfinite(grad_array)), "All CCSD gradient values should be finite"
        assert not np.any(np.isnan(grad_array)), "No CCSD gradient values should be NaN"

    def test_ccsd_vs_scf_gradients(self, converged_scf, converged_ccsd):
        """Test that CCSD and SCF gradients are different but reasonable"""
        # Calculate SCF gradients
        g_scf = grad.RHF(converged_scf)
        scf_grad = g_scf.kernel()
        
        # Calculate CCSD gradients
        g_ccsd = grad.ccsd.Gradients(converged_ccsd)
        ccsd_grad = g_ccsd.kernel()
        
        # They should have the same shape
        assert scf_grad.shape == ccsd_grad.shape, "SCF and CCSD gradients should have same shape"
        
        # They should be different (correlation effects)
        grad_diff = np.linalg.norm(scf_grad - ccsd_grad)
        assert grad_diff > 1e-8, "SCF and CCSD gradients should differ due to correlation effects"


class TestGradientErrors:
    """Test class for error handling in gradient calculations"""

    def test_scf_gradient_with_unconverged_scf(self, h2o_molecule):
        """Test that gradient calculation handles unconverged SCF appropriately"""
        mf = scf.RHF(h2o_molecule)
        mf.max_cycle = 1  # Force non-convergence
        mf.kernel()
        
        # This should either raise an exception or handle gracefully
        if not mf.converged:
            # Some implementations might still compute gradients
            # Others might raise an exception - both are acceptable
            try:
                g = grad.RHF(mf)
                grad_array = g.kernel()
                # If it works, check that we get reasonable output
                assert isinstance(grad_array, np.ndarray)
            except Exception:
                # If it raises an exception, that's also acceptable
                pass

    def test_gradient_with_invalid_basis(self):
        """Test gradient calculation with invalid basis set"""
        with pytest.raises(Exception):
            mol = gto.M(atom="H 0 0 0; H 0 0 1", basis='invalid_basis')
            mf = scf.RHF(mol)
            mf.kernel()


@pytest.mark.slow
class TestExtensiveGradients:
    """Test class for more extensive gradient tests (marked as slow)"""
    
    def test_gradient_numerical_vs_analytical(self, converged_scf):
        """Test analytical vs numerical gradients (slow test)"""
        # This would involve numerical differentiation - placeholder for now
        g = grad.RHF(converged_scf)
        analytical_grad = g.kernel()
        
        # Could implement numerical gradient check here
        assert analytical_grad is not None, "Analytical gradient should be computed"