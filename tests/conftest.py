"""
Pytest configuration and shared fixtures for Organic Morse Solver tests
"""

import pytest


WATER_TRIPLE_STRING = """
O 0.0 0.0 0.0
H 0.8 0.6 0.0
H -0.8 0.6 0.0
"""


@pytest.fixture(scope="session")
def water_coords():
    """Shared distorted water geometry used across tests."""
    return WATER_TRIPLE_STRING


@pytest.fixture(scope="session")
def sample_molecules():
    """Fixture providing various test molecules"""
    molecules = {
        "h2o": WATER_TRIPLE_STRING,
        "h2o_distorted": WATER_TRIPLE_STRING,
        "h2": """
H 0.0 0.0 0.0
H 0.0 0.0 0.74
""",
        "nh3": """
N 0.0 0.0 0.0
H 0.0 0.9377 -0.3816
H 0.8121 -0.4688 -0.3816
H -0.8121 -0.4688 -0.3816
"""
    }
    return molecules


@pytest.fixture
def tolerance_settings():
    """Fixture providing common tolerance settings for numerical comparisons"""
    return {
        "energy_tolerance": 1e-8,
        "gradient_tolerance": 1e-6,
        "coordinate_tolerance": 1e-4,
        "convergence_tolerance": 1e-6
    }


@pytest.fixture(scope="session")
def test_basis_sets():
    """Fixture providing various basis sets for testing"""
    return {
        "minimal": "sto-3g",
        "double_zeta": "6-31g",
        "triple_zeta": "6-311g",
        "aug_double_zeta": "aug-cc-pvdz"
    }


def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    config.addinivalue_line(
        "markers", "requires_pyscf: mark test as requiring PySCF installation"
    )
    config.addinivalue_line(
        "markers", "computational: mark test as computationally intensive"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    for item in items:
        # Mark tests with 'scf' in name as slow
        if "scf" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark tests with 'optimization' in name as computational
        if "optimization" in item.name.lower():
            item.add_marker(pytest.mark.computational)
        
        # Mark all tests as requiring pyscf
        item.add_marker(pytest.mark.requires_pyscf)


@pytest.fixture
def suppress_pyscf_output():
    """Fixture to suppress PySCF output during tests"""
    import sys
    from io import StringIO
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    yield
    
    # Restore stdout
    sys.stdout = old_stdout