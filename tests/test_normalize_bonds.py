#!/usr/bin/env python3
"""Tests for bond normalization helpers in normalize_bonds.py."""

import numpy as np

from normalize_bonds import (
    parse_dual_bond_axes,
    create_symmetric_antisymmetric_vectors,
    process_bond_displacements,
)

# Constants used across normalization tests
DUAL_BOND_AXES = "(2,1);(3,1)"  # Two O-H stretches in distorted water
HYDROGEN_MASS = 1.00784
OXYGEN_MASS = 15.999


def parse_xyz_block(xyz_block: str) -> tuple[list[str], np.ndarray]:
    """Convert XYZ text block into atom labels and coordinate array."""
    atoms: list[str] = []
    coords: list[list[float]] = []
    for line in xyz_block.strip().splitlines():
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"Invalid coordinate line: {line}")
        symbol, x_val, y_val, z_val = parts
        atoms.append(symbol)
        coords.append([float(x_val), float(y_val), float(z_val)])
    return atoms, np.array(coords, dtype=float)


def test_parse_dual_bond_axes_returns_zero_based_pairs():
    """Dual bond axes parser should convert to zero-based index tuples."""
    bond1, bond2 = parse_dual_bond_axes(DUAL_BOND_AXES)
    assert bond1 == (1, 0)
    assert bond2 == (2, 0)


def test_create_symmetric_antisymmetric_vectors_unit_norm(water_coords):
    """Mass-weighted symmetric/antisymmetric vectors stay normalized."""
    atoms, positions = parse_xyz_block(water_coords)
    bond1, bond2 = parse_dual_bond_axes(DUAL_BOND_AXES)

    e_sym, e_anti = create_symmetric_antisymmetric_vectors(
        positions,
        atoms,
        bond1,
        bond2,
        HYDROGEN_MASS,
        OXYGEN_MASS,
    )

    assert e_sym.shape == (3 * len(atoms),)
    assert e_anti.shape == (3 * len(atoms),)
    assert np.isclose(np.linalg.norm(e_sym), 1.0)
    assert np.isclose(np.linalg.norm(e_anti), 1.0)
    assert np.isclose(float(np.dot(e_sym, e_anti)), 0.0)


def test_process_bond_displacements_dual_axes_produces_mass_weighted_modes(water_coords):
    """process_bond_displacements should honor dual bond axes setup."""
    atoms, positions = parse_xyz_block(water_coords)
    bond1, bond2 = parse_dual_bond_axes(DUAL_BOND_AXES)

    e_sym, _ = create_symmetric_antisymmetric_vectors(
        positions,
        atoms,
        bond1,
        bond2,
        HYDROGEN_MASS,
        OXYGEN_MASS,
    )

    pos_plus, pos_minus, formatter, direction = process_bond_displacements(
        positions,
        atoms,
        dual_bond_axes=DUAL_BOND_AXES,
        delta=0.01,
        m1=HYDROGEN_MASS,
        m2=OXYGEN_MASS,
    )

    assert pos_plus.shape == positions.shape
    assert pos_minus.shape == positions.shape
    assert direction.shape == positions.shape

    # Displacements should be symmetric around the original geometry
    expected_diff = 2.0 * 0.01 * direction
    assert np.allclose(pos_plus - pos_minus, expected_diff)
    assert np.isclose(np.linalg.norm(direction.ravel()), 1.0)

    formatted = formatter(pos_plus)
    assert formatted.splitlines()[0].startswith("O ")
    assert "H" in formatted

    # Direction vector should match the symmetric mode from normalize_bonds
    assert np.allclose(direction.reshape(-1), e_sym, atol=1e-12)