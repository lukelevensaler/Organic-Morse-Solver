import os
import time
from typing import Any
import numpy as np
from pyscf import gto, scf
from pyscf import grad
from tqdm import tqdm

try:
	# geomopt Berny solver for correlated gradients
	from pyscf.geomopt import berny_solver
except Exception:
	berny_solver = None

# First we optimize the provided geometry 
# Bond axis-related computations are outsourced to normalize_bonds.py
def optimize_geometry_scf(coords_string: str, specified_spin: int, basis: str | None = None, maxsteps: int = 50) -> str:
	"""
	Run an SCF geometry optimization (via SCF gradients + Berny).

	coords_string: either a path to a file or a multiline XYZ-style string (Element x y z)
	specified_spin: spin multiplicity value
	basis: basis set name supplied by the user (required)
	maxsteps: maximum Berny steps

	Returns an XYZ-style block string with optimized coordinates (same format as input).

	"""
	if basis is None:
		raise ValueError("optimize_geometry_scf requires a user-specified basis set")

	# read coords
	if os.path.isfile(coords_string):
		with open(coords_string, 'r') as fh:
			coord_text = fh.read()
	else:
		coord_text = coords_string

	# parse as in compute_µ_derivatives
	lines = [ln.strip() for ln in coord_text.splitlines() if ln.strip()]
	atoms = []
	positions = []
	for ln in lines:
		parts = ln.split()
		if len(parts) != 4:
			raise ValueError(f"Invalid coordinate line: '{ln}'. Expected 'Element x y z'.")
		atoms.append(parts[0])
		positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
	positions = np.array(positions, dtype=float)

	# Deterministic SCF-based optimization
	if berny_solver is None or grad is None:
		msg = (
			"PySCF berny solver or grad module not available; cannot perform geometry optimization.\n"
			"Install full PySCF with geomopt support, or disable optimization in the workflow.\n"
		)
		raise RuntimeError(msg)

	# Build molecule and run a single SCF
	mol = gto.M(
		atom="\n".join(f"{a} {x} {y} {z}" for a, (x, y, z) in zip(atoms, positions)),
		basis=basis,
		spin=specified_spin,
		unit="Angstrom",
	)
	with tqdm(desc="SCF for Geometry Optimization", unit="step", colour='blue') as pbar:
		mf = scf.UHF(mol) if specified_spin != 0 else scf.RHF(mol)
		mf.conv_tol = 1e-8
		mf.max_cycle = 400
		mf.kernel()
		pbar.update(1)
		pbar.set_postfix(converged=getattr(mf, "converged", False), energy=f"{getattr(mf, 'e_tot', float('nan')):.6f}")

	if not getattr(mf, "converged", False):
		print("WARNING: SCF for geometry optimization did not reach target conv_tol; proceeding with last density.")

	# Always use SCF gradients (UHF/RHF) deterministically
	with tqdm(desc="Building SCF Gradients", unit="step", colour='cyan') as pbar:
		print("Building SCF gradients")
		g = grad.UHF(mf) if specified_spin != 0 else grad.RHF(mf)
		pbar.update(1)

	with tqdm(desc="Testing Gradient Calculation", unit="step", colour='yellow') as pbar:
		print("Testing gradient calculation")
		print(f"Gradient object type: {type(g)}")
		grad_array = g.kernel()
		if isinstance(grad_array, np.ndarray) and grad_array.size > 0:
			print(f"Gradient test successful, shape: {grad_array.shape}")
			print(f"Gradient norm: {np.linalg.norm(grad_array):.8f}")
			pbar.update(1)
			pbar.set_postfix(shape=f"{grad_array.shape}")
		else:
			raise RuntimeError(f"Gradients returned invalid result: {type(grad_array)}")

	assert berny_solver is not None
	print("Starting deterministic SCF berny geometry optimization...")
	print("Solving Berny geometry optimization with SCF gradients...")
	start_time = time.time()
	mol_opt: Any = berny_solver.optimize(g, maxsteps=int(maxsteps))
	elapsed = time.time() - start_time
	print(f"Berny geometry optimization completed in {elapsed:.1f}s")

	if not hasattr(mol_opt, 'atom_coords') or not hasattr(mol_opt, 'atom_symbol'):
		if hasattr(mol_opt, 'atom_coord'):
			coords_opt = mol_opt.atom_coord()
			symbols = mol_opt.atom_symbol() if hasattr(mol_opt, 'atom_symbol') else [mol_opt.atom_pure_symbol(i) for i in range(len(coords_opt))]
		elif hasattr(mol_opt, 'atom'):
			coords_opt = mol_opt.atom_coords()
			symbols = [mol_opt.atom_symbol(i) for i in range(mol_opt.natm)]
		else:
			raise RuntimeError("Cannot extract optimized coordinates from berny result")
	else:
		coords_opt = mol_opt.atom_coords()
		symbols = [mol_opt.atom_symbol(i) for i in range(len(coords_opt))]

	optimized_coords = "\n".join(f"{symbols[i]} {x:.6f} {y:.6f} {z:.6f}" for i, (x, y, z) in enumerate(coords_opt))
	print("Deterministic SCF geometry optimization completed successfully!")
	return optimized_coords
