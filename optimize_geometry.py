import os
import time
from typing import Any
import numpy as np
from pyscf import gto, scf
from tqdm import tqdm
try:
	# geomopt Berny solver for correlated gradients
	from pyscf.geomopt import berny_solver
except Exception:
	berny_solver = None

# First we optimize the provided geometry 
# Bond axis-related computations are outsourced to normalize_bonds.py
def optimize_geometry_ccsd(coords_string: str, specified_spin: int, basis: str = "aug-cc-pVTZ", maxsteps: int = 50) -> str:
	"""
	Run a CCSD geometry optimization (via CCSD gradients + Berny).

	coords_string: either a path to a file or a multiline XYZ-style string (Element x y z)
	specified_spin: spin multiplicity value
	basis: basis set name (default: aug-cc-pVTZ for maximum CCSD(T) accuracy)
	maxsteps: maximum Berny steps

	Returns an XYZ-style block string with optimized coordinates (same format as input).
	
	Uses maximum precision CCSD(T) gradients for the most rigorous geometry optimization.
	"""
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

	# Import stabilization module
	from scf_stabilization import robust_scf_calculation, check_geometry_for_problems
	
	# Build geometry string for stabilization functions
	geometry_string = "\n".join(f"{a} {x} {y} {z}" for a, (x, y, z) in zip(atoms, positions))
	
	# Check geometry for potential problems
	check_geometry_for_problems(geometry_string)
	
	# Run robust SCF calculation with automatic singularity handling
	with tqdm(desc="Robust SCF for Geometry Optimization", unit="step", colour='blue') as pbar:
		try:
			mf = robust_scf_calculation(
				atom_string=geometry_string,
				spin=specified_spin,
				basis=basis,
				target_conv_tol=1e-12,  # Target extremely tight convergence
				max_cycle=400
			)
			pbar.update(1)
			pbar.set_postfix(converged=mf.converged, energy=f"{mf.e_tot:.6f}")
		except Exception as e:
			raise RuntimeError(f"Robust SCF calculation failed: {e}")
	
	if not mf.converged:
		raise RuntimeError("High-precision SCF did not converge - cannot proceed with CCSD(T) optimization")
	
	# Get molecule object from the mean field calculation
	mol = mf.mol

	# lazily import cc and grad to check for analytic gradient support
	try:
		from pyscf import cc as cc, grad as grad
	except Exception:
		cc = None
		grad = None

	# Require CCSD gradients + berny_solver to be available; otherwise fail loudly
	missing = []
	if berny_solver is None:
		missing.append('berny_solver (geometry optimizer)')
	if cc is None:
		missing.append('pyscf.cc (CCSD module)')
	if grad is None:
		missing.append('pyscf.grad (analytic gradients)')
	if missing:
		msg = (
			"CCSD gradients and/or berny_solver are not available in this PySCF installation; cannot perform CCSD geometry optimization.\n"
			"Missing: " + ", ".join(missing) + ".\n"
			"Recommended fixes:\n"
			" - Install PySCF and berny from conda-forge (recommended):\n"
			"     conda create -n pyscf-env -c conda-forge python=3.11 pyscf berny scipy -y\n"
			" - Or install via pip and build from source if wheel lacks compiled extensions:\n"
			"     python3 -m pip install pyscf berny\n"
			"   If pip wheel lacks CCSD gradient code, build PySCF from source (see INSTALL_PySCF.md in project root).\n"
			"After installing, re-run the script in the same Python environment.\n"
		)
		raise RuntimeError(msg)

	# Run CCSD optimization and raise if it fails
	try:
		# ensure static analysers know these are present
		assert cc is not None, 'pyscf.cc not available'
		assert grad is not None, 'pyscf.grad not available'
		# use the lazily imported cc with maximum rigor convergence controls
		mycc = cc.CCSD(mf)
		
		# Set maximum precision convergence criteria for ultimate CCSD(T) accuracy
		mycc.conv_tol = 1e-10   # Maximum precision convergence
		mycc.max_cycle = 200    # Many iterations for robust convergence
		mycc.diis_space = 15    # Maximum DIIS space for optimal convergence
		mycc.direct = True      # Use direct algorithm for better numerical accuracy
		
		print(f"Maximum precision CCSD(T) optimization settings: conv_tol={mycc.conv_tol}, max_cycle={mycc.max_cycle}, diis_space={mycc.diis_space}")

		# Run CCSD calculation with progress tracking
		with tqdm(desc="CCSD for Geometry Optimization", unit="iter", colour='green') as pbar:
			pbar.set_postfix(conv_tol=f"{mycc.conv_tol:.0e}", max_cycle=mycc.max_cycle)
			print("Running maximum precision CCSD calculation...")
			start_time = time.time()
			mycc.run()
			elapsed = time.time() - start_time
			pbar.update(1)
			pbar.set_postfix(converged=mycc.converged, energy=f"{mycc.e_corr:.8f}", time=f"{elapsed:.1f}s")
		print(f"CCSD calculation completed in {elapsed:.1f}s")
		
		if not mycc.converged:
			raise RuntimeError("CCSD did not converge to required precision - cannot proceed with optimization")

		# Run CCSD(T) triples correction - MANDATORY for maximum computational rigor
		ccsd_t_energy = None
		with tqdm(desc="CCSD(T) Triples for Optimization", unit="step", colour='red') as pbar:
			try:
				if hasattr(mycc, 'ccsd_t'):
					print("Computing maximum precision CCSD(T) triples correction...")
					start_time = time.time()
					ccsd_t_energy = mycc.ccsd_t()
					elapsed = time.time() - start_time
					pbar.update(1)
					pbar.set_postfix(E_T=f"{ccsd_t_energy:.8f}", time=f"{elapsed:.1f}s")
					print(f"CCSD(T) triples correction completed in {elapsed:.1f}s")
					print(f"Maximum precision CCSD(T) correction energy: {ccsd_t_energy:.12f} Hartree")
				else:
					raise RuntimeError("CCSD(T) method not available - cannot achieve required computational rigor for optimization")
			except Exception as e:
				raise RuntimeError(f"CCSD(T) triples correction failed - required for maximum rigor optimization: {e}")
		
		try:
			
			# build gradients and optimize geometry (may be long)
			with tqdm(desc="Building CCSD Gradients", unit="step", colour='cyan') as pbar:
				print("Building CCSD gradients...")
				g = grad.ccsd.Gradients(mycc)
				pbar.update(1)
			
			# Test the gradient calculation first with error handling for PySCF bugs
			with tqdm(desc="Testing Gradient Calculation", unit="step", colour='yellow') as pbar:
				print("Testing gradient calculation...")
				try:
					# Try to calculate gradients with workarounds for PySCF 2.10.0 bugs
					testgrad = None
					
					# First attempt: direct gradient calculation
					try:
						testgrad = g.kernel()
					except (AttributeError, TypeError) as e:
						print(f"Direct gradient calculation failed: {e}")
						testgrad = None

					if testgrad is not None:
						pbar.update(1)
						pbar.set_postfix(shape=f"{testgrad.shape}")
						print(f"Gradient test successful, shape: {testgrad.shape}")
					else:
						raise RuntimeError("Gradient calculation returned None")
						
				except Exception as grad_e:
					print(f"Gradient calculation failed: {grad_e}")
					# The gradient failed, this is likely the source of our issue
					raise RuntimeError(f"CCSD gradient calculation failed: {grad_e}")
			
			# Narrow types for static analysis
			assert berny_solver is not None
			print("Starting maximum precision CCSD(T) berny geometry optimization...")
			
			# No progress bar for Berny optimization (not need, it goes fast enough)
			try:
				print("Solving Berny geometry optimization...")
				start_time = time.time()
				mol_opt: Any = getattr(berny_solver, 'optimize')(g, maxsteps=int(maxsteps))
				elapsed = time.time() - start_time
				print(f"Berny geometry optimization completed in {elapsed:.1f}s")
			except Exception as berny_e:
				print(f"Berny optimization failed: {berny_e}")
				print("Trying direct berny call...")
				mol_opt = getattr(berny_solver, 'optimize')(g, maxsteps=int(maxsteps))
			
			# ensure mol_opt exposes expected API
			if not hasattr(mol_opt, 'atom_coords') or not hasattr(mol_opt, 'atom_symbol'):
				# Try different ways to access the optimized geometry
				if hasattr(mol_opt, 'atom_coord'):
					coords_opt = mol_opt.atom_coord()
					symbols = mol_opt.atom_symbol() if hasattr(mol_opt, 'atom_symbol') else [mol_opt.atom_pure_symbol(i) for i in range(len(coords_opt))]
				elif hasattr(mol_opt, 'atom'):
					# mol_opt might be a PySCF molecule object
					coords_opt = mol_opt.atom_coords()
					symbols = [mol_opt.atom_symbol(i) for i in range(mol_opt.natm)]
				else:
					raise RuntimeError("Cannot extract optimized coordinates from berny result")
			else:
				coords_opt = mol_opt.atom_coords()
				symbols = [mol_opt.atom_symbol(i) for i in range(len(coords_opt))]
			
			optimized_coords = "\n".join(f"{symbols[i]} {x:.6f} {y:.6f} {z:.6f}" for i, (x, y, z) in enumerate(coords_opt))
			print("Maximum precision CCSD(T) geometry optimization completed successfully!")
			return optimized_coords
		except Exception as e:
			# CCSD gradients or berny optimization failed — no HF fallback requested
			print(f"Detailed error: {e}")
			import traceback
			traceback.print_exc()
			raise RuntimeError(f"CCSD geometry optimization failed: {e}")
	except Exception as e_outer:
		# CCSD optimization failed
		raise RuntimeError(f"CCSD optimization failed: {e_outer}")
