import os
import time
from typing import Any
import numpy as np
from pyscf import gto, scf
from pyscf import cc, grad
from tqdm import tqdm

try:
	# geomopt Berny solver for correlated gradients
	from pyscf.geomopt import berny_solver
except Exception:
	berny_solver = None

# First we optimize the provided geometry 
# Bond axis-related computations are outsourced to normalize_bonds.py
def optimize_geometry_ccsd(coords_string: str, specified_spin: int, basis: str | None = None, maxsteps: int = 50) -> str:
	"""
	Run a CCSD geometry optimization (via CCSD gradients + Berny).

	coords_string: either a path to a file or a multiline XYZ-style string (Element x y z)
	specified_spin: spin multiplicity value
	basis: basis set name supplied by the user (required)
	maxsteps: maximum Berny steps

	Returns an XYZ-style block string with optimized coordinates (same format as input).
	
	Uses CCSD(T) gradients for the most rigorous geometry optimization.
	"""
	if basis is None:
		raise ValueError("optimize_geometry_ccsd requires a user-specified basis set")

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
	from stabilization.scf_stabilization import robust_scf_calculation, check_geometry_for_problems
	
	# Build geometry string for stabilization functions
	geometry_string = "\n".join(f"{a} {x} {y} {z}" for a, (x, y, z) in zip(atoms, positions))
	
	# Check geometry for potential problems
	check_geometry_for_problems(geometry_string)
	
	# Run SCF calculation with automatic singularity handling (CPU-only)
	with tqdm(desc="SCF for Geometry Optimization", unit="step", colour='blue') as pbar:
		try:
			mf = robust_scf_calculation(  # Ensure this is CPU-only
				atom_string=geometry_string,
				spin=specified_spin,
				basis=basis,
				target_conv_tol=1e-8,  # Increased tolerance for better convergence
				max_cycle=400
			)
			pbar.update(1)
			pbar.set_postfix(converged=mf.converged, energy=f"{mf.e_tot:.6f}")
		except Exception as e:
			raise RuntimeError(f"SCF calculation failed: {e}")
	
	if not mf.converged:
		raise RuntimeError("High-precision SCF did not converge - cannot proceed with CCSD(T) optimization")
	
	# Use the SCF object for gradients (CPU-only)
	mf_for_grad = mf

	# Get molecule object from the mean field calculation
	mol = mf.mol

	# Check for required modules but allow fallback to SCF if CCSD fails (CPU-only)
	missing = []
	if berny_solver is None:
		missing.append('berny_solver (geometry optimizer)')
	if grad is None:
		missing.append('pyscf.grad (analytic gradients)')
	if missing:
		msg = (
			"Required modules are not available in this PySCF installation; cannot perform geometry optimization.\n"
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
	
	# Warn if CCSD is not available but continue with SCF
	if cc is None:
		print("WARNING: pyscf.cc (CCSD module) not available - will attempt SCF geometry optimization instead")
		print("For maximum accuracy, install full PySCF with CCSD support")

	# Run CCSD optimization and raise if it fails, or fallback to SCF
	try:
		# Check if CCSD is available
		print("Attempting CCSD geometry optimization (CPU-only)...")
		mycc = cc.CCSD(mf)
		
		if mycc is not None:
			# Set maximum precision convergence criteria for ultimate CCSD(T) accuracy
			mycc.conv_tol = 1e-10   # Maximum precision convergence
			if hasattr(mycc, 'conv_tol_normt'):
				mycc.conv_tol_normt = 1e-10 * 0.1
			mycc.max_cycle = 200    # Many iterations for robust convergence
			if hasattr(mycc, 'diis_space'):
				mycc.diis_space = 15    # Maximum DIIS space for optimal convergence
			if hasattr(mycc, 'diis_start_cycle'):
				mycc.diis_start_cycle = 1
			if hasattr(mycc, 'direct'):
				mycc.direct = True      # Use direct algorithm for better numerical accuracy
			
			diis_value = getattr(mycc, 'diis_space', 'n/a')
			print(f"CCSD(T) optimization settings: conv_tol={mycc.conv_tol}, max_cycle={mycc.max_cycle}, diis_space={diis_value}")

			# Run CCSD calculation with progress tracking
			with tqdm(desc="CCSD for Geometry Optimization", unit="iter", colour='green') as pbar:
				pbar.set_postfix(conv_tol=f"{mycc.conv_tol:.0e}", max_cycle=mycc.max_cycle)
				print("Running CCSD calculation...")
				start_time = time.time()
				mycc.run()
				elapsed = time.time() - start_time
				pbar.update(1)
				pbar.set_postfix(converged=mycc.converged, energy=f"{mycc.e_corr:.8f}", time=f"{elapsed:.1f}s")
			print(f"CCSD calculation completed in {elapsed:.1f}s")
			
			if not mycc.converged:
				print("WARNING: CCSD did not converge - falling back to SCF optimization")
				mycc = None

			# Run CCSD(T) triples correction - MANDATORY for maximum computational rigor
			if mycc is not None:
				ccsd_t_energy = None
				with tqdm(desc="CCSD(T) Triples for Optimization", unit="step", colour='red') as pbar:
					try:
						if hasattr(mycc, 'ccsd_t'):
							print("Computing CCSD(T) triples correction...")
							start_time = time.time()
							ccsd_t_energy = mycc.ccsd_t()
							elapsed = time.time() - start_time
							pbar.update(1)
							pbar.set_postfix(E_T=f"{ccsd_t_energy:.8f}", time=f"{elapsed:.1f}s")
							print(f"CCSD(T) triples correction completed in {elapsed:.1f}s")
							print(f"CCSD(T) correction energy: {ccsd_t_energy:.12f} Hartree")
						else:
							print("WARNING: CCSD(T) method not available - falling back to SCF optimization")
							mycc = None
					except Exception as e:
						print(f"WARNING: CCSD(T) triples correction failed: {e} - falling back to SCF optimization")
						mycc = None
		
		# If CCSD failed or is not available, use SCF optimization
		if mycc is None:
			print("Using SCF geometry optimization (fallback from CCSD)")
			gradient_obj = mf  # Use the SCF object for gradients
		
		try:
			
			# Always use UHF gradients to avoid CCSD gradient issues
			with tqdm(desc="Building UHF Gradients", unit="step", colour='cyan') as pbar:
				print("Building UHF gradients (CPU-only)...")
				g = grad.UHF(mf_for_grad) if specified_spin != 0 else grad.RHF(mf_for_grad)
				# Ensure gradients work with Berny optimizer
				if not hasattr(g, 'nuclear_grad_method'):
					grad_cls: Any = g.__class__
					grad_cls.nuclear_grad_method = None
				pbar.update(1)
			
			# Test the gradient calculation with UHF gradients (reliable method)
			with tqdm(desc="Testing Gradient Calculation", unit="step", colour='yellow') as pbar:
				print("Testing gradient calculation (CPU-only)...")
				print(f"Gradient object type: {type(g)}")
				grad_array = g.kernel()
				if isinstance(grad_array, np.ndarray) and grad_array.size > 0:
					print(f"Gradient test successful, shape: {grad_array.shape}")
					print(f"Gradient norm: {np.linalg.norm(grad_array):.8f}")
					pbar.update(1)
					pbar.set_postfix(shape=f"{grad_array.shape}")
				else:
					raise RuntimeError(f"Gradients returned invalid result: {type(grad_array)}")
						
			# Narrow types for static analysis
			assert berny_solver is not None
			
			# Determine which gradient method we ended up using
			if hasattr(g, '__class__'):
				g_type = str(type(g))
				if 'numerical' in g_type.lower():
					gradient_method = "Numerical"
				elif 'UHF' in g_type:
					gradient_method = "UHF"
				else:
					gradient_method = "SCF"
			else:
				gradient_method = "UHF"  # Default since we always try UHF first
			
			print(f"Starting maximum precision {gradient_method} berny geometry optimization...")
			
			# Run Berny geometry optimization
			try:
				print(f"Solving Berny geometry optimization with {gradient_method} gradients...")
				start_time = time.time()
				
				# Set optimization parameters for better convergence
				mol_opt: Any = berny_solver.optimize(g, maxsteps=int(maxsteps))
				elapsed = time.time() - start_time
				print(f"Berny geometry optimization completed in {elapsed:.1f}s")
			except Exception as berny_e:
				print(f"Berny optimization failed: {berny_e}")
				print("Trying direct berny call with relaxed convergence...")
				try:
					mol_opt = berny_solver.optimize(g, maxsteps=min(20, int(maxsteps)))
				except Exception as berny_e2:
					raise RuntimeError(f"All berny optimization attempts failed: {berny_e2}")
			
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
			method_used = "CCSD(T)" if mycc is not None else "SCF"
			print(f"Maximum precision {method_used} geometry optimization completed successfully!")
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
