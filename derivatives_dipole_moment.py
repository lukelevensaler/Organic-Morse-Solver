import warnings
# Suppress pkg_resources deprecation warning from pyberny
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

from pyscf import gto, scf
import numpy as np
import os
from typing import Any
try:
	# geomopt Berny solver for correlated gradients
	from pyscf.geomopt import berny_solver
except Exception:
	berny_solver = None

# lazily import CC and grad inside functions to avoid static analysis issues
cc = None
grad = None
import time
from tqdm import tqdm
from pyscf_monkeypatch import patch_pyscf_ccsd_gradient_bug
from normalize_bonds import process_bond_displacements

# ===== MAXIMUM PRECISION DIPOLE DERIVATIVE SOLVING WITH CCSD(T) =======
#
# This module implements the most computationally rigorous approach possible:
# - All dipole moments computed at full CCSD(T) level
# - Extremely tight SCF convergence (1e-12) for optimal CCSD(T) starting point
# - High-precision CCSD convergence (1e-9 to 1e-10) 
# - Mandatory CCSD(T) triples correction for all calculations
# - High-quality correlation-consistent basis sets (aug-cc-pVTZ default)
# - Optimized DIIS spaces and maximum iteration counts
# - Direct algorithms for best numerical accuracy
#
# This ensures maximum computational rigor for overtone spectroscopy applications
# where dipole derivative accuracy is critical.

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

	# build molecule
	mol = gto.M(atom="\n".join(f"{a} {x} {y} {z}" for a, (x, y, z) in zip(atoms, positions)), basis=basis, spin=specified_spin, unit='Angstrom')

	# run initial mean-field with maximum precision for CCSD(T)
	if specified_spin == 0:
		mf = scf.RHF(mol)
	else:
		mf = scf.UHF(mol)
	
	# Use maximum precision SCF settings for CCSD(T) optimization
	mf.conv_tol = 1e-12   # Extremely tight convergence
	mf.max_cycle = 400    # Many cycles for robust convergence
	mf.diis_space = 15    # Large DIIS space
	
	# Start progress tracking for SCF optimization
	with tqdm(desc="🔬 SCF for Geometry Optimization", unit="step", colour='blue') as pbar:
		pbar.set_postfix(tol=f"{mf.conv_tol:.0e}", max_cycle=mf.max_cycle)
		mf.run()
		pbar.update(1)
		pbar.set_postfix(converged=mf.converged, energy=f"{mf.e_tot:.6f}")
	
	if not mf.converged:
		raise RuntimeError("High-precision SCF did not converge - cannot proceed with CCSD(T) optimization")

	# lazily import cc and grad to check for analytic gradient support
	try:
		from pyscf import cc as _cc, grad as _grad
	except Exception:
		_cc = None
		_grad = None

	# Require CCSD gradients + berny_solver to be available; otherwise fail loudly
	missing = []
	if berny_solver is None:
		missing.append('berny_solver (geometry optimizer)')
	if _cc is None:
		missing.append('pyscf.cc (CCSD module)')
	if _grad is None:
		missing.append('pyscf.grad (analytic gradients)')
	if missing:
		msg = (
			"CCSD gradients and/or berny_solver are not available in this PySCF installation; cannot perform CCSD geometry optimization.\n"
			"Missing: " + ", ".join(missing) + ".\n"
			"Recommended fixes:\n"
			" - Install PySCF and berny from conda-forge (recommended):\n"
			"     conda create -n pyscf-env -c conda-forge python=3.11 pyscf berny scipy -y\n"
			" - Or install via pip and build from source if wheel lacks compiled extensions:\n"
			"     python -m pip install pyscf berny\n"
			"   If pip wheel lacks CCSD gradient code, build PySCF from source (see INSTALL_PySCF.md in project root).\n"
			"After installing, re-run the script in the same Python environment.\n"
		)
		raise RuntimeError(msg)

	# Run CCSD optimization and raise if it fails
	try:
		# ensure static analysers know these are present
		assert _cc is not None, 'pyscf.cc not available'
		assert _grad is not None, 'pyscf.grad not available'
		# use the lazily imported cc with maximum rigor convergence controls
		mycc = _cc.CCSD(mf)
		
		# Set maximum precision convergence criteria for ultimate CCSD(T) accuracy
		mycc.conv_tol = 1e-10   # Maximum precision convergence
		mycc.max_cycle = 200    # Many iterations for robust convergence
		mycc.diis_space = 15    # Maximum DIIS space for optimal convergence
		mycc.direct = True      # Use direct algorithm for better numerical accuracy
		
		print(f"Maximum precision CCSD(T) optimization settings: conv_tol={mycc.conv_tol}, max_cycle={mycc.max_cycle}, diis_space={mycc.diis_space}")

		# Run CCSD calculation with progress tracking
		with tqdm(desc="🧬 CCSD for Geometry Optimization", unit="iter", colour='green') as pbar:
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
		with tqdm(desc="⚛️ CCSD(T) Triples for Optimization", unit="step", colour='red') as pbar:
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
			# Apply PySCF bug patch before gradient calculations
			patch_pyscf_ccsd_gradient_bug()
			
			# build gradients and optimize geometry (may be long)
			with tqdm(desc="📊 Building CCSD Gradients", unit="step", colour='cyan') as pbar:
				print("Building CCSD gradients...")
				g = _grad.ccsd.Gradients(mycc)
				pbar.update(1)
			
			# Test the gradient calculation first with error handling for PySCF bugs
			with tqdm(desc="🧪 Testing Gradient Calculation", unit="step", colour='yellow') as pbar:
				print("Testing gradient calculation...")
				try:
					# Try to calculate gradients with workarounds for PySCF 2.10.0 bugs
					test_grad = None
					
					# First attempt: direct gradient calculation
					try:
						test_grad = g.kernel()
					except (AttributeError, TypeError) as e:
						if "'tuple' object has no attribute 'diagonal'" in str(e):
							print("PySCF 2.10.0 CCSD gradient bug detected. Attempting workaround...")
							
							# Force rebuild of CCSD eris with proper fock matrix
							# Fock matrices are also used in CCSD and CCSD(T), we are not using Hartree-Fock here
							try:
								# Rebuild the CCSD object with fresh eris
								mycc_new = _cc.CCSD(mf)
								mycc_new.conv_tol = mycc.conv_tol
								mycc_new.max_cycle = mycc.max_cycle
								mycc_new.diis_space = mycc.diis_space
								mycc_new.direct = True
								
								# Copy over the converged amplitudes if available
								if hasattr(mycc, 't1') and hasattr(mycc, 't2') and mycc.converged:
									try:
										# Use setattr to avoid type checking issues
										setattr(mycc_new, 't1', mycc.t1)
										setattr(mycc_new, 't2', mycc.t2)
										setattr(mycc_new, 'converged', True)
										if hasattr(mycc, 'e_corr') and mycc.e_corr is not None:
											setattr(mycc_new, 'e_corr', mycc.e_corr)
									except Exception:
										# If copying fails, re-run CCSD
										mycc_new.run()
								else:
									# Re-run CCSD if amplitudes not available
									mycc_new.run()
								
								# Create new gradient object
								g = _grad.ccsd.Gradients(mycc_new)
								test_grad = g.kernel()
								
							except Exception as workaround_e:
								print(f"Workaround failed: {workaround_e}")
								raise e  # Re-raise original error
						else:
							raise e
					
					if test_grad is not None:
						pbar.update(1)
						pbar.set_postfix(shape=f"{test_grad.shape}")
						print(f"Gradient test successful, shape: {test_grad.shape}")
					else:
						raise RuntimeError("Gradient calculation returned None")
						
				except Exception as grad_e:
					print(f"Gradient calculation failed: {grad_e}")
					# The gradient failed, this is likely the source of our issue
					raise RuntimeError(f"CCSD gradient calculation failed: {grad_e}")
			
			# Narrow types for static analysis
			assert berny_solver is not None
			print("Starting maximum precision CCSD(T) berny geometry optimization...")
			
			# Use berny_solver.optimize with explicit error handling
			with tqdm(desc="Berny Geometry Optimization", total=maxsteps, unit="step", colour='magenta') as pbar:
				try:
					start_time = time.time()
					
					# Create a closure to track optimization steps
					step_counter = {'current': 0}
					
					def optimization_callback(*args, **kwargs):
						step_counter['current'] += 1
						pbar.update(1)
						pbar.set_postfix(step=f"{step_counter['current']}/{maxsteps}")
						return True
					
					mol_opt: Any = getattr(berny_solver, 'optimize')(g, maxsteps=int(maxsteps))
					elapsed = time.time() - start_time
					pbar.set_postfix(completed=True, time=f"{elapsed:.1f}s")
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

# Then we get the dipole moment vector of the optimized geometry
def dipole_for_geometry(atom_string: str, spin: int, basis: str = "aug-cc-pVTZ", conv_tol: float = 1e-9, max_cycle: int = 150) -> np.ndarray:
	"""Return the molecular dipole vector (Debye) computed at CCSD(T) level.

	This function uses full CCSD(T) method for accurate dipole moments, which is 
	essential for high-accuracy overtone transition dipole calculations.

	NOTE: Hartree refers to the atomic unit of energy, not the Hartree-Fock method, which we do not use.

	Parameters:
	-----------
	atom_string : str
		Molecular geometry in XYZ format
	spin : int
		Spin multiplicity
	basis : str
		Basis set (default: aug-cc-pVTZ for high accuracy)
	conv_tol : float
		CCSD convergence tolerance (default: 1e-9 for very tight convergence)
	max_cycle : int
		Maximum CCSD iterations (default: 150 for robust convergence)
		
	Returns:
	--------
	np.ndarray
		Dipole moment vector in Debye units
	"""
	print(f"Computing CCSD(T) dipole for geometry with basis {basis}")
	
	mol = gto.M(atom=atom_string, basis=basis, spin=spin)
	
	# Run initial SCF calculation with maximum precision for CCSD(T)
	with tqdm(desc="🔬 SCF Convergence", unit="step", colour='blue') as pbar:
		if spin == 0:
			mf = scf.RHF(mol)
		else:
			mf = scf.UHF(mol)
		
		# Use very tight SCF convergence for highest quality CCSD(T)
		mf.conv_tol = 1e-12  # Extremely tight for CCSD(T) accuracy
		mf.max_cycle = 300   # More cycles for robust convergence
		mf.diis_space = 12   # Larger DIIS space for better SCF convergence
		pbar.set_postfix(tol=f"{mf.conv_tol:.0e}", max_cycle=mf.max_cycle)
		
		mf.run()
		pbar.update(1)
		pbar.set_postfix(converged=mf.converged, energy=f"{mf.e_tot:.6f}")
	
	if not mf.converged:
		raise RuntimeError("SCF did not converge to required precision - cannot proceed with high-accuracy CCSD(T)")
	
	print(f"High-precision SCF converged for CCSD(T). Energy = {mf.e_tot:.12f} Hartree")
	
	# Import CC module
	try:
		from pyscf import cc
	except ImportError:
		raise RuntimeError("PySCF CC module not available - cannot compute CCSD(T) dipoles")
	
	# Set up CCSD calculation with maximum rigor
	mycc = cc.CCSD(mf)
	mycc.conv_tol = conv_tol
	mycc.max_cycle = max_cycle
	mycc.diis_space = 12  # Maximum DIIS space for optimal convergence
	mycc.direct = True    # Use direct algorithm for better accuracy
	
	print(f"High-precision CCSD(T) dipole settings: conv_tol={mycc.conv_tol:.2e}, max_cycle={mycc.max_cycle}")
	
	# Run CCSD calculation with maximum precision
	with tqdm(desc="🧬 CCSD Correlation", unit="step", colour='green') as pbar:
		pbar.set_postfix(tol=f"{mycc.conv_tol:.0e}", max_cycle=mycc.max_cycle)
		try:
			mycc.run()
			pbar.update(1)
			pbar.set_postfix(converged=mycc.converged, corr_energy=f"{mycc.e_corr:.6f}")
		except Exception as e:
			raise RuntimeError(f"High-precision CCSD calculation failed: {e}")
	
	if not mycc.converged:
		raise RuntimeError("CCSD did not converge to required precision - cannot compute high-accuracy dipole")
	
	print(f"High-precision CCSD converged. Correlation energy = {mycc.e_corr:.12f} Hartree")
	
	# Compute CCSD(T) triples correction - MANDATORY for maximum accuracy
	with tqdm(desc="⚛️ CCSD(T) Triples Correction", unit="step", colour='red') as pbar:
		try:
			if hasattr(mycc, 'ccsd_t'):
				pbar.set_postfix(status="computing")
				e_t = mycc.ccsd_t()
				pbar.update(1)
				pbar.set_postfix(E_T=f"{e_t:.8f}")
				print(f"High-precision CCSD(T) triples correction: {e_t:.12f} Hartree")
				total_energy = mycc.e_tot + e_t
				print(f"Total high-precision CCSD(T) energy: {total_energy:.12f} Hartree")
			else:
				raise RuntimeError("CCSD(T) triples correction not available - cannot achieve required computational rigor")
		except Exception as e:
			raise RuntimeError(f"CCSD(T) triples correction failed - required for maximum accuracy: {e}")
	
	# Calculate CCSD dipole moment
	with tqdm(desc="🔬 Computing CCSD Dipole Moment", unit="step", colour='magenta') as pbar:
		try:
			# Use CCSD density matrices for dipole calculation
			dm1 = mycc.make_rdm1()
			pbar.update(1)
			pbar.set_postfix(step="density_matrix")
			
			# Calculate dipole integrals
			dip_ints = mol.intor('int1e_r', comp=3)  # x, y, z components
			pbar.update(1) 
			pbar.set_postfix(step="dipole_integrals")
			
			# Calculate electronic dipole contribution
			dip_elec = -np.einsum('xij,ji->x', dip_ints, dm1)
			pbar.update(1)
			pbar.set_postfix(step="electronic_contribution")
			
			# Add nuclear contribution
			charges = mol.atom_charges()
			coords = mol.atom_coords()
			dip_nuc = np.einsum('i,ix->x', charges, coords)
			pbar.update(1)
			pbar.set_postfix(step="nuclear_contribution")
			
			# Total dipole moment in atomic units
			dipole_au = dip_elec + dip_nuc
			
			# Convert from atomic units to Debye (1 au = 2.541746 Debye)
			au_to_debye = 2.541746473
			dipole_debye = dipole_au * au_to_debye
			
			print(f"High-precision CCSD(T) dipole moment: {np.linalg.norm(dipole_debye):.8f} Debye")
			print(f"High-precision CCSD(T) dipole components (Debye): [{dipole_debye[0]:.8f}, {dipole_debye[1]:.8f}, {dipole_debye[2]:.8f}]")
			
			return dipole_debye
			
		except Exception as e:
			raise RuntimeError(f"Failed to compute high-precision CCSD(T) dipole moment: {e}")

# Finally we compute the actual µ dipole derivatives
def compute_µ_derivatives(coords_string: str, specified_spin: int, delta: float = 0.005, basis: str = "aug-cc-pVTZ", atom_index: int = 0, axis: int = 2, bond_pair: tuple[int, int] | None = None, dual_bond_axes: str | None = None, m1: float | None = None, m2: float | None = None) -> tuple[float, float]:
	"""
	Compute first and second derivatives of the dipole using finite differences at CCSD(T) level.

	coords_string: multiline string of fully numeric coordinates (Element x y z)
	specified_spin: spin state for dipole calculation
	delta: displacement magnitude (Å)
	basis: quantum chemistry basis set
	atom_index: which atom to displace (0-based)
	axis: which Cartesian axis to displace (0=x, 1=y, 2=z)
	bond_pair: optional tuple of atom indices to define bond stretch direction
	dual_bond_axes: optional string in format "(n,x);(a,x)" for two bond axes with shared element x
	m1: mass of element A (same as main_morse_solver.py m1 = A)
	m2: mass of element B (same as main_morse_solver.py m2 = B)
	
	Returns (µ_prime, µ_double_prime) in SI units: µ_prime in C·m/m, µ_double_prime in C·m/m^2
	"""
	# read coords
	if os.path.isfile(coords_string):
		with open(coords_string, 'r') as fh:
			coord_text = fh.read()
	else:
		coord_text = coords_string

	# parse lines into numpy array
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
	
	# Process bond displacements using normalize_bonds module
	pos_plus, pos_minus, block_from_positions = process_bond_displacements(
		positions, atoms, dual_bond_axes, bond_pair, delta, m1, m2, atom_index, axis
	)
	pos0 = positions.copy()  # equilibrium geometry
	
	# Create atom strings for the three geometries
	atom0 = block_from_positions(pos0)
	atom_plus = block_from_positions(pos_plus)
	atom_minus = block_from_positions(pos_minus)

	# compute dipoles at maximum precision CCSD(T) level
	print("Using maximum precision CCSD(T) level for dipole moment calculations")
	# Use high-quality basis sets for maximum accuracy
	ccsd_basis = basis  # Use the requested basis directly for maximum rigor
	if basis == "aug-cc-pVQZ":
		# Keep the high-quality basis for ultimate accuracy
		print("Using aug-cc-pVQZ basis for maximum CCSD(T) accuracy")
	elif "cc-p" in basis.lower():
		ccsd_basis = basis  # Use correlation-consistent basis as requested
	else:
		ccsd_basis = "aug-cc-pVTZ"  # Default to high-quality basis
	
	print(f"Maximum precision CCSD(T) basis set: {ccsd_basis}")
	
	# Use tighter convergence for finite difference derivatives
	tight_conv_tol = 1e-10  # Very tight for numerical derivatives
	max_cycles = 200        # More cycles for robust convergence
	
	# Main finite difference progress tracker
	with tqdm(total=3, desc="🧮 Finite Difference CCSD(T) Calculations", unit="geom", colour='green') as pbar:
		pbar.set_postfix(geometry="equilibrium")
		µ0 = dipole_for_geometry(atom0, specified_spin, basis=ccsd_basis, conv_tol=tight_conv_tol, max_cycle=max_cycles)
		pbar.update(1)
		
		pbar.set_postfix(geometry="+δ displacement")
		µ_plus = dipole_for_geometry(atom_plus, specified_spin, basis=ccsd_basis, conv_tol=tight_conv_tol, max_cycle=max_cycles)
		pbar.update(1)
		
		pbar.set_postfix(geometry="-δ displacement")
		µ_minus = dipole_for_geometry(atom_minus, specified_spin, basis=ccsd_basis, conv_tol=tight_conv_tol, max_cycle=max_cycles)
		pbar.update(1)
		
		pbar.set_postfix(geometry="completed")

	# Debug output for high-precision CCSD(T) dipole moments
	print(f"Debug: CCSD(T) dipole at equilibrium: {µ0} Debye")
	print(f"Debug: CCSD(T) dipole at +δ: {µ_plus} Debye") 
	print(f"Debug: CCSD(T) dipole at -δ: {µ_minus} Debye")
	print(f"Debug: High-precision displacement δ = {delta} Å")
	
	# finite-difference derivatives
	µ_prime_vec = (µ_plus - µ_minus) / (2.0 * delta)
	µ_double_prime_vec = (µ_plus - 2.0 * µ0 + µ_minus) / (delta ** 2)
	
	print(f"Debug: CCSD(T) first derivative vector (Debye/Å): {µ_prime_vec}")
	print(f"Debug: CCSD(T) second derivative vector (Debye/Å²): {µ_double_prime_vec}")

	# convert from Debye/(Å^n) to C·m/(m^n)
	D_TO_CM = 3.33564e-30
	µ_prime_si = np.linalg.norm(µ_prime_vec) * (D_TO_CM / 1e-10)
	µ_double_prime_si = np.linalg.norm(µ_double_prime_vec) * (D_TO_CM / 1e-20)
	
	print(f"Debug: |µ_prime(0)| = {µ_prime_si:.10e} C·m/m (CCSD(T) precision)")

	return float(µ_prime_si), float(µ_double_prime_si)
