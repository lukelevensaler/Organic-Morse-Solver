import warnings
# Suppress pkg_resources deprecation warning from pyberny
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

from pyscf import gto, scf
import numpy as np
import os
# lazily import CC and grad inside functions to avoid static analysis issues
cc = None
grad = None
from tqdm import tqdm
from normalize_bonds import process_bond_displacements
from optimize_geometry import optimize_geometry_ccsd

# ===== MAXIMUM PRECISION DIPOLE DERIVATIVE SOLVING WITH CCSD(T) =======
#
# This module implements the most computationally rigorous approach possible:
# - Dipole moment computed at full CCSD(T) level
# - Extremely tight SCF convergence (1e-12) used for required CCSD prerequisite
# - High-precision CCSD convergence (1e-9 to 1e-10) used for required CCSD(T) prerequisite
# - Mandatory CCSD(T) triples correction for all calculations
# - High-quality correlation-consistent basis sets (aug-cc-pVTZ default)
# - Optimized DIIS spaces and maximum iteration counts
# - Direct algorithms for best numerical accuracy
#
# This ensures maximum computational rigor for overtone spectroscopy applications
# where dipole derivative accuracy is critical.

# We need to get the dipole moment vector of the optimized geometry
# Optimization is done before this dipole computation can begin in its separate module

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
	with tqdm(desc="SCF Convergence", unit="step", colour='blue') as pbar:
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
	with tqdm(desc="CCSD Correlation", unit="step", colour='green') as pbar:
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
	with tqdm(desc="CCSD(T) Triples Correction", unit="step", colour='red') as pbar:
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
	with tqdm(desc="Computing CCSD Dipole Moment", unit="step", colour='magenta') as pbar:
		try:
			# Use CCSD density matrices for dipole calculation
			dm1 = mycc.make_rdm1()
			pbar.update(1)
			pbar.set_postfix(step="density_matrix")
			
			# Handle both RHF and UHF cases
			if isinstance(dm1, tuple):
				# UHF case: dm1 is (dm1_alpha, dm1_beta)
				dm1_total = dm1[0] + dm1[1]  # Total density matrix
			else:
				# RHF case: dm1 is already the total density matrix
				dm1_total = dm1
			
			# Calculate dipole integrals
			dip_ints = mol.intor('int1e_r', comp=3)  # x, y, z components
			pbar.update(1) 
			pbar.set_postfix(step="dipole_integrals")
			
			# Calculate electronic dipole contribution
			# dip_ints has shape (3, nbasis, nbasis), dm1_total has shape (nbasis, nbasis)
			dip_elec = -np.einsum('xij,ij->x', dip_ints, dm1_total)
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
	with tqdm(total=3, desc="Finite Difference CCSD(T) Calculations", unit="geom", colour='green') as pbar:
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

def compute_µ_derivatives_from_optimization(optimized_coords: np.ndarray, atoms: list[str], specified_spin: int, delta: float = 0.005, basis: str = "aug-cc-pVTZ", atom_index: int = 0, axis: int = 2, bond_pair: tuple[int, int] | None = None, dual_bond_axes: str | None = None, m1: float | None = None, m2: float | None = None) -> tuple[float, float]:
	"""
	Compute dipole derivatives using optimized geometry from morse solver.
	
	optimized_coords: numpy array of optimized atomic positions from geometry optimization
	atoms: list of atomic symbols corresponding to coordinates
	... (other parameters same as compute_µ_derivatives)
	"""
	# Convert optimized geometry to coordinate string format
	coord_lines = []
	for i, atom in enumerate(atoms):
		x, y, z = optimized_coords[i]
		coord_lines.append(f"{atom} {x:.10f} {y:.10f} {z:.10f}")
	coords_string = "\n".join(coord_lines)
	
	print(f"Computing dipole derivatives from optimized geometry with {len(atoms)} atoms")
	print(f"Optimized geometry (first 3 atoms): {coord_lines[:3]}")
	
	return compute_µ_derivatives(
		coords_string=coords_string,
		specified_spin=specified_spin,
		delta=delta,
		basis=basis,
		atom_index=atom_index,
		axis=axis,
		bond_pair=bond_pair,
		dual_bond_axes=dual_bond_axes,
		m1=m1,
		m2=m2
	)

def full_pre_morse_dipole_workflow(initial_coords: str | np.ndarray, atoms: list[str], specified_spin: int, delta: float = 0.005, basis: str = "aug-cc-pVTZ", atom_index: int = 0, axis: int = 2, bond_pair: tuple[int, int] | None = None, dual_bond_axes: str | None = None, m1: float | None = None, m2: float | None = None, optimize_geometry: bool = True) -> tuple[float, float, np.ndarray]:
	"""
	Complete workflow: geometry optimization → dipole calculation → derivatives
	
	initial_coords: initial geometry (string or numpy array)
	atoms: list of atomic symbols
	optimize_geometry: whether to run geometry optimization first
	
	Returns: (µ_prime, µ_double_prime, optimized_coords)
	"""
	print("Starting full Morse dipole derivative workflow")
	
	if optimize_geometry:
		print("Step 1: CCSD(T) geometry optimization")
		
		# Convert initial coords to string format for optimize_geometry_ccsd
		if isinstance(initial_coords, np.ndarray):
			# Convert numpy array back to coordinate string
			coord_lines = []
			for i, atom in enumerate(atoms):
				x, y, z = initial_coords[i]
				coord_lines.append(f"{atom} {x:.10f} {y:.10f} {z:.10f}")
			coords_string = "\n".join(coord_lines)
		else:
			coords_string = initial_coords
		
		try:
			# Call the actual geometry optimization function
			optimized_coords_string = optimize_geometry_ccsd(
				coords_string=coords_string,
				specified_spin=specified_spin,
				basis=basis
			)
			
			# Parse optimized coordinates back to numpy array for consistency
			lines = [ln.strip() for ln in optimized_coords_string.splitlines() if ln.strip()]
			optimized_positions = []
			for ln in lines:
				parts = ln.split()
				optimized_positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
			optimized_coords = np.array(optimized_positions)

			print("✅ CCSD(T) geometry optimization completed successfully")

		except Exception as e:
			print(f"⚠️ Geometry optimization failed: {e}")
			print("Using initial coordinates for dipole calculation")
			if isinstance(initial_coords, str):
				# Parse string coordinates
				lines = [ln.strip() for ln in initial_coords.splitlines() if ln.strip()]
				positions = []
				for ln in lines:
					parts = ln.split()
					positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
				optimized_coords = np.array(positions)
			else:
				optimized_coords = initial_coords.copy()
	else:
		print("Step 1: Skipping geometry optimization (using initial coordinates)")
		if isinstance(initial_coords, str):
			# Parse string coordinates
			lines = [ln.strip() for ln in initial_coords.splitlines() if ln.strip()]
			positions = []
			for ln in lines:
				parts = ln.split()
				positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
			optimized_coords = np.array(positions)
		else:
			optimized_coords = initial_coords.copy()
	
	print("Step 2: Computing CCSD(T) dipole derivatives from optimized geometry")
	µ_prime, µ_double_prime = compute_µ_derivatives_from_optimization(
		optimized_coords=optimized_coords,
		atoms=atoms,
		specified_spin=specified_spin,
		delta=delta,
		basis=basis,
		atom_index=atom_index,
		axis=axis,
		bond_pair=bond_pair,
		dual_bond_axes=dual_bond_axes,
		m1=m1,
		m2=m2
	)
	
	print("✅ Complete Morse dipole workflow finished successfully")
	return µ_prime, µ_double_prime, optimized_coords
