import warnings
import random

# Suppress pkg_resources deprecation warning from pyberny
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

from pyscf import gto, scf
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from tqdm import tqdm
from normalize_bonds import process_bond_displacements
from optimize_geometry import optimize_geometry_scf
from stabilization.scf_stabilization import robust_scf_calculation

# NOTE: For deterministic behaviour we lock all stochastic sources here.
# This ensures that repeated runs with identical inputs produce the
# same SCF-based dipole derivatives and hence the same ε values.
np.random.seed(12345)
random.seed(12345)

# Container for dipole derivative data obtained from polynomial fits.


@dataclass(slots=True)
class DipoleDerivativeResult:
	mu_prime: float
	mu_double_prime: float
	mu_triple_prime: float
	mu_quadruple_prime: float
	component_derivatives: dict[int, np.ndarray] = field(repr=False)
	displacement_grid: np.ndarray = field(repr=False)
	dipole_samples: np.ndarray = field(repr=False)
	polynomial_coefficients: list[np.ndarray] = field(repr=False)

# ===== MAXIMUM PRECISION DIPOLE DERIVATIVE SOLVING WITH SCF ONLY =======
#
# This module now implements a deterministic SCF-only approach:
# - Dipole moment computed at SCF level (since SCF excels at reproducibility)
# - Tight but practical SCF convergence used for all prerequisite calculations
# - No triples corrections or correlated densities are involved
# - High-quality basis sets are still supported via the user-specified basis
# - Deterministic settings (fixed seeds, single SCF path) for reproducibility
#
# This ensures maximum numerical stability and determinism for overtone
# spectroscopy applications where dipole derivative reproducibility is critical.

# We need to get the dipole moment vector of the optimized geometry
# Optimization is done before this dipole computation can begin in its separate module


def dipole_for_geometry(atom_string: str, spin: int, basis: str | None = None,
					   conv_tol: float = 1e-9, max_cycle: int = 300,
					   enable_stabilized_attempt: bool = True,
					   stabilized_direct: bool = True,
					   stabilized_diis_space: int = 16,
					   stabilized_max_cycle: int = 400,
					   stabilized_conv_tol: float = 1e-7,
					   stabilized_level_shift: float = 0.2) -> np.ndarray:
    
	"""Return the molecular dipole vector (Debye/Å) at SCF level only.

	To guarantee deterministic behaviour and avoid run-to-run variation
	from correlated-method convergence issues, this implementation *always*
	uses SCF densities for the dipole evaluation. Geometry optimization is also
	implemented with SCF-only gradients.

	Parameters
	----------
	atom_string : str
		Molecular geometry in XYZ format
	spin : int
		Spin multiplicity
	basis : str
		User-specified basis set (required)
	conv_tol : float
		SCF convergence tolerance (default: 1e-9)
	max_cycle : int
		Maximum SCF iterations for the initial configuration (default: 300)
	enable_stabilized_attempt : bool
		Whether to run a stabilized SCF retry when the direct solve fails
	stabilized_direct : bool
		If True attempt a direct SCF solve first before stabilization fallback
	stabilized_diis_space : int
		DIIS subspace dimension applied when stabilization is enabled
	stabilized_max_cycle : int
		Maximum SCF cycles used for the stabilized retry path
	stabilized_conv_tol : float
		Relaxed convergence tolerance for the stabilized retry
	stabilized_level_shift : float
		Initial level shift supplied to the stabilized retry

	Returns
	-------
	np.ndarray
		Dipole moment vector in Debye units
	"""

	if basis is None:
		raise ValueError("dipole_for_geometry requires a user-specified basis set")

	print(f"Computing SCF dipole for geometry with basis {basis}")
	
	# Build molecule for the initial direct attempt
	mol = gto.M(atom=atom_string, basis=basis, spin=spin, unit="Angstrom")

	def direct_scf_run() -> scf.hf.SCF:
		with tqdm(desc="SCF Convergence", unit="step", colour='blue') as pbar:
			mf_direct = scf.UHF(mol) if spin != 0 else scf.RHF(mol)
			mf_direct.conv_tol = conv_tol
			mf_direct.max_cycle = max_cycle
			if enable_stabilized_attempt and stabilized_diis_space is not None and hasattr(mf_direct, 'diis_space'):
				mf_direct.diis_space = stabilized_diis_space
			mf_direct.kernel()
			pbar.update(1)
			pbar.set_postfix(converged=getattr(mf_direct, "converged", False), energy=f"{getattr(mf_direct, 'e_tot', float('nan')):.6f}")
		return mf_direct

	mf: scf.hf.SCF

	if enable_stabilized_attempt and not stabilized_direct:
		print("Skipping direct SCF per configuration; running stabilized solver...")
		with tqdm(desc="Stabilized SCF Convergence", unit="pass", colour='cyan') as pbar:
			mf = robust_scf_calculation(
				atom_string=atom_string,
				spin=spin,
				basis=basis,
				target_conv_tol=conv_tol,
				max_cycle=stabilized_max_cycle,
				initial_conv_tol=stabilized_conv_tol,
				initial_level_shift=stabilized_level_shift,
				initial_diis_space=stabilized_diis_space,
			)
			pbar.update(1)
			pbar.set_postfix(converged=getattr(mf, "converged", False), energy=f"{getattr(mf, 'e_tot', float('nan')):.6f}")
	else:
		mf = _direct_scf_run()
		if enable_stabilized_attempt and not getattr(mf, "converged", False):
			print("Direct SCF did not converge; invoking stabilized fallback.")
			with tqdm(desc="Stabilized SCF Convergence", unit="pass", colour='cyan') as pbar:
				mf = robust_scf_calculation(
					atom_string=atom_string,
					spin=spin,
					basis=basis,
					target_conv_tol=conv_tol,
					max_cycle=stabilized_max_cycle,
					initial_conv_tol=stabilized_conv_tol,
					initial_level_shift=stabilized_level_shift,
					initial_diis_space=stabilized_diis_space,
				)
				pbar.update(1)
				pbar.set_postfix(converged=getattr(mf, "converged", False), energy=f"{getattr(mf, 'e_tot', float('nan')):.6f}")

	# For dipole evaluation we allow slightly underconverged SCF and use
	# the last available density matrix rather than aborting the workflow.
	if not getattr(mf, "converged", False):
		print("WARNING: SCF did not reach target conv_tol; using last iteration density for dipole.")
	else:
		print(f"High-precision SCF converged for dipole evaluation. Energy = {mf.e_tot:.12f} Hartree")

	# From this point onward we compute the dipole from the SCF
	# density matrix.

	try:
		dm1 = mf.make_rdm1()
		mol = mf.mol
		dip_ints = mol.intor('int1e_r', comp=3)
		if isinstance(dm1, tuple):
			dm1_total = dm1[0] + dm1[1]
		else:
			dm1_total = dm1
		if len(dm1_total.shape) == 3:
			dm1_total = np.sum(dm1_total, axis=0)
		dip_elec = -np.einsum('xij,ij->x', dip_ints, dm1_total)
		charges = mol.atom_charges()
		coords = mol.atom_coords()
		dip_nuc = np.einsum('i,ix->x', charges, coords)
		dipole_au = dip_elec + dip_nuc
		au_to_debye = 2.541746473
		dipole_debye = dipole_au * au_to_debye
		print(f"✅ SCF dipole moment: {np.linalg.norm(dipole_debye):.6f} Debye")
		print(f"SCF dipole components (Debye): [{dipole_debye[0]:.6f}, {dipole_debye[1]:.6f}, {dipole_debye[2]:.6f}]")
		return dipole_debye
	except Exception as scf_e:
		print(f"SCF dipole calculation failed: {scf_e}. Using nuclear dipole as final fallback...")
		mol = mf.mol
		charges = mol.atom_charges()
		coords = mol.atom_coords()
		dip_nuc = np.einsum('i,ix->x', charges, coords)
		au_to_debye = 2.541746473
		dipole_debye = dip_nuc * au_to_debye
		print(f"Nuclear dipole moment (final fallback): {np.linalg.norm(dipole_debye):.6f} Debye")
		return dipole_debye

# Finally we compute the actual µ dipole derivatives
def compute_µ_derivatives(coords_string: str, specified_spin: int, delta: float = 0.005, basis: str | None = None, atom_index: int = 0, axis: int = 2, bond_pair: tuple[int, int] | None = None, dual_bond_axes: str | None = None, m1: float | None = None, m2: float | None = None,
						enable_stabilized_attempt: bool = True,
						stabilized_direct: bool = True,
						stabilized_diis_space: int = 20,
						stabilized_max_cycle: int = 400,
						stabilized_conv_tol: float = 1e-7,
						stabilized_level_shift: float = 0.2,
						max_displacement: float | None = None,
						num_displacements: int = 7,
						poly_order: int = 5) -> DipoleDerivativeResult:
	"""
	Model the dipole surface µ(Q) with a higher-order polynomial and extract
	derivatives up to the fourth order directly from the fit.

	Parameters
	----------
	coords_string : str
		Molecular geometry (path or literal XYZ block).
	specified_spin : int
		Spin state for the SCF calculation.
	delta : float
		Legacy displacement magnitude (Å) retained for backwards compatibility. Used
		to define the default maximum displacement when the latter is not provided.
	basis : str | None
		User-specified basis set (required).
	atom_index : int
		Atom index for Cartesian displacements when no bond information is supplied.
	axis : int
		Cartesian axis (0, 1, or 2) for direct displacements.
	bond_pair : tuple[int, int] | None
		Optional pair of atom indices defining a bond stretch direction.
	dual_bond_axes : str | None
		Optional dual-bond specification "(n,x);(a,x)".
	m1, m2 : float | None
		User-supplied masses (amu) required for dual bond axes.
	enable_stabilized_attempt : bool
		Whether to retry SCF with the stabilization stack if the direct solve fails.
	stabilized_direct : bool
		Run the stabilized solver immediately instead of attempting a direct kernel.
	stabilized_diis_space : int
		DIIS space to impose on the stabilized SCF retry path.
	stabilized_max_cycle : int
		Maximum SCF cycles for the stabilized run.
	stabilized_conv_tol : float
		Relaxed convergence target for the stabilized pass.
	stabilized_level_shift : float
		Level shift for the stabilized retry.
	max_displacement : float | None
		Largest displacement magnitude (Å) to sample. Defaults to the greater of
		``0.04`` Å or ``delta * floor(num_displacements/2)``.
	num_displacements : int
		Total geometries to sample along the displacement coordinate (must be odd).
	poly_order : int
		Polynomial order used to fit each dipole component (>=4 to recover up to the
		fourth derivative).

	Returns
	-------
	DipoleDerivativeResult
		Aggregated first through fourth derivatives (Debye·Å⁻¹, Debye·Å⁻², ...)
		along with diagnostic data from the polynomial fit.
	"""
	if basis is None:
		raise ValueError("compute_µ_derivatives requires a user-specified basis set")

	if num_displacements < 3 or num_displacements % 2 == 0:
		raise ValueError("num_displacements must be an odd integer >= 3")
	if poly_order < 2:
		raise ValueError("poly_order must be at least 2")

	# read coords
	if os.path.isfile(coords_string):
		with open(coords_string, 'r') as fh:
			coord_text = fh.read()
	else:
		coord_text = coords_string

	# parse lines into numpy array
	lines = [ln.strip() for ln in coord_text.splitlines() if ln.strip()]
	atoms: list[str] = []
	positions_list: list[list[float]] = []
	for ln in lines:
		parts = ln.split()
		if len(parts) != 4:
			raise ValueError(f"Invalid coordinate line: '{ln}'. Expected 'Element x y z'.")
		atoms.append(parts[0])
		positions_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
	positions = np.array(positions_list, dtype=float)
	pos0 = positions.copy()

	# Process bond displacements and extract the unit direction vector (Å)
	_, _, block_from_positions, direction = process_bond_displacements(
		positions, atoms, dual_bond_axes, bond_pair, delta, m1, m2, atom_index, axis
	)

	if not np.any(direction):
		raise RuntimeError("Displacement direction is zero; check displacement inputs")

	if max_displacement is None:
		max_displacement = max(0.04, delta * (num_displacements // 2))

	print("Using SCF level for dipole moment calculations with polynomial surface fitting")
	print(f"SCF basis set: {basis}")
	print(f"Sampling ±{max_displacement:.4f} Å with {num_displacements} geometries")

	# Relaxed settings for dipole sampling SCFs
	conv_tol_sampling = 1e-4
	max_cycles_sampling = 120

	displacements = np.linspace(-max_displacement, max_displacement, num_displacements)
	dipole_samples: list[np.ndarray] = []

	with tqdm(total=num_displacements, desc="Dipole Surface SCF", unit="geom", colour='green') as pbar:
		for disp in displacements:
			geom = pos0 + direction * disp
			atom_block = block_from_positions(geom)
			dipole_vec = dipole_for_geometry(
				atom_block,
				specified_spin,
				basis=basis,
				conv_tol=conv_tol_sampling,
				max_cycle=max_cycles_sampling,
				enable_stabilized_attempt=enable_stabilized_attempt,
				stabilized_direct=stabilized_direct,
				stabilized_diis_space=stabilized_diis_space,
				stabilized_max_cycle=stabilized_max_cycle,
				stabilized_conv_tol=stabilized_conv_tol,
				stabilized_level_shift=stabilized_level_shift,
			)
			dipole_samples.append(dipole_vec)
			pbar.update(1)
			pbar.set_postfix(disp=f"{disp:+.4f} Å", norm=f"{np.linalg.norm(dipole_vec):.4f} D")

	dipole_samples_array = np.array(dipole_samples, dtype=float)

	# Fit each Cartesian component with an order >= poly_order polynomial
	max_possible_order = min(poly_order, num_displacements - 1)
	derivative_vectors: dict[int, np.ndarray] = {
		1: np.zeros(3, dtype=float),
		2: np.zeros(3, dtype=float),
		3: np.zeros(3, dtype=float),
		4: np.zeros(3, dtype=float),
	}
	polynomial_coefficients: list[np.ndarray] = []

	for comp in range(3):
		coeffs = np.polyfit(displacements, dipole_samples_array[:, comp], deg=max_possible_order)
		polynomial_coefficients.append(coeffs)
		for order in range(1, 5):
			if order <= len(coeffs) - 1:
				derivative_coeffs = np.polyder(coeffs, m=order)
				derivative_vectors[order][comp] = np.polyval(derivative_coeffs, 0.0)
			else:
				derivative_vectors[order][comp] = 0.0

	µ_prime_vec = derivative_vectors[1]
	µ_double_prime_vec = derivative_vectors[2]
	µ_triple_prime_vec = derivative_vectors[3]
	µ_quadruple_prime_vec = derivative_vectors[4]

	print(f"Debug: SCF first derivative vector (Debye/Å): {µ_prime_vec}")
	print(f"Debug: SCF second derivative vector (Debye/Å²): {µ_double_prime_vec}")
	print(f"Debug: SCF third derivative vector (Debye/Å³): {µ_triple_prime_vec}")
	print(f"Debug: SCF fourth derivative vector (Debye/Å⁴): {µ_quadruple_prime_vec}")

	µ_prime = float(np.linalg.norm(µ_prime_vec))
	µ_double_prime = float(np.linalg.norm(µ_double_prime_vec))
	µ_triple_prime = float(np.linalg.norm(µ_triple_prime_vec))
	µ_quadruple_prime = float(np.linalg.norm(µ_quadruple_prime_vec))

	print(f"Debug: |µ_prime(0)| = {µ_prime:.10e} Debye/Å")
	print(f"Debug: |µ_double_prime(0)| = {µ_double_prime:.10e} Debye/Å²")
	print(f"Debug: |µ_triple_prime(0)| = {µ_triple_prime:.10e} Debye/Å³")
	print(f"Debug: |µ_quadruple_prime(0)| = {µ_quadruple_prime:.10e} Debye/Å⁴")

	component_derivative_copy = {order: vec.copy() for order, vec in derivative_vectors.items()}

	return DipoleDerivativeResult(
		mu_prime=µ_prime,
		mu_double_prime=µ_double_prime,
		mu_triple_prime=µ_triple_prime,
		mu_quadruple_prime=µ_quadruple_prime,
		component_derivatives=component_derivative_copy,
		displacement_grid=displacements.copy(),
		dipole_samples=dipole_samples_array.copy(),
		polynomial_coefficients=[coeff.copy() for coeff in polynomial_coefficients],
	)

def compute_µ_derivatives_from_optimization(optimized_coords: np.ndarray, atoms: list[str], specified_spin: int, delta: float = 0.005, basis: str | None = None, atom_index: int = 0, axis: int = 2, bond_pair: tuple[int, int] | None = None, dual_bond_axes: str | None = None, m1: float | None = None, m2: float | None = None,
									   enable_stabilized_attempt: bool = True,
									   stabilized_direct: bool = True,
									   stabilized_diis_space: int = 20,
									   stabilized_max_cycle: int = 400,
									   stabilized_conv_tol: float = 1e-7,
									   stabilized_level_shift: float = 0.2,
									   max_displacement: float | None = None,
									   num_displacements: int = 7,
									   poly_order: int = 5) -> DipoleDerivativeResult:
	"""
	Compute dipole derivatives using optimized geometry from morse solver.
	
	optimized_coords: numpy array of optimized atomic positions from geometry optimization
	atoms: list of atomic symbols corresponding to coordinates
	basis: user-specified basis set propagated from the workflow
	... (other parameters same as compute_µ_derivatives)
	"""
	if basis is None:
		raise ValueError("compute_µ_derivatives_from_optimization requires a user-specified basis set")

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
		m2=m2,
		enable_stabilized_attempt=enable_stabilized_attempt,
		stabilized_direct=stabilized_direct,
		stabilized_diis_space=stabilized_diis_space,
		stabilized_max_cycle=stabilized_max_cycle,
		stabilized_conv_tol=stabilized_conv_tol,
		stabilized_level_shift=stabilized_level_shift,
		max_displacement=max_displacement,
		num_displacements=num_displacements,
		poly_order=poly_order,
	)

def full_pre_morse_dipole_workflow(initial_coords: str | np.ndarray, atoms: list[str], specified_spin: int, delta: float = 0.005, basis: str | None = None, atom_index: int = 0, axis: int = 2, bond_pair: tuple[int, int] | None = None, dual_bond_axes: str | None = None, m1: float | None = None, m2: float | None = None, optimize_geometry: bool = True,
					 enable_stabilized_attempt: bool = True,
					 stabilized_direct: bool = True,
					 stabilized_diis_space: int = 20,
					 stabilized_max_cycle: int = 400,
					 stabilized_conv_tol: float = 1e-7,
					 stabilized_level_shift: float = 0.2,
					 max_displacement: float | None = None,
					 num_displacements: int = 7,
					 poly_order: int = 5) -> tuple[DipoleDerivativeResult, np.ndarray]:
	"""
	Complete workflow: geometry optimization → dipole surface sampling → derivatives.

	Parameters mirror :func:`compute_µ_derivatives` with additional control over the
	initial optimization step. When ``optimize_geometry`` is False, the supplied
	coordinates are used directly for the dipole sampling grid.

	Returns
	-------
	DipoleDerivativeResult, np.ndarray
		Structured derivative summary together with the optimized (or fallback)
		coordinates used for sampling.
	"""
	if basis is None:
		raise ValueError("full_pre_morse_dipole_workflow requires a user-specified basis set")

	print("Starting full Morse dipole derivative workflow")
	
	if optimize_geometry:
		print("Step 1: SCF geometry optimization")
		
		# Convert initial coords to string format for optimize_geometry_scf
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
			optimized_coords_string = optimize_geometry_scf(
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

			print("✅ SCF geometry optimization completed successfully")

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
	
	print("Step 2: Computing SCF dipole derivatives from optimized geometry")
	derivatives = compute_µ_derivatives_from_optimization(
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
		m2=m2,
		enable_stabilized_attempt=enable_stabilized_attempt,
		stabilized_direct=stabilized_direct,
		stabilized_diis_space=stabilized_diis_space,
		stabilized_max_cycle=stabilized_max_cycle,
		stabilized_conv_tol=stabilized_conv_tol,
		stabilized_level_shift=stabilized_level_shift,
		max_displacement=max_displacement,
		num_displacements=num_displacements,
		poly_order=poly_order,
	)

	print("✅ Complete Morse dipole workflow finished successfully")
	return derivatives, optimized_coords
