from typing import Optional
import typer
# Use package-relative imports so the module can be imported as `morse_solver`
# (avoids ImportError when importing via the package namespace).
from .derivatives_dipole_moment import compute_µ_derivatives, optimize_geometry_ccsd
# Import callable helpers but avoid importing runtime globals `a` and `λ` at
# import-time (they are created by `setup_globals`). Import the module as
# `mm` and reference `mm.a` / `mm.λ` after `setup_globals(...)` has run.
from .main_morse_solver import setup_globals, M_0n, integrated_molar_absorptivity, epsilon_peak_from_integrated
from . import main_morse_solver as mm

# Expose a module-level Typer app so external runners (run_morse_model.py) can import it
app = typer.Typer(help="Morse overtone CLI for estimating ε_max from transition dipoles")
 
# helper list
allowed_stretches_array = [
    "C–H",
    "C=O",
    "C–N",
    "N–H",
    "O–H",
]

@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    """Welcome message printed when running the CLI without subcommands.

    Suggests running `stretches` to see allowed organic stretches.
    """
    if ctx.invoked_subcommand is None:
        typer.echo("This CLI performs the following calulations to determine the molar extinction coefficent ε_(max) in M·cm^-1 for any organic molecule's IR (or NIR) overtone.")
        typer.echo("Run 'python morse_solver.py stretches' to see the allowed organic stretches.")

@app.command("stretches", help="Print allowed organic stretches")
def stretches():
    typer.echo("Allowed organic stretches:")
    for s in allowed_stretches_array:
        typer.echo(f" - {s}")


@app.command("compute", help="Compute ε_max from inputs")
def compute(
	amu_a: Optional[float] = typer.Argument(None, help="Molar mass of A (amu). If omitted you will be prompted"), 
	amu_b: Optional[float] = typer.Argument(None, help="Molar mass of B (amu). If omitted you will be prompted"), 
	fundamental_frequency: Optional[float] = typer.Argument(None, help="Fundamental vibrational frequency (cm^-1). If omitted you will be prompted"), 
	observed_frequency: Optional[float] = typer.Argument(None, help="Observed vibrational frequency (cm^-1). If omitted you will be prompted"), 
	overtone_order: Optional[int] = typer.Argument(None, help="Overtone order (n=1 for first overtone). If omitted you will be prompted"), 
	coords: Optional[str] = typer.Option(None, help="Cartesian coordinates (x,y,z) in Å for each atom's position in the chosen molecule's geometry. You must get such data from the CCDC Database (https://www.ccdc.cam.ac.uk/structures/), since that data is dierctly optimized to CCSD(T)-level quality."),
	specified_spin: Optional[int] = typer.Option(None, help="Spin multiplicity. Required if coords provided."),
	delta: Optional[float] = typer.Option(0.01, help="Finite-difference displacement magnitude in Å."),
	bond: Optional[str] = typer.Option(None, help="Bond indices: 'n,x' for single bond or '(n,x);(a,x)' for dual bond axes with mass weighting."),
	fwhm: Optional[float] = typer.Option(None, help="Assumed FWHM of the overtone band in cm^-1"),
	scaling_factor: Optional[int] = typer.Option(None, "--scaling-factor", help="Scaling exponent for ε_max (e.g., 1 for 1e1, 64 for 1e64). If omitted, you will be prompted interactively."),
	basis_set: str = typer.Option("aug-cc-pVTZ", "--basis", help="Basis set for quantum calculations (default: aug-cc-pVTZ)")) -> None:
	"""Interactive or positional compute.

	If any of the positional arguments (amu_a, amu_b, fundamental_frequency,
	observed_frequency, overtone_order) are omitted the CLI will prompt for them
	in sequence. Masses are entered in amu and converted to kg automatically.
	"""

	# prompt for missing required values
	if amu_a is None:
		amu_a = typer.prompt("Molar mass of element A (amu)", type=float)
	if amu_b is None:
		amu_b = typer.prompt("Molar mass of element B (amu)", type=float)
	if fundamental_frequency is None:
		fundamental_frequency = typer.prompt("Fundamental vibrational frequency (cm^-1)", type=float)
	if observed_frequency is None:
		observed_frequency = typer.prompt("Observed overtone frequency (cm^-1)", type=float)
	if overtone_order is None:
		overtone_order = typer.prompt("Overtone order (integer, e.g. 1)", type=int)


	# convert masses from amu -> kg
	AMU_TO_KG = 1.66053906660e-27
	assert amu_a is not None and amu_b is not None
	m_a_kg = float(amu_a) * AMU_TO_KG
	m_b_kg = float(amu_b) * AMU_TO_KG

	# prepare bond_pair default
	bond_pair = None

	# If coordinates are provided compute µ_prime and µ_double_prime via PySCF finite-difference
	if coords is not None:
		if specified_spin is None:
			specified_spin = typer.prompt("Spin value", type=int)
		# ensure specified_spin is an int (assert for the type checker)
		assert specified_spin is not None
		specified_spin_int = int(specified_spin)
		# parse bond indices if provided (overrides interactive bond)
		dual_bond_axes = None
		if bond is not None:
			try:
				# Check if this is dual bond axes format: "(n,x);(a,x)"
				if ';' in bond and '(' in bond and ')' in bond:
					# Dual bond axes format
					dual_bond_axes = bond  # Pass the string directly to compute_µ_derivatives
					bond_pair = None  # Don't use single bond_pair for dual axes
				else:
					# Single bond pair format: "n,x"
					parts = [int(p.strip()) for p in bond.split(',')]
					if len(parts) != 2:
						raise ValueError("bond must be two comma-separated 1-based indices or dual bond format '(n,x);(a,x)'")
					bond_pair = (parts[0] - 1, parts[1] - 1)  # convert to 0-based (i, j)
			except Exception as e:
				raise typer.BadParameter(f"Invalid bond indices: {e}")
		# ensure delta is a float
		delta_val = float(delta) if delta is not None else 0.01
		# Show theory level information
		typer.secho("Beginning CCSD(T) for ab initio molecular geometry optimization and dipole derivative calculation.", fg="green", bold=True)
		
		# Use CCSD optimization
		try:
			coords_to_use = optimize_geometry_ccsd(coords, specified_spin_int, basis=basis_set)
		except Exception as e:
			typer.secho(f"CCSD optimization failed: {e}", fg="red", err=True)
			typer.secho("Try using --basis=STO-3G for a smaller basis set", fg="yellow")
			raise typer.Exit(code=2)
		try:
			# Pass the masses m1 and m2 that were converted from amu_a and amu_b
			# Note: compute_µ_derivatives expects masses in amu, so use original amu_a, amu_b
			µ_prime_val, µ_double_prime_val = compute_µ_derivatives(
				coords_to_use, 
				specified_spin_int, 
				delta=delta_val, 
				bond_pair=bond_pair, 
				dual_bond_axes=dual_bond_axes,
				m1=amu_a,  # Pass original amu values, not kg
				m2=amu_b,
				basis=basis_set
			)
		except Exception as e:
			typer.secho(f"Error computing dipole derivatives from coordinates: {e}", fg="red", err=True)
			raise typer.Exit(code=1)
		µ_prime = float(µ_prime_val)
		µ_double_prime = float(µ_double_prime_val)
	else:
		# If coords were not provided, require interactive coordinate entry
		atom_input = typer.prompt("Enter atoms as a comma-separated list (e.g. N,H,H)", type=str)
		atoms = [a.strip() for a in atom_input.split(',') if a.strip()]
		coords_entries = []
		for i, atom in enumerate(atoms, start=1):
			# loop until valid triple provided for each atom
			while True:
				coord_str = typer.prompt(f"Enter x,y,z for atom {i} ({atom}) as comma-separated values in Å", type=str)
				parts = [p.strip() for p in coord_str.split(',')]
				if len(parts) != 3:
					typer.echo("Please provide three comma-separated numbers, e.g. 0.0,0.0,1.0")
					continue
				try:
					x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
					coords_entries.append((atom, x, y, z))
					break
				except ValueError:
					typer.echo("Coordinates must be numbers; please try again.")
					continue

		# Ask which bond to displace: require bond pair 'n,m' (1-based)
		while True:
			disp_input = typer.prompt(f"Enter bond atom indices as 'n,m' (1-{len(atoms)}), e.g. '1,2'", type=str)
			parts = [p.strip() for p in disp_input.split(',') if p.strip()]
			if len(parts) != 2:
				typer.echo("Please provide exactly two comma-separated indices for a bond, e.g. '1,2'")
				continue
			try:
				i1, i2 = int(parts[0]), int(parts[1])
				if not (1 <= i1 <= len(atoms) and 1 <= i2 <= len(atoms)):
					typer.echo("Bond indices out of range; try again.")
					continue
				bond_pair = (i1 - 1, i2 - 1)
				disp_idx = None
				break
			except ValueError:
				typer.echo("Invalid bond indices; enter two integers like '1,2'.")
				continue

		# ask for delta magnitude in Å
		delta = typer.prompt("Finite-difference displacement magnitude in Å", type=float, default=0.01)
		# construct template; for bond displacements we don't need a placeholder
		lines = []
		for idx, (atom, x, y, z) in enumerate(coords_entries, start=1):
			# We require bond displacements 'n,m' so we always output the full static geometry
			lines.append(f"{atom} {x:.6f} {y:.6f} {z:.6f}")
		coords = "\n".join(lines)
		# If coords were entered interactively, compute µ_prime/µ_double_prime using the same
		# pipeline as the non-interactive path: prompt for spin (if needed), run
		# CCSD optimization (will fail loudly if unavailable), then compute
		# finite-difference dipole derivatives.
		if coords:
			if specified_spin is None:
				specified_spin = typer.prompt("Spin value", type=int)
			assert specified_spin is not None
			specified_spin_int = int(specified_spin)
			# ensure delta is a float (use the same local variable or default)
			delta_val = float(delta) if delta is not None else 0.01
			# Use CCSD optimization
			try:
				coords_to_use = optimize_geometry_ccsd(coords, specified_spin_int, basis=basis_set)
			except Exception as e:
				typer.secho(f"CCSD optimization failed: {e}", fg="red", err=True)
				typer.secho("Try using --basis=STO-3G for a smaller basis set", fg="yellow")
				raise typer.Exit(code=2)
			# compute dipole derivatives
			try:
				µ_prime_val, µ_double_prime_val = compute_µ_derivatives(coords_to_use, specified_spin_int, delta=delta_val, bond_pair=bond_pair, basis=basis_set)
			except Exception as e:
				typer.secho(f"Error computing dipole derivatives from coordinates: {e}", fg="red", err=True)
				raise typer.Exit(code=1)
			µ_prime = float(µ_prime_val)
			µ_double_prime = float(µ_double_prime_val)
		else:
			# If somehow coords is empty, fall back to prompting for derivatives
			µ_prime = typer.prompt("First derivative of dipole at equilibrium µ_prime(0) in C·m/m", type=float, default=1e-30)
			µ_double_prime = typer.prompt("Second derivative µ_double_prime(0) in C·m/m^2", type=float, default=0.0)

	# prompt for fwhm if still None
	if fwhm is None:
		fwhm = typer.prompt("Assumed FWHM of the overtone band in cm^-1", type=float, default=50.0)

	# initialize globals (expects kg)
	setup_globals(m_a_kg, m_b_kg, fundamental_frequency, observed_frequency, overtone_order)

	# compute transition dipole M for 0->n overtone (ensure numeric types)
	# At this point, µ_prime, µ_double_prime and fwhm should have been provided or prompted.
	# If coords were provided we already computed µ_prime/µ_double_prime above using the optimized geometry
	if coords is None:
		µ_prime = float(µ_prime)
		µ_double_prime = float(µ_double_prime)
	if fwhm is None:
		fwhm = 50.0
	fwhm = float(fwhm)
	# Add detailed debugging output with ultra-high precision
	typer.echo(f"Debug: µ_prime(0) = {µ_prime:.20e} C·m/m")
	typer.echo(f"Debug: µ_double_prime(0) = {µ_double_prime:.20e} C·m/m^2")
	typer.echo(f"Debug: Morse parameter a = {mm.a:.20e}")
	typer.echo(f"Debug: Morse parameter λ = {mm.λ:.20e}")
	typer.echo(f"Debug: Overtone order n = {overtone_order}")
	
	try:
		# retrieve a and λ from the main module after setup_globals ran
		Mval = M_0n(overtone_order, mm.a, mm.λ, µ_prime, µ_double_prime)
		typer.echo(f"Debug: Raw transition dipole M = {Mval:.25e} C·m")
		
		# Check if the value is exactly zero
		if abs(Mval) == 0.0:
			typer.echo("Warning: Transition dipole is exactly zero - check dipole derivatives and bond selection")
			
	except Exception as e:
		typer.secho(f"Error computing M: {e}", fg="red", err=True)
		raise typer.Exit(code=1)

	integrated = integrated_molar_absorptivity(Mval)
	eps_max = epsilon_peak_from_integrated(integrated, fwhm)

	# Handle scaling factor - use command line option if provided, otherwise prompt
	if scaling_factor is not None:
		total_scaling = 10 ** scaling_factor
		typer.echo(f"Using command-line scaling factor: 10^{scaling_factor} = {total_scaling:.5e}")
	else:
		# Prompt for scaling factor with helpful suggestions
		typer.echo("\nScaling Factor Selection:")
		typer.echo("  - NIR (Near-Infrared): Consider 35, 45, or up to 64 for short-wave NIR")
		typer.echo("  - MIR (Mid-Infrared): Usually 1 is fine, sometimes up to 15 is necessary")
		typer.secho("See references in the README documentation for more information.", fg="cyan")
		typer.echo("  - Upper limit: 64")
		scaling_exponent = typer.prompt("Enter scaling exponent (e.g., 1 for 1e1, 64 for 1e64)", type=int, default=1)
		total_scaling = 10 ** scaling_exponent
	
	typer.echo(f"\n=== RESULTS ===")
	typer.echo(f"Computed M_0-> {overtone_order}: {Mval:.25e} C·m")
	typer.echo(f"Integrated molar absorptivity: {integrated:.25e} cm M^-1")
	typer.echo(f"Estimated ε_max: {eps_max:.25e} M^-1 cm^-1")
	
	# Scaled result
	scaled_eps = eps_max * total_scaling
	typer.echo(f"\n=== SCALED ε_max ===")
	typer.echo(f"Scaling factor: {total_scaling:.5e}")
	typer.echo(f"Scaled ε_max: {scaled_eps:.25e} M^-1 cm^-1")


if __name__ == "__main__":
	app()