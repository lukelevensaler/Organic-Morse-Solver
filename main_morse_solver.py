import numpy as np
import scipy.constants
import scipy.special

from stabilization import high_precision_arithmetic as hp

# === Unit Conversion Constants ===
DEBYE_TO_C_M = 3.33564e-30  # 1 Debye in Coulomb·meter
ANGSTROM_TO_M = 1e-10       # 1 Ångström in meters

# ===== BASICS ======
# Note: compute all quantities at runtime via setup_globals(...) to avoid
# referencing undefined names at import time (A, B, etc. were placeholders).

def setup_globals(A, B, fundamental_frequency, observed_frequency, overtone_order):
	"""Compute and install global Morse parameters from user inputs.

	This function sets module-level variables used by the rest of the code so
	functions defined earlier that reference those globals will work at runtime.
	"""
	global m1, m2, ṽ_e, ṽ_obs, n, µ, x_e, D_e_cm, hc, D_e, w_e, a, λ, V, Ẽ_v

	# defining inputs
	m1 = A
	m2 = B
	ṽ_e = fundamental_frequency  # in cm^-1
	ṽ_obs = observed_frequency  # in cm^-1
	n = overtone_order  # integer

	# reduced mass
	µ = (m1 * m2) / (m1 + m2)

	# anharmonicity constant
	# ν_obs ≈ n ν_e − n(n+1) ν_e x_e  ⇒  x_e = (n ν_e − ν_obs) / (n(n+1) ν_e)
	x_e = (ṽ_e * n - ṽ_obs) / (ṽ_e * n * (n + 1))

	# dissociation energy in cm^-1 (use magnitude of x_e to avoid
	# sign-convention issues that would make D_e negative)
	D_e_cm = ṽ_e / (4 * abs(x_e))

	# conversion of dissociation energy to Joules
	hc = scipy.constants.Planck * scipy.constants.speed_of_light * 100  # h*c in J*cm
	D_e = D_e_cm * hc

	# harmonic angular frequency
	w_e = 2 * (np.pi) * (scipy.constants.speed_of_light) * ṽ_e

	# morse paramter a; use |D_e| to ensure a real, positive value
	a = w_e / np.sqrt((2 * abs(D_e)) / µ)

	# dimensionless morse parameter λ (DO NOT CONFUSE WITH PYTHON lambda)
	# λ = sqrt(2 µ D_e) / (a ħ); depend only on |D_e|
	λ = np.sqrt(2 * µ * abs(D_e)) / (a * scipy.constants.hbar)

	# morse potential (measured from equilibrium at Q=0)
	def V_local(Q):
		return D_e * (1 - np.exp(-a * Q))**2
	V = V_local

	# morse-level energy (callable preserving the name Ẽ_v)
	Ẽ_v = lambda v: ṽ_e * (v + 0.5) - (ṽ_e * x_e) * (v + 0.5)**2

	# expose the computed globals back to module namespace
	# (they are already assigned via `global`, kept here as documentation)
	return None


# ===== MORSE EIGENFUNCTIONS (analytic form) ======


def ψ_v(Q, v, a, λ):
	"""Normalized Morse eigenfunction ψ_v(Q).

	Usage: ψ_v(Q, v, a, λ)
	"""
	# exponential variable
	y = 2 * λ * np.exp(-a * Q)
	y = np.maximum(y, np.finfo(float).eps)  # restrict domain to y > 0

	# Laguerre parameter alpha
	alpha = 2 * λ - 2 * v - 1

	# normalization constant
	Nv = N_v(v, a, λ)

	# the normalized Morse eigenfunctions (in Q-space)
	return Nv * (y ** (λ - v - 0.5)) * np.exp(-y / 2.0) * scipy.special.eval_genlaguerre(v, alpha, y)
 
# normalization constant (function)
def N_v(v, a, λ):
	"""Normalization constant for Morse eigenfunction.

	N_v = sqrt( a * (2*λ - 2*v - 1) * Gamma(v+1) / Gamma(2*λ - v) )
	
	For large λ, use logarithmic arithmetic to avoid overflow.
	"""
	factor = 2*λ - 2*v - 1
	
	if factor <= 0:
		print(f"N_v Debug: Invalid factor {factor} for v={v}, λ={λ}")
		return 0.0
	
	# Check if we need logarithmic arithmetic
	gamma_arg = 2*λ - v
	if gamma_arg > 170:  # gamma(171) overflows
		print(f"N_v Debug: Using logarithmic arithmetic for large gamma argument {gamma_arg}")
		
		# N_v = sqrt(a * factor * exp(log_gamma(v+1) - log_gamma(2λ-v)))
		# log(N_v) = 0.5 * (log(a) + log(factor) + log_gamma(v+1) - log_gamma(2λ-v))
		log_a = np.log(a)
		log_factor = np.log(factor)
		log_gamma_v1 = scipy.special.loggamma(v + 1)
		log_gamma_2λv = scipy.special.loggamma(2*λ - v)
		
		log_N_v = 0.5 * (log_a + log_factor + log_gamma_v1 - log_gamma_2λv)
		
		print(f"N_v Debug: log_a={log_a:.6e}, log_factor={log_factor:.6e}")
		print(f"N_v Debug: log_gamma({v+1})={log_gamma_v1:.6e}")
		print(f"N_v Debug: log_gamma({2*λ - v})={log_gamma_2λv:.6e}")
		print(f"N_v Debug: log_N_v={log_N_v:.6e}")
		
		# Use high-precision evaluation of N_v to avoid forced underflow/overflow
		log_N_v_hp = hp.high_precision_log_N_v(v, float(a), float(λ))
		N_v_hp = hp.high_precision_N_v(v, float(a), float(λ))
		print(f"N_v Debug: high-precision log_N_v={log_N_v_hp}, N_v={N_v_hp}")
		return float(N_v_hp)
	else:
		# Use direct computation for moderate values
		result = np.sqrt(a * factor * scipy.special.gamma(v + 1) / scipy.special.gamma(2*λ - v))
		print(f"N_v Debug: Direct computation, N_v = {result:.6e}")
		return result


def _convert_mu_derivative_to_SI(mu_value: float, order: int) -> float:
	"""Convert µ-derivative expressed in Debye·Å^{-order} to SI.

	Parameters
	----------
	mu_value : float
		Derivative magnitude in Debye per Å^{order}.
	order : int
		Derivative order (1 → µ′, 2 → µ″, etc.).

	Returns
	-------
	float
		Derivative in Coulomb·meter^{1-order} suitable for SI overlap integrals.
	"""
	if mu_value == 0.0:
		return 0.0
	return mu_value * DEBYE_TO_C_M / (ANGSTROM_TO_M ** order)


def _overlap_in_angstrom_units(overlap_value: float, power: int) -> float:
	"""Express overlap integral ⟨Q^power⟩ in Å^{power} for Debye bookkeeping."""
	if power == 0:
		return overlap_value
	return overlap_value / (ANGSTROM_TO_M ** power)


# ===== DIPLOE EXPANSION & OVERLAP INTEGRALS =====

# Dipole expansion around Q=0 (user variables preserved):
# µ(Q) = µ_0 + µ_prime(0) Q + 1/2 µ_double_prime(0) Q^2 + ...
# We will work with the derivatives: µ_prime = µ_prime(0), µ_double_prime = µ_double_prime(0)

def laguerre_coeffs(n, alpha):
	"""Return coefficients c_j for L_n^{(alpha)}(y) = sum_{j=0}^n c_j y^j.

	c_j = (-1)^j / j! * binom(n+alpha, n-j)
	We compute binom using gamma to allow non-integer alpha.
	Returns array of length n+1 where coeffs[j] corresponds to y^j.
	"""
	j = np.arange(0, n+1)
	# Compute binomial-like term via log-gamma to avoid invalid divisions when
	# arguments to Gamma are non-positive integers (which produce infinities).
	# binom(n+alpha, n-j) = exp(gammaln(n+alpha+1) - (gammaln(n-j+1)+gammaln(alpha+j+1)))
	log_numer = scipy.special.gammaln(n + alpha + 1)
	log_denom = scipy.special.gammaln(n - j + 1) + scipy.special.gammaln(alpha + j + 1)
	# Where log_denom is not finite (singular Gamma), set the log-binomial to -inf
	log_binom = log_numer - log_denom
	# exponentiate safely; non-finite log_binom becomes 0.0 after exp(-inf)
	with np.errstate(over='ignore', invalid='ignore'):
		binom_vals = np.exp(np.where(np.isfinite(log_binom), log_binom, -np.inf))

	# j! = Gamma(j+1) is always finite for non-negative integer j, so safe to compute
	j_fact = np.exp(scipy.special.gammaln(j + 1))
	coeffs = ((-1.0) ** j) / j_fact * binom_vals
	return coeffs


def overlap_Sk(v_i, v_f, a, λ, k):
	"""Compute S_k^{(i,f)} = <ψ_i| Q^k |ψ_f> for k=1 or 2 using finite-sum reduction.

	- v_i, v_f: vibrational quantum numbers (integers)
	- a, λ: Morse parameters
	- k: integer 1 or 2

	Returns a float value for the overlap integral in Q-space.
	"""
	# parameters for the two Laguerre polynomials
	alpha_i = 2*λ - 2*v_i - 1
	alpha_f = 2*λ - 2*v_f - 1

	# coefficients for each polynomial: L_{v}^{(alpha)}(y) = sum_j c_j y^j
	c_i = laguerre_coeffs(v_i, alpha_i)
	c_f = laguerre_coeffs(v_f, alpha_f)

	# exponent from wavefunctions: ψ_v ∝ y^{λ-v-1/2} e^{-y/2} L_v^{(α)}(y)
	power_i = λ - v_i - 0.5
	power_f = λ - v_f - 0.5

	# overall power in integrand y^{p-1} with p = power_i + power_f + j + l + 1
	# From measure dQ = -(1/a) dy/y and two wavefunctions -> y^{power_i+power_f} * (1/y)
	# so integrand y^{power_i+power_f - 1 + j + l} e^{-y} * (ln(y/(2λ))^k)
	base_power = power_i + power_f

	# normalization constants
	N_i = N_v(v_i, a, λ)
	N_f = N_v(v_f, a, λ)

	# prepare polygamma and gamma functions via scipy
	total = 0.0
	# double sum over polynomial coefficients
	for j, cj in enumerate(c_i):
		for l, cl in enumerate(c_f):
			coeff = cj * cl
			beta = base_power - 0.0 + j + l  # exponent of y in y^{beta}
			# integral uses y^{beta-1} e^{-y} so use Gamma(beta)
			# ensure argument for Gamma is positive; numeric λ should make it so
			if beta <= 0:
				# Gamma singular or undefined; skip or handle numerically
				continue
			G = scipy.special.gamma(beta)
			if k == 0:
				term = coeff * G
			elif k == 1:
				term = coeff * G * scipy.special.digamma(beta)
			elif k == 2:
				term = coeff * G * (scipy.special.digamma(beta)**2 + scipy.special.polygamma(1, beta))
			else:
				raise ValueError("k must be 0,1 or 2")
			total += term

	# prefactors: from change of variable and normalization and powers of (2λ)
	# Q^k contributes (-1/a)^k * (ln(y/(2λ)))^k; the ln factors were handled above via k
	# the remaining prefactor is from dQ = -dy/(a y) and the y^... used Gamma with y^{beta-1}
	prefactor = (N_i * N_f) * ((-1.0 / a) ** k)
	# note: sign from dQ cancels when integrating 0->∞ because limits invert; we take absolute
	return prefactor * total


def S1(v_i, v_f, a, λ):
	"""S_1 = <ψ_i|Q|ψ_f>"""
	return overlap_Sk(v_i, v_f, a, λ, k=1)


def S2(v_i, v_f, a, λ):
	"""S_2 = <ψ_i|Q^2|ψ_f>"""
	return overlap_Sk(v_i, v_f, a, λ, k=2)


# Transition dipole using low-order expansion
def M_if(v_i, v_f, a, λ, µ_prime, µ_double_prime=0.0):
	"""Approximate transition dipole ``M_{i→f}`` returned in Debye.

	The inputs ``µ_prime`` / ``µ_double_prime`` are now expected in Debye per Å
	and Debye per Å² respectively. They are converted internally to SI before
	forming the overlap with ``S₁`` and ``S₂``. The returned transition dipole
	is expressed in Debye for user-facing convenience, while downstream
	intensity formulas convert back to SI as needed.
	"""
	S1_val = S1(v_i, v_f, a, λ)
	S2_val = S2(v_i, v_f, a, λ)

	mu1_si = _convert_mu_derivative_to_SI(µ_prime, 1)
	mu2_si = _convert_mu_derivative_to_SI(µ_double_prime, 2)

	M_si = mu1_si * S1_val
	if µ_double_prime != 0.0:
		M_si += 0.5 * mu2_si * S2_val

	return M_si / DEBYE_TO_C_M


# ===== Associated Laguerre and 0→n overtone overlaps =====

def laguerre_c_series(n, alpha):
	"""Return c_m coefficients for the representation

	L_n^{(alpha)}(y) = sum_{m=0}^n (-1)^m (c_m / m!) y^m

	where c_m = binom(n+alpha, n-m) = Gamma(n+alpha+1) / (Gamma(n-m+1)*Gamma(alpha+m+1)).
	Returns array c of length n+1 with c[m] = c_m.
	"""
	m = np.arange(0, n+1)
	# Use log-gamma to compute c_m = Gamma(n+alpha+1) / (Gamma(n-m+1)*Gamma(alpha+m+1))
	log_numer = scipy.special.gammaln(n + alpha + 1)
	log_denom = scipy.special.gammaln(n - m + 1) + scipy.special.gammaln(alpha + m + 1)
	log_c = log_numer - log_denom
	with np.errstate(over='ignore', invalid='ignore'):
		c = np.exp(np.where(np.isfinite(log_c), log_c, -np.inf))
	return c


def S1_0n(n, a, λ):
	"""Compute S1 = <ψ_0|Q|ψ_n> using the finite-sum reduction for overtone n.

	Follows the formula:
	S1 = -N0*Nn / a^2 * sum_{m=0}^n (-1)^m (c_m / m!) I_m^{(1)}
	I_m^{(1)} = Gamma(beta) * ψ(beta) - ln(2λ) * Gamma(beta), with beta = 2λ-5+m
	"""
	# For all parameter regimes, use the dedicated high-precision
	# implementation as the single source of truth. It internally
	# handles both moderate and extreme cases robustly.
	return hp.high_precision_S1_0n(n, float(a), float(λ))


def S2_0n(n, a, λ):
	"""Compute S2 = <ψ_0|Q^2|ψ_n> using finite-sum reduction for overtone n.

	Follows the formula:
	S2 = +N0*Nn / a^3 * sum_{m=0}^n (-1)^m (c_m / m!) I_m^{(2)}
	I_m^{(2)} = Gamma(beta) * [ ψ(beta)^2 + ψ1(beta) - 2 ln(2λ) ψ(beta) + (ln(2λ))^2 ]
	where beta = 2λ-5+m
	"""
	# As with S1_0n, always delegate to the high-precision
	# implementation to avoid duplicated, fragile logic here.
	return hp.high_precision_S2_0n(n, float(a), float(λ))


def S3_0n(n, a, λ):
	"""Compute S3 = <ψ_0|Q^3|ψ_n> using high-precision quadrature."""
	return hp.high_precision_S3_0n(n, float(a), float(λ))


def S4_0n(n, a, λ):
	"""Compute S4 = <ψ_0|Q^4|ψ_n> using high-precision quadrature."""
	return hp.high_precision_S4_0n(n, float(a), float(λ))


def M_0n(n, a, λ, µ_prime, µ_double_prime=0.0, µ_triple_prime=0.0, µ_quadruple_prime=0.0):
	"""Compute the 0→n transition dipole and return it in Debye.

	All µ-derivatives are expected in Debye per Å^k. The function converts
	them to SI values for the internal overlap evaluation, but reports all
	intermediate and final transition dipoles in Debye for user-facing output.
	"""
	S1 = S1_0n(n, a, λ)
	S2 = S2_0n(n, a, λ)
	S3 = S3_0n(n, a, λ) if µ_triple_prime != 0.0 else 0.0
	S4 = S4_0n(n, a, λ) if µ_quadruple_prime != 0.0 else 0.0

	S1_angstrom = _overlap_in_angstrom_units(S1, 1)
	S2_angstrom2 = _overlap_in_angstrom_units(S2, 2)
	S3_angstrom3 = _overlap_in_angstrom_units(S3, 3) if µ_triple_prime != 0.0 else 0.0
	S4_angstrom4 = _overlap_in_angstrom_units(S4, 4) if µ_quadruple_prime != 0.0 else 0.0

	print(f"Debug: S1 overlap integral = {S1:.6e} (≈ {S1_angstrom:.6e} Å)")
	print(f"Debug: S2 overlap integral = {S2:.6e} (≈ {S2_angstrom2:.6e} Å²)")
	if µ_triple_prime != 0.0:
		print(f"Debug: S3 overlap integral = {S3:.6e} (≈ {S3_angstrom3:.6e} Å³)")
	if µ_quadruple_prime != 0.0:
		print(f"Debug: S4 overlap integral = {S4:.6e} (≈ {S4_angstrom4:.6e} Å⁴)")

	mu1_si = _convert_mu_derivative_to_SI(µ_prime, 1)
	mu2_si = _convert_mu_derivative_to_SI(µ_double_prime, 2)
	mu3_si = _convert_mu_derivative_to_SI(µ_triple_prime, 3)
	mu4_si = _convert_mu_derivative_to_SI(µ_quadruple_prime, 4)

	M_si = mu1_si * S1
	M1_debye = µ_prime * S1_angstrom
	print(f"Debug: μ1 * S1 = {µ_prime:.6e} Debye/Å * {S1_angstrom:.6e} Å = {M1_debye:.6e} Debye")

	if µ_double_prime != 0.0:
		M2_si = 0.5 * mu2_si * S2
		M2_debye = 0.5 * µ_double_prime * S2_angstrom2
		print(f"Debug: 0.5 * μ2 * S2 = 0.5 * {µ_double_prime:.6e} Debye/Å² * {S2_angstrom2:.6e} Å² = {M2_debye:.6e} Debye")
		M_si += M2_si
		print(f"Debug: Total M = {(M_si / DEBYE_TO_C_M):.6e} Debye")
	else:
		print("Debug: μ2 = 0, no second-order contribution")

	if µ_triple_prime != 0.0:
		M3_si = (1.0 / 6.0) * mu3_si * S3
		M3_debye = (1.0 / 6.0) * µ_triple_prime * S3_angstrom3
		print(f"Debug: (1/6) * μ3 * S3 = (1/6) * {µ_triple_prime:.6e} Debye/Å³ * {S3_angstrom3:.6e} Å³ = {M3_debye:.6e} Debye")
		M_si += M3_si
		print(f"Debug: Total M after cubic term = {(M_si / DEBYE_TO_C_M):.6e} Debye")
	else:
		print("Debug: μ3 = 0, no third-order contribution")

	if µ_quadruple_prime != 0.0:
		M4_si = (1.0 / 24.0) * mu4_si * S4
		M4_debye = (1.0 / 24.0) * µ_quadruple_prime * S4_angstrom4
		print(f"Debug: (1/24) * μ4 * S4 = (1/24) * {µ_quadruple_prime:.6e} Debye/Å⁴ * {S4_angstrom4:.6e} Å⁴ = {M4_debye:.6e} Debye")
		M_si += M4_si
		print(f"Debug: Total M after quartic term = {(M_si / DEBYE_TO_C_M):.6e} Debye")
	else:
		print("Debug: μ4 = 0, no fourth-order contribution")

	return M_si / DEBYE_TO_C_M


# Conversion to integrated molar absorptivity and peak ε for Gaussian lineshape
def integrated_molar_absorptivity(M_debye: float):
	"""Return integrated molar absorptivity (cm M⁻¹) given ``M`` in Debye."""
	M_abs = np.abs(M_debye)
	return 4.319e-9 * (M_abs ** 2)


def epsilon_peak_from_integrated(integrated, fwhm_cm_inv):
	"""
	For Gaussian lineshape, return ε_max given integrated area and FWHM in cm^-1.

	Parameters
	----------
	integrated : float
		Integrated molar absorptivity (cm M^-1).
	fwhm_cm_inv : float
		Full-width at half-maximum in cm^-1.
  """
	# For a Gaussian lineshape, area A = eps_max * FWHM * sqrt(pi / (4*ln(2)))
	# therefore eps_max = A / (FWHM * sqrt(pi / (4*ln(2))))
	if fwhm_cm_inv <= 0:
		raise ValueError("fwhm_cm_inv must be positive")
	factor = np.sqrt(np.pi / (4.0 * np.log(2.0)))
	return integrated / (fwhm_cm_inv * factor)




	