import numpy as np
import scipy.constants
import scipy.special

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
	x_e = (ṽ_e * n - ṽ_obs) / (ṽ_e * (n**2 + 0.5))

	# dissociation energy in cm^-1
	D_e_cm = (ṽ_e / (4 * x_e))

	# conversion of dissociation energy to Joules
	hc = scipy.constants.Planck * scipy.constants.speed_of_light * 100  # h*c in J*cm
	D_e = D_e_cm * hc

	# harmonic angular frequency
	w_e = 2 * (np.pi) * (scipy.constants.speed_of_light) * ṽ_e

	# morse paramter a
	a = (w_e) / np.sqrt((2 * D_e) / µ)

	# dimensionless morse parameter λ (DO NOT CONFUSE WITH PYTHON lambda)
	# λ = sqrt(2 µ D_e) / (a ħ)
	λ = np.sqrt(2 * µ * D_e) / (a * scipy.constants.hbar)

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
	"""
	return np.sqrt(a * (2*λ - 2*v - 1) * scipy.special.gamma(v + 1) / scipy.special.gamma(2*λ - v))


# ===== DIPLOE EXPANSION & OVERLAP INTEGRALS =====

# Dipole expansion around Q=0 (user variables preserved):
# µ(Q) = µ_0 + µ'(0) Q + 1/2 µ''(0) Q^2 + ...
# We will work with the derivatives: mu1 = µ'(0), mu2 = µ''(0)

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
def M_if(v_i, v_f, a, λ, mu1, mu2=0.0):
	"""Approximate vibrational transition dipole M_{i->f}.

	Parameters
	----------
	v_i, v_f : int
		Initial and final vibrational quantum numbers.
	a : float
		Morse parameter a (units: 1/length, e.g., 1/m).
	λ : float
		Dimensionless Morse parameter.
	mu1 : float
		First derivative of the molecular dipole at equilibrium, µ'(0)
		(units: C·m per meter, i.e., C). Typical molecular-scale
		vibrational derivatives are ~1e-30 to 1e-29 C·m/m for X–H/O–H/N–H
		stretches (user should supply an appropriate value).
	mu2 : float, optional
		Second derivative µ''(0) (units: C·m per m^2). Typical scales
		are ~1e-40 to 1e-39 C·m/m^2. Default 0.0 to ignore quadratic term.

	Returns
	-------
	M : float
		Transition dipole (C·m).
	"""
	# mu1 = µ'(0), mu2 = µ''(0)
	M = mu1 * S1(v_i, v_f, a, λ)
	if mu2 != 0.0:
		M += 0.5 * mu2 * S2(v_i, v_f, a, λ)
	return M


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
	alpha_n = 2 * λ - 2 * n - 1
	c = laguerre_c_series(n, alpha_n)
	N0 = N_v(0, a, λ)
	Nn = N_v(n, a, λ)
	m = np.arange(0, n+1)
	beta = 2 * λ - 5 + m
	# mask out non-positive beta to avoid Gamma singularities
	mask = beta > 0
	if not np.any(mask):
		return 0.0
	beta_m = beta[mask]
	cm = c[mask]
	mm = m[mask]
	# compute special functions safely; guard against non-positive λ
	if λ <= 0:
		raise ValueError("λ must be positive for Morse integrals")
	with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
		G = scipy.special.gamma(beta_m)
		ψ = scipy.special.digamma(beta_m)
		# compute I1 = Gamma(beta) * (digamma(beta) - ln(2λ)) but skip non-finite parts
		log2λ = np.log(2.0 * λ)
		I1 = G * ψ - log2λ * G
		# replace any non-finite contributions with zero so sum is stable
		I1 = np.where(np.isfinite(I1), I1, 0.0)
	terms = ((-1.0) ** mm) * (cm / scipy.special.gamma(mm + 1)) * I1
	return - (N0 * Nn) / (a ** 2) * np.sum(terms)


def S2_0n(n, a, λ):
	"""Compute S2 = <ψ_0|Q^2|ψ_n> using finite-sum reduction for overtone n.

	Follows the formula:
	S2 = +N0*Nn / a^3 * sum_{m=0}^n (-1)^m (c_m / m!) I_m^{(2)}
	I_m^{(2)} = Gamma(beta) * [ ψ(beta)^2 + ψ1(beta) - 2 ln(2λ) ψ(beta) + (ln(2λ))^2 ]
	where beta = 2λ-5+m
	"""
	alpha_n = 2 * λ - 2 * n - 1
	c = laguerre_c_series(n, alpha_n)
	N0 = N_v(0, a, λ)
	Nn = N_v(n, a, λ)
	m = np.arange(0, n+1)
	beta = 2 * λ - 5 + m
	mask = beta > 0
	if not np.any(mask):
		return 0.0
	beta_m = beta[mask]
	cm = c[mask]
	mm = m[mask]
	# compute special functions safely; guard against non-positive λ
	if λ <= 0:
		raise ValueError("λ must be positive for Morse integrals")
	with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
		G = scipy.special.gamma(beta_m)
		ψ = scipy.special.digamma(beta_m)
		ψ1 = scipy.special.polygamma(1, beta_m)
		log2λ = np.log(2.0 * λ)
		L2 = (log2λ) ** 2
		I2 = G * (ψ ** 2 + ψ1 - 2.0 * log2λ * ψ + L2)
		I2 = np.where(np.isfinite(I2), I2, 0.0)
	terms = ((-1.0) ** mm) * (cm / scipy.special.gamma(mm + 1)) * I2
	return (N0 * Nn) / (a ** 3) * np.sum(terms)


def M_0n(n, a, λ, mu1, mu2=0.0):
	"""Convenience wrapper: compute M_{0->n} ≈ mu1*S1 + 0.5*mu2*S2

	Parameters
	----------
	n : int
		Overtone quantum number (final state).
	a, λ : floats
		Morse parameters (see file-level definitions).
	mu1 : float
		µ'(0) (units: C·m/m). Typical expected range: 1e-30 -- 1e-29 C·m/m.
	mu2 : float, optional
		µ''(0) (units: C·m/m^2). Typical expected range: 1e-40 -- 1e-39 C·m/m^2.

	Returns
	-------
	M : float
		Transition dipole M_{0->n} in C·m.
	"""
	S1 = S1_0n(n, a, λ)
	S2 = S2_0n(n, a, λ)
	
	# Debug output for overlap integrals
	print(f"Debug: S1 overlap integral = {S1:.6e}")
	print(f"Debug: S2 overlap integral = {S2:.6e}")
	
	M = mu1 * S1
	print(f"Debug: μ1 * S1 = {mu1:.6e} * {S1:.6e} = {M:.6e}")
	
	if mu2 != 0.0:
		M2_contrib = 0.5 * mu2 * S2
		print(f"Debug: 0.5 * μ2 * S2 = 0.5 * {mu2:.6e} * {S2:.6e} = {M2_contrib:.6e}")
		M += M2_contrib
		print(f"Debug: Total M = {M:.6e}")
	else:
		print("Debug: μ2 = 0, no second-order contribution")
	
	return M


# Conversion to integrated molar absorptivity and peak ε for Gaussian lineshape
def integrated_molar_absorptivity(M):
	"""Return integrated molar absorptivity (cm M^-1) from transition dipole M (C·m).

	Parameters
	----------
	M : float
		Transition dipole in C·m.

	Returns
	-------
	integrated : float
		Integrated molar absorptivity, in units cm M^-1.
	"""
	return 4.319e-9 * (np.abs(M)**2)


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




	