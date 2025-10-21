# This CLI performs the following calulations to determine the molar extinction coefficent ε in $$\{M·cm}^{-1}\$$ for any organic molecule's NIR overtone.

## Inputs required

* Molar mass (amu) of element A in an A-B stretch
* Molar mass (amu) of element B in an A-B stretch
* Fundamental frequency of molecule in $$\{cm}^{-1}\$$ (wavenumber)
* Observed frequency of molecule in $$\{cm}^{-1}\$$ (wavenumber)
* Approximate integer overtone order of the molecule's observed wavenumber relative to the fundamental wavenumber

## Allowed organic stretches:

* C–H 
* C=O 
* C–N 
* N–H 
* O–H

---

## 1. Coordinate, reduced mass, and Morse potential

- Mass-weighted stretch coordinate: denote the (mass-weighted) normal coordinate as $$\{Q}\$$ (units: m).
- Reduced mass for an A-B bond (e.g. N–H):


$$\
\mu \=\ \frac{m_A m_B}{m_A + m_B}
\$$


- Morse potential (measured from equilibrium at $$\{Q=0}\$$):

$$\
V(Q) \=\ D_e\bigl(1 - e^{-aQ}\bigr)^2
\$$

where

  - $$\(D_e\)$$ is the dissociation energy (J),
  - $$\(a\)$$ is the Morse inverse length parameter $$\{m}^{-1}\$$.


---

## 2. Spectroscopic (vibrational) energy levels and relations

- Morse-level energy (wavenumbers, $$\tilde\nu\$$ in $$\{cm}^{-1}\$$):
$$\{
\tilde E_v \=\ \tilde\nu_e\bigl(v+\tfrac12\bigr) \ - \ \tilde\nu_e x_e\bigl(v+\tfrac12\bigr)^2}
\$$
where
  - $$\tilde\nu_e\$$ is the fundamental ($$\{cm}^{-1}\$$),
  - $$\{x_e}\$$ is the anharmonicity constant (dimensionless).

- Relation between anharmonicity and Morse well depth:
$${
\tilde\nu_e x_e = \frac{\tilde\nu_e^2}{4 D_e} \qquad \text{with} D_e \text{in}{cm}^{-1}}$$


Equivalently:

$$\{D_e = \frac{\tilde\nu_e}{4 x_e}}\$$

- Conversion from $$\{D_e}\$$ in $$\{cm}^{-1}\$$ to joules:
$$\(
D_e(\mathrm{J}) \;=\; D_e(\mathrm{cm^{-1}}) \times h c
\)$$
with $$\(h c = 1.98644586\times10^{-23}\ \mathrm{J\cdot cm}\)$$.

- Harmonic angular frequency (rad s$$\(^{-1}\)$$):
$$\(
\omega_e \;=\; 2\pi c\,\tilde\nu_e
\)$$
with $$\(c\)$$ in cm s$$\(^{-1}\)$$ when $$\(\tilde\nu_e\)$$ is in $$\{cm}^{-1}\$$.

- Relation to Morse parameter $$\(a\)$$ (SI):
$$\(
a \;=\; \frac{\omega_e}{\sqrt{2 D_e(\mathrm{J})/\mu}}.
\)$$

- Dimensionless Morse parameter $$\(\lambda\)$$:
$$\(
\lambda \;=\; \frac{\sqrt{2\mu D_e(\mathrm{J})}}{a\hbar} \;=\; \frac{1}{2 x_e}\quad(\text{spectroscopic shortcut}).
\)$$

---

## 3. Morse eigenfunctions (analytic form)

Introduce the exponential variable
$$\(
y \;=\; 2\lambda e^{-aQ}, \qquad y>0.
\)$$

The normalized Morse eigenfunctions (in $$\(Q\)$$-space, written using $$\(y\)$$) are:
$$\(
\boxed{\;\psi_v(Q) \;=\; N_v\,y^{\lambda - v - 1/2}\,e^{-y/2}\,L_v^{(2\lambda - 2v - 1)}(y)\;}
\)$$
where
- $$\(L_v^{(\alpha)}(y)\)$$ is the associated Laguerre polynomial, and
- the normalization constant is
$$\(
N_v \;=\; \sqrt{ \dfrac{a\,(2\lambda - 2v - 1)\,\Gamma(v+1)}{\Gamma(2\lambda - v)} } .
\)$$

**Remarks:**
- Allowed $$\(v\)$$ satisfy $$\(v < \lambda - \
- For large $$\(\lambda\)$$ (stiff bond), the Morse eigenfunctions approach harmonic oscillator shapes near equilibrium.

---

## 4. Dipole expansion and transition dipole

- Expand the molecular dipole along the normal coordinate $$\(Q\)$$:
$$\(
\mu(Q) \;=\; \mu_0 + \mu'(0) Q + \tfrac12\mu''(0) Q^2 + \cdots
\)$$
where $$\(\mu'(0) = d\mu/dQ|_{0}\)$$, etc.

- Vibrational transition dipole (approximate, keeping low-order derivatives):
$$\(
M_{i\to f} \;=\; \langle\psi_i|\mu(Q)|\psi_f\rangle \approx \mu'(0)\langle\psi_i|Q|\psi_f\rangle + \tfrac12\mu''(0)\langle\psi_i|Q^2|\psi_f\rangle.
\)$$

Define the overlap integrals
$$\(
S_1^{(i,f)} \equiv \langle\psi_i|Q|\psi_f\rangle,\qquad S_2^{(i,f)} \equiv \langle\psi_i|Q^2|\psi_f\rangle.
\)$$

---

## 5. Change of variables and finite-sum reduction

Under $$\(y=2\lambda e^{-aQ}\)$$, we have
$$\(
Q \;=\; -\frac{1}{a}\ln\frac{y}{2\lambda},\qquad dQ = -\frac{1}{a}\frac{dy}{y}.
\)$$
So the overlaps reduce to integrals of the form
$$\(
\int_0^{\infty} y^{p-1} e^{-y} L_m^{(\alpha_m)}(y) L_n^{(\alpha_n)}(y) \Bigl(\ln\frac{y}{2\lambda}\Bigr)^k \,dy,
\)$$
with small integer $$\(k\)$$ (0,1,2).

Use the associated-Laguerre expansion
$$\(
L_n^{(\alpha)}(y) = \sum_{j=0}^{n} \frac{(-1)^j}{j!} \binom{n+\alpha}{n-j} y^j
\)$$
(which is a finite polynomial). Multiplying two such polynomials gives a finite double sum and the integrals reduce to Gamma-function evaluations and derivatives.

Key Gamma/digamma identities used:
$$\(
\int_0^{\infty} y^{\beta-1} e^{-y} \,dy = \Gamma(\beta),
\)$$
$$\(
\int_0^{\infty} y^{\beta-1} e^{-y} \ln y\,dy = \Gamma(\beta)\,\psi(\beta),
\)$$
$$\(
\int_0^{\infty} y^{\beta-1} e^{-y} \ln^2 y\,dy = \Gamma(\beta)\bigl[\psi(\beta)^2 + \psi^{(1)}(\beta)\bigr],
\)$$
where $$\(\psi\)$$ is the digamma function and $$\(\psi^{(1)}\)$$ is the trigamma.

Therefore each overlap becomes a finite sum of terms like $$\(\Gamma(\beta)\)$$, $$\(\Gamma(\beta)\psi(\beta)\)$$, and $$\(\Gamma(\beta)\psi^{(1)}(\beta)\)$$.

---

## 6. Conversion from transition dipole to molar absorptivity

- Integrated molar absorptivity in conventional units (cm M$$\(^{-1}\)$$) relates to the squared transition dipole via:
$$\(
\boxed{\;\int \varepsilon(\tilde\nu)\,d\tilde\nu \;=\; 4.319\times10^{-9}\;|M_{i\to f}|^2\;}
\)$$
(valid when $$\(\tilde\nu\)$$ is in $$\{cm}^{-1}\$$ and $$\(M\)$$ in C·m).

- For a Gaussian lineshape with FWHM $$\(\Delta\tilde\nu\)$$, the peak molar extinction is
$$\(
\varepsilon_{\max} \;=\; \frac{\int\varepsilon(\tilde\nu)\,d\tilde\nu}{\Delta\tilde\nu}\sqrt{\frac{4\ln2}{\pi}}.
\)$$

## 7. Associated Laguerre Polynomial for Overtone
- Degree = n
$$\(
L_n^{(\alpha_n)}(y) = \sum_{m=0}^{n} (-1)^m \frac{c_m}{m!} y^m
\)$$
- $$\(\alpha_n = 2\lambda - 2 n - 1\)$$
- Coefficients:
$$\(
\begin{aligned}
 c_0 &= \binom{n+\alpha_n}{n},\\
 c_1 &= \binom{n+\alpha_n}{n-1},\\
 \vdots \\
 c_n &= \frac{1}{n!} \binom{n+\alpha_n}{0}
\end{aligned}
\)$$

---

## 8. Overlap Integrals S1 and S2
- Linear term:
$$\(
S_1 = -\frac{N_0 N_n}{a^2} \sum_{m=0}^{n} (-1)^m \frac{c_m}{m!} I_m^{(1)}, \quad I_m^{(1)} = \int_0^\infty y^{2\lambda-6+m} e^{-y} (\ln y - \ln 2\lambda) dy
\)$$
- Quadratic term:
$$\(
S_2 = \frac{N_0 N_n}{a^3} \sum_{m=0}^{n} (-1)^m \frac{c_m}{m!} I_m^{(2)}, \quad I_m^{(2)} = \int_0^\infty y^{2\lambda-6+m} e^{-y} (\ln y - \ln 2\lambda)^2 dy
\)$$
- Each integral can be expressed using **Gamma, digamma, and trigamma functions** for numerical evaluation.

## 9. Laguerre Polynomial Coefficients (Degree n)
- For overtone 0→n, $$\(\alpha_n = 2\lambda - 2 n - 1\)$$
- The associated Laguerre polynomial expansion:
$$\(
L_n^{(\alpha_n)}(y) = \sum_{m=0}^{n} (-1)^m c_m y^m / m!
\)$$
- Binomial-based coefficients:
$$\(
\begin{aligned}
c_0 &= \binom{\alpha_n + n}{n},\\
c_1 &= \binom{\alpha_n + n}{n-1},\\
&\vdots\\
c_n &= \frac{1}{n!} \binom{\alpha_n + n}{0} = \frac{1}{n!}.
\end{aligned}
\)$$

---

## 10. S1 Overlap (Q Linear Term)
$$\(
S_1 = -\frac{N_0 N_n}{a^2} \sum_{m=0}^{n} (-1)^m \frac{c_m}{m!} I_m^{(1)},
\)$$
with
$$\(
I_m^{(1)} = \int_0^{\infty} y^{2\lambda-6+m} e^{-y} (\ln y - \ln 2\lambda) dy.
\)$$

## 11. Compute Gamma Functions

### 11a) Express I_m^{(1)} via Gamma and Digamma Functions
$$\(
I_m^{(1)} = \Gamma(2\lambda-5+m) \psi(2\lambda-5+m) - \ln(2\lambda) \Gamma(2\lambda-5+m),
\)$$
where $$\(\psi(x)\)$$ is the digamma function.

### 11b) Term-by-Term Sum
$$\(
S_1 = -\frac{N_0 N_n}{a^2} \sum_{m=0}^{n} (-1)^m c_m I_m^{(1)}/m!.
\)$$
- High-precision evaluation or Stirling/log-Gamma approximations are recommended.

---

## 12. S2 Overlap (Q² Term)
$$\(
S_2 = \frac{N_0 N_n}{a^3} \sum_{m=0}^{n} (-1)^m \frac{c_m}{m!} I_m^{(2)},
\)$$
with
$$\(
I_m^{(2)} = \int_0^{\infty} y^{2\lambda-6+m} e^{-y} (\ln y - \ln 2\lambda)^2 dy.
\)$$

## 13. Express Gamma Fucntions

### 13a) Express I_m^{(2)} via Digamma and Trigamma Functions
$$\(
I_m^{(2)} = \Gamma(2\lambda-5+m) \Bigl[ \psi(2\lambda-5+m)^2 + \psi^{(1)}(2\lambda-5+m) - 2 \ln 2\lambda \psi(2\lambda-5+m) + (\ln 2\lambda)^2 \Bigr],
\)$$
where $$\(\psi^{(1)}(x)\)$$ is the trigamma function.

### 13b) Term-by-Term Sum
$$\(
S_2 = \frac{N_0 N_n}{a^3} \sum_{m=0}^{n} (-1)^m c_m I_m^{(2)}/m!.
\)$$

- Numerical evaluation uses the same approach as for S1.

---

## 14. Transition Dipole Matrix Element
Expand the dipole along the normal coordinate:
$$\(
\mu(Q) = \mu_0 + \mu'(0) Q + \frac12 \mu''(0) Q^2 + \cdots
\)$$

The transition dipole for 0→n is approximately:
$$\(
M_{0\to n} \approx \mu'(0) S_1 + \frac12 \mu''(0) S_2.
\)$$

## 15. Assign dipole derivatives
- Linear derivative: $$\(\mu'(0)\)$$ (user-assigned, typical scale 10⁻³⁰ to 10⁻²⁹ C·m/m for X–H, O–H, N–H, weaker for C–H)
- Quadratic derivative: $$\(\mu''(0)\)$$ (user-assigned, typically 10⁻⁴⁰ to 10⁻³⁹ C·m/m²)

## 16. Compute M_{0→n}
$$\(
M_{0\to n} = \mu'(0) S_1 + \frac12 \mu''(0) S_2
\)$$
- S1 and S2 are obtained from Step 3 for the chosen bond and overtone.

---

## 17. Integrated Molar Absorptivity
Use the general formula for NIR vibrational transitions:
$$\(
\int \varepsilon(\tilde\nu) d\tilde\nu = 4.319\times10^{-9} |M_{0\to n}|^2 \quad [\mathrm{cm\,M^{-1}}]
\)$$
- Units: C·m → M·cm⁻¹
- User can adjust constants if necessary for units.

---

## 18. Peak Molar Extinction Coefficient
Assuming a Gaussian lineshape with FWHM $$\(\Delta\tilde\nu\)$$ (user-assigned, e.g., 50–100 cm⁻¹):
$$\(
\varepsilon_{\max} = \frac{\int \varepsilon \, d\tilde\nu}{\Delta\tilde\nu} \sqrt{\frac{4\ln2}{\pi}}
\)$$
- Plug in the computed integral from Step 2 and the user-specified FWHM.

---

## 19. Remarks
- This general framework allows **any bond type and overtone** to produce a theoretical ε_max.
- The pure vibrational Morse model typically yields extremely small values for higher overtones (0→3, 0→4).
- Enhancements can be included via:
  1. Vibronic (electronic) coupling
  2. Charge-transfer contributions
  3. H-bond polarizability
  4. Multi-oscillator collective effects