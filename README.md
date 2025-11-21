

# Welcome to the Organic Morse Solver, a quantum computation for deriving molar extinction coefficients for IR/NIR peaks.


## This CLI performs the following quantum chemistry and Morse Model-based anharmonicty calulations to determine the molar extinction coefficent ε in $$\{M·cm}^{-1}\$$ for any organic molecule's IR (or NIR) overtone peak. Its algorithm can even compute ε values for fundamnetal peaks, at full anharmonic accuracy.




## Inputs required


* Molar mass (amu) of element A in an A-B stretch
* Molar mass (amu) of element B in an A-B stretch
* Fundamental frequency of molecule in $$\{cm}^{-1}\$$ (wavenumber)
* Observed frequency of molecule in $$\{cm}^{-1}\$$ (wavenumber)
* Approximate integer overtone order of the molecule's observed wavenumber relative to the fundamental wavenumber


## Morse model parameters used

For an IR (or NIR) overtone transition from $v=0$ to $v=n$, the solver assumes the standard Morse-oscillator relation between the fundamental frequency $\nu_e$, the anharmonicity constant $x_e$, and the observed overtone wavenumber $\nu_{0\to n}$:

$$
\nu_{0\to n} \approx n\,\nu_e - n(n+1)\,\nu_e x_e.
$$

Solving for $x_e$ in terms of the fundamental and observed overtone gives

$$
x_e = \frac{n\,\nu_e - \nu_{0\to n}}{n(n+1)\,\nu_e}.
$$

From $\nu_e$ and $x_e$, the dissociation energy in wavenumber units is taken as the magnitude

$$
D_e^{(\mathrm{cm}^{-1})} = \frac{\nu_e}{4\,|x_e|},
$$

which avoids sign-convention issues that can make $D_e$ negative. This $D_e^{(\mathrm{cm}^{-1})}$ is then converted to Joules and used, together with the reduced mass, to construct the Morse parameters $a$ and $\lambda$ that enter the high-precision overlap and intensity calculations.


## Allowed organic stretches:


* C–H
* C=O
* C–N
* N–H
* O–H


## Installation


### Prerequisites


- **Conda** or **Miniconda**: Required for environment management
 - Download from [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- **Git**: Required for cloning the repository


### Step-by-Step Installation


1. **Clone the Repository**
  ```bash
  git clone https://github.com/lukelevensaler/Organic-Morse-Solver.git
  cd Organic-Morse-Solver
  ```


2. **Create the Conda Environment**
 
 The repository includes an `environment.yml` file that specifies all required dependencies including:
 - **PySCF**: Quantum chemistry calculations (SCF-level theory in the current implementation)
  - **NumPy/SciPy**: Numerical computations and special functions
  - **Typer**: CLI framework
  - **PyBerny**: Geometry optimization
  - **H5PY**: Data storage for quantum chemistry results
  - **High-Precision Libraries**: Optimized BLAS/LAPACK for numerical stability
  - **Parallel Computing**: MPI support for distributed quantum chemistry calculations


  Create the environment named `morse_solver`:
  ```bash
  conda env create -f environment.yml
  ```


3. **Activate the Environment**
  ```bash
  conda activate morse_solver
  ```


4. **Verify Installation**
 
  Test that the CLI works correctly:
  ```bash
  python run_morse_model.py stretches
  ```
 
  You should see the list of allowed organic stretches:
  ```
  Allowed organic stretches:
   - C–H
   - C=O
   - C–N
   - N–H
   - O–H
  ```


### Usage After Installation


Once installed, you can run the solver from the repository directory:


```bash
# Activate the environment (if not already active)
conda activate morse_solver


# Run the CLI
python3 run_morse_model.py compute --help
```


### Troubleshooting


**Common Issues:**


1. **Conda environment creation fails**: Ensure you have sufficient disk space (~2GB) and internet connectivity
2. **PySCF import errors**: The environment includes all required quantum chemistry dependencies
3. **Permission errors**: Ensure you have write access to your conda installation directory


---


# Part A: Optimizing Molecular Geometry


This section describes the ab initio geometry optimization process, which is powered by Berny computations and Self-consistent field (SCF) evalutaions.


## A.1 Initial Geometry Setup
- Input: Cartesian coordinates in XYZ format (Element x y z) in Angstrom units
- Parse atomic symbols and positions into molecular structure
- Build PySCF molecule object with specified basis set (default: aug-cc-pVTZ)


## A.2 High-Precision SCF Calculation
Maximum precision SCF settings for geometry optimization:
- Convergence tolerance: $1 \times 10^{-12}$ (extremely tight for SCF accuracy)
- Maximum cycles: 400 (robust convergence)
- DIIS space: 15 (large space for optimal convergence)


## A.3 SCF Gradient-Based Optimization
Uses analytic SCF gradients with Berny solver for geometry optimization:


### SCF Setup (Deterministic Rigor):
- Convergence tolerance: $1 \times 10^{-12}$ (extremely tight)
- Maximum cycles: 200 (robust convergence) 
- DIIS space: 15 (maximum space)
- Direct algorithm: enabled for numerical accuracy


### Berny Geometry Optimization:
- Maximum steps: 50 (user-configurable)
- Uses analytic SCF gradients for force evaluation
- Convergence on energy and gradient thresholds


**Output**: Optimized Cartesian coordinates at SCF level


---


# Part B: Getting a Dipole Derivative from the Optimized Molecular Geometry


This section describes the finite-difference calculation of dipole moment derivatives using SCF dipole moments.


## B.1 Coordinate Displacement Strategy


### Single Bond Axis (Default):
For bond pair $(i,j)$, create bond vector and normalize:
$$
\vec{u}_{ij} = \frac{\vec{r}_j - \vec{r}_i}{|\vec{r}_j - \vec{r}_i|}
$$


Displacement along bond axis:
$$
\vec{r}_i^{\pm} = \vec{r}_i \pm \frac{\delta}{2}\vec{u}_{ij}, \quad \vec{r}_j^{\pm} = \vec{r}_j \mp \frac{\delta}{2}\vec{u}_{ij}
$$


### Dual Bond Axes (Advanced):
For two bonds sharing a common atom, e.g., $(n,x)$ and $(a,x)$:


#### Step 1: Individual Bond Vectors
$$
\vec{e}_1 = \frac{\vec{r}_x - \vec{r}_n}{|\vec{r}_x - \vec{r}_n|}, \quad \vec{e}_2 = \frac{\vec{r}_x - \vec{r}_a}{|\vec{r}_x - \vec{r}_a|}
$$


#### Step 2: Symmetric/Antisymmetric Combinations
$$
\vec{e}_{\text{sym}} = \frac{\vec{e}_1 + \vec{e}_2}{\sqrt{2}}, \quad \vec{e}_{\text{anti}} = \frac{\vec{e}_1 - \vec{e}_2}{\sqrt{2}}
$$


#### Step 3: Mass Weighting
Apply mass weighting using user-provided masses $m_1$ and $m_2$:
$$
\vec{e}_{\text{sym}}^{(i)} \leftarrow \frac{\vec{e}_{\text{sym}}^{(i)}}{\sqrt{m_i}}, \quad \vec{e}_{\text{anti}}^{(i)} \leftarrow \frac{\vec{e}_{\text{anti}}^{(i)}}{\sqrt{m_i}}
$$


#### Step 4: Renormalization
$$
\vec{e}_{\text{sym}} \leftarrow \frac{\vec{e}_{\text{sym}}}{|\vec{e}_{\text{sym}}|}, \quad \vec{e}_{\text{anti}} \leftarrow \frac{\vec{e}_{\text{anti}}}{|\vec{e}_{\text{anti}}|}
$$


**Primary displacement**: Use symmetric mode (larger projection typically).


## B.2 SCF Dipole Moment Calculations


### High-Precision Settings:
- Basis set: aug-cc-pVTZ (or higher quality if specified)
- SCF convergence: $1 \times 10^{-12}$ (extremely tight)
- Maximum cycles: 200 (robust convergence)


### SCF Dipole Calculation Sequence:
1. **SCF Calculation**: High-precision self-consistent field
2. **SCF Solution**: Restricted or unrestricted Hartree–Fock
3. **Dipole Moment**: Computed from SCF density matrices


### Dipole Moment Formula:
$$
\vec{\mu} = -\text{Tr}[\mathbf{D}^{\text{SCF}} \cdot \hat{\vec{\mu}}] + \vec{\mu}_{\text{nuc}}
$$
where $\mathbf{D}^{\text{SCF}}$ is the SCF density matrix and $\vec{\mu}_{\text{nuc}}$ is the nuclear contribution.


## B.3 Finite Difference Derivatives


### Geometries Required:
- Equilibrium: $\vec{\mu}_0$
- Positive displacement: $\vec{\mu}_+$ (geometry displaced by $+\delta$)
- Negative displacement: $\vec{\mu}_-$ (geometry displaced by $-\delta$)


### First Derivative (Linear Term):
$$
\vec{\mu}'(0) = \frac{\vec{\mu}_+ - \vec{\mu}_-}{2\delta}
$$


### Second Derivative (Quadratic Term):
$$
\vec{\mu}''(0) = \frac{\vec{\mu}_+ - 2\vec{\mu}_0 + \vec{\mu}_-}{\delta^2}
$$


### Magnitude Conversion to SI Units:
$$
|\mu'(0)| = |\vec{\mu}'(0)| \times \frac{3.33564 \times 10^{-30}}{10^{-10}} \quad \text{[C·m/m]}
$$
$$
|\mu''(0)| = |\vec{\mu}''(0)| \times \frac{3.33564 \times 10^{-30}}{10^{-20}} \quad \text{[C·m/m²]}
$$


**Output**: First and second dipole derivatives in SI units for Morse model input.


---


# High-Precision Arithmetic and Stabilization


The Morse overtone integrals in this solver are evaluated using dedicated routines in the `stabilization` subdirectory, chiefly `high_precision_arithmetic.py`. These routines are designed for extreme alternating series where individual terms are astronomically large, yet the final physical quantities (transition dipole moments) are tiny.


**API Map (functions → equations):**
- `high_precision_log_gamma(x)`: implements the Stirling-form approximation for $\ln\Gamma(x)$ in Section 1.
- `high_precision_digamma(x)`, `high_precision_polygamma(1, x)`: implement the large-$x$ expansions for $\psi(x)$ and $\psi_1(x)$ in Section 1.
- `high_precision_log_N_v(v, a, \lambda)`, `high_precision_N_v(v, a, \lambda)`: implement the normalization $N_v$ in Section 2 and Part C.3.
- `high_precision_S1_0n(n, a, \lambda)`: evaluates the linear overlap $S_1 = \langle\psi_0|Q|\psi_n\rangle$ as in Sections 4.1, 8, 10, and 11.
- `high_precision_S2_0n(n, a, \lambda)`: evaluates the quadratic overlap $S_2 = \langle\psi_0|Q^2|\psi_n\rangle$ as in Sections 4.2, 8, 12, and 13.
- `high_precision_alternating_sum_from_logs(...)`: performs the stabilized alternating sum described in Section 5.


## 1. Asymptotic Gamma and Polygamma


For large real arguments $x$, the code never forms $\Gamma(x)$ directly. Instead it works with $\ln\Gamma(x)$ and asymptotic expansions:


- Log-gamma (Stirling form) for large $x$:
$$
\ln\Gamma(x)
\;\approx\;
\Bigl(x - \tfrac12\Bigr)\ln x - x + \tfrac12\ln(2\pi),
\qquad x \gg 1.
$$


This is implemented in `high_precision_log_gamma(x)` using `Decimal.ln` for all logarithms. For moderate $x$ the SciPy special function is used and converted into a `Decimal`.


- Digamma function $\psi(x) = \frac{d}{dx}\ln\Gamma(x)$ for large $x$:
$$
\psi(x)
\;\approx\;
\ln x \;-\; \frac{1}{2x} \;-\; \frac{1}{12x^2}
\;+\; \frac{1}{120x^4}
\;-\; \frac{1}{252x^6}
\;+\;\cdots,
\qquad x \gg 1.
$$


This asymptotic series is implemented in `high_precision_digamma(x)` (with SciPy used for moderate $x$). 


- Trigamma (polygamma of order 1), $\psi_1(x) = \frac{d}{dx}\psi(x)$, is evaluated by `high_precision_polygamma(1, x)` using the corresponding high-precision representation (SciPy for moderate $x$ and asymptotics for large $x$).


All of these are performed in `Decimal` arithmetic with a globally set precision (see `getcontext().prec`) so that the logarithms and polynomial corrections retain far more than 16 digits.


## 2. Normalization Constants in Log Space


The Morse eigenfunction normalization constants
$$
N_v
=
\sqrt{
\frac{
a\,(2\lambda - 2v - 1)\,\Gamma(v+1)
}{
\Gamma(2\lambda - v)
}
}
$$
are never formed directly. Instead, `high_precision_log_N_v(v,a,\lambda)` computes
$$
\ln N_v
=
\frac12\Bigl[
\ln a
+ \ln(2\lambda - 2v - 1)
+ \ln\Gamma(v+1)
- \ln\Gamma(2\lambda - v)
\Bigr],
$$
entirely in `Decimal` log space using `high_precision_log_gamma`. The actual $N_v$ is only obtained, if needed, via
$$
N_v = \exp\bigl(\ln N_v\bigr),
$$
and underflows are handled explicitly (values below about $\ln 10^{-100}$ are clamped).


These log-space normalization constants are used both in the $S_1$ and $S_2$ overlap integrals.


## 3. Laguerre Coefficients in Log Space


The Morse overlaps used in `high_precision_S1_0n` and `high_precision_S2_0n` involve associated Laguerre polynomials $L_n^{(\alpha_n)}(y)`. In the high-precision routines, the coefficients $c_m$ of the polynomial expansion
$$
L_n^{(\alpha_n)}(y) = \sum_{m=0}^{n} c_m\, y^m
$$
are computed via Gamma functions in log form:
$$
\ln c_m
=
\ln\Gamma(n+\alpha_n+1)
\;-\;
\ln\Gamma(n-m+1)
\;-\;
\ln\Gamma(\alpha_n+m+1),
\qquad m=0,\dots,n.
$$


This is exactly what the loop over $m$ in `high_precision_S1_0n_log_space` and `high_precision_S2_0n` does: it builds a list of `Decimal` values `log_c_values` (or `log_c_vals`) holding $\ln c_m$. By design,
$$
c_m > 0 \quad\Rightarrow\quad \text{sign}(c_m) = +1,
$$
which is stored separately as `c_signs`.


## 4. Log-Space Construction of the Overlap Terms


### 4.1. $S_1 = \langle\psi_0|Q|\psi_n\rangle$ (`high_precision_S1_0n`)


The function `high_precision_S1_0n` (which calls `high_precision_S1_0n_log_space` internally) implements the high-precision evaluation of the linear overlap $S_1$ for the $0\to n$ transition. In log-space form, the integral is reduced to a finite sum over $m$,
$$
S_1^{(0,n)}
=
\sum_{m=0}^n
(-1)^m
\frac{c_m}{m!}\, I_m,
$$
where
$$
I_m
=
\int_0^\infty
y^{\beta-1}\,e^{-y}
\Bigl(\ln\tfrac{y}{2\lambda}\Bigr)
\,dy
\quad\text{with}\quad
\beta = 2\lambda - 5 + m.
$$


Analytically, this integral can be expressed in terms of $\Gamma$ and $\psi$:
$$
I_m
=
\Gamma(\beta)\,
\Bigl[\psi(\beta) - \ln(2\lambda)\Bigr].
$$


The code evaluates this in log-magnitude and sign form:


1. Compute $\ln\Gamma(\beta)$ via `high_precision_log_gamma(beta)`.
2. Compute $\psi(\beta)$ via `high_precision_digamma(beta)`.
3. Compute $\ln(2\lambda)$ as `log_2lambda = (2*lambda_dec).ln()`.
4. Form the difference
 $$
 \Delta
 =
 \psi(\beta) - \ln(2\lambda).
 $$
5. Determine the sign and log-magnitude:
 $$
 \operatorname{sign}(I_m) = \operatorname{sign}(\Delta),\qquad
 \ln|I_m| = \ln\Gamma(\beta) + \ln|\Delta|.
 $$


The factorial $m!$ is likewise handled via high-precision log-gamma:
$$
\ln m! = \ln\Gamma(m+1).
$$


Thus, each term in the sum is represented purely through logs:
$$
 ext{term}_m
=
(-1)^m\,
c_m\,I_m / m!,
$$
$$
\ln|\text{term}_m|
=
\ln c_m
+
\ln|I_m|
-
\ln m!,
$$
$$
\operatorname{sign}(\text{term}_m)
=
(-1)^m \cdot \operatorname{sign}(c_m) \cdot \operatorname{sign}(I_m).
$$


The `high_precision_S1_0n_log_space` routine builds:


- `log_terms[m] = \ln|\text{term}_m|$ (as `Decimal`),
- `term_signs[m] = \operatorname{sign}(\text{term}_m)` (as `int` $\pm1$),


and **never** directly forms $c_m$, $I_m$, or $m!$ in floating point.


After the sum $\sum_m \text{term}_m$ is computed (see §5), the overall normalization factor is applied in log space:
$$
S_1^{(0,n)} =
N_0 N_n\,a^{-2}\,\sum_{m=0}^n \text{term}_m,
$$
with
$$
\ln|N_0 N_n a^{-2}|
=
\ln N_0
+
\ln N_n
-
2\ln a.
$$


Only at the very end is this combined log prefactor and log sum converted back to a real number via exponentiation.


### 4.2. $S_2 = \langle\psi_0|Q^2|\psi_n\rangle$ (`high_precision_S2_0n`)


The high-precision quadratic-overlap routine `high_precision_S2_0n` mirrors the structure of `high_precision_S1_0n` but uses an $I_m^{(2)}$ integral that contains both $\psi$ and $\psi_1$:
$$
S_2^{(0,n)} =
\sum_{m=0}^n
(-1)^m
\frac{c_m}{m!}\, I_m^{(2)},
$$
with
$$
I_m^{(2)}
=
\Gamma(\beta)
\bigl[
\psi(\beta)^2
+
\psi_1(\beta)
-
2\ln(2\lambda)\,\psi(\beta)
+
\ln(2\lambda)^2
\bigr].
$$


The code evaluates the bracket
$$
B(\beta)
=
\psi(\beta)^2 + \psi_1(\beta)
-
2\ln(2\lambda)\,\psi(\beta)
+
\ln(2\lambda)^2
$$
in high precision, then decomposes $I_m^{(2)}$ into a sign and a log-magnitude:
$$
\operatorname{sign}(I_m^{(2)}) = \operatorname{sign}(B(\beta)),\qquad
\ln|I_m^{(2)}| = \ln\Gamma(\beta) + \ln|B(\beta)|.
$$


As for $S_1$, the term magnitudes and signs are stored as `log_terms` and `term_signs`, and the overall prefactor
$$
N_0 N_n a^{-3}
$$
is applied in log space using $\ln N_v$ from `high_precision_log_N_v` and $\ln a$ from `Decimal.ln`.


## 5. Alternating Sum in Log Space


The summation of the series
$$
\sum_{m=0}^n \text{term}_m
$$
is carried out by `high_precision_alternating_sum_from_logs(log_terms, term_signs)`:


1. Each pair $(\ln|t_m|, \operatorname{sign}(t_m))$ is converted back to a `Decimal` value:
 $$
 t_m = \operatorname{sign}(t_m)\,\exp\bigl(\ln|t_m|\bigr).
 $$
 Extremely small terms with $\ln|t_m|$ below a cutoff (e.g. less than about $-1000$) are discarded as numerically irrelevant.


2. The terms $t_m$ are then summed purely in `Decimal` arithmetic:
 $$
 S = \sum_{m=0}^n t_m
 $$
 with 200 (or more) digits of precision, which preserves dozens of significant digits even when the partial sums suffer catastrophic cancellation.


This strategy ensures that while individual terms can be as large as
$$
|t_m| \sim 10^{32000},
$$
the final sums
$$
S_1^{(0,n)},\; S_2^{(0,n)}
$$
can be accurately computed even when they are in the range $10^{-50}$–$10^{-200}$ in SI units.


## 6. Effective Precision and Practical Settings


Internally, `high_precision_arithmetic.py` uses Python’s `decimal` context to set a high, but tractable, working precision
$$
 exttt{getcontext().prec} = \mathcal{O}(10^2\text{–}10^3),
$$
and then:


- All sensitive operations (log-gamma, digamma, trigamma, logarithms) are done in `Decimal`.
- All large products and powers are expressed via $\ln(\cdot)$ and only exponentiated once at the very end.
- Alternating sums are accumulated directly in high precision to avoid loss of significance.


This is the stabilization backbone in the `stabilization` subdirectory that makes the Morse transition dipole calculations numerically feasible for very high overtones.


---


# Part C: The Morse Model


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
$$
\tilde\nu_e x_e = \frac{\tilde\nu_e^2}{4 D_e} \qquad \text{(with } D_e \text{ in cm}^{-1}\text{)}
$$




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
Use the general formula for IR (or NIR) vibrational transitions:
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


- Since IR overtones usually have extremely small molar exticntion coefficients in general, and NIR overtones have even smaller values, the results are scaled by a factor of $$\10^64\$$. This constant scalar multiple at the end of all calculations ensures that the results are scientifically usable, representing a *relative* molar absorptivity value.


---


## How To Use the CLI


The Morse solver provides two usage modes: **batch mode** (all parameters at once) and **interactive mode** (step-by-step prompts).


### Batch Mode (All Parameters at Once)


Provide all required parameters in a single command for automated workflows:


```bash
python3 \
 --m1 12.011 \
 --m2 1.008 \
 --fundamental 2900.0 \
 --observed 8700.0 \
 --overtone 3 \
 --coords "C 0.0 0.0 0.0\nH 1.1 0.0 0.0" \
 --specified-spin 0 \
 --bond "0,1" \
 --delta 0.005 \
 --basis aug-cc-pVTZ \
 --fwhm 75.0
```


**Parameters:**
- `--m1`, `--m2`: Atomic masses (amu) for elements A and B
- `--fundamental`: Fundamental frequency (cm⁻¹)
- `--observed`: Observed overtone frequency (cm⁻¹)
- `--overtone`: Integer overtone number (n for 0→n transition)
- `--coords`: Molecular geometry in XYZ format (quoted multiline string)
- `--specified-spin`: Spin multiplicity (0 for singlet, 1 for doublet, etc.)
- `--bond`: Bond atom indices as "i,j" (0-based)
- `--delta`: Finite difference displacement (Angstrom, default: 0.005)
- `--basis`: Quantum chemistry basis set (default: aug-cc-pVTZ). Can be overridden with higher quality sets like aug-cc-pVQZ or  aug-cc-pV5Z for maximum accuracy.
- `--fwhm`: Line width for peak extinction (cm⁻¹, default: 75.0)


### Interactive Mode (Step-by-Step)


Run without parameters for guided input:


```bash
python3 run_morse_solver
```


The CLI will prompt for each parameter:


```
Morse Solver for IR (or NIR) Overtone Extinction Coefficients


Enter atomic mass of element A (amu): 12.011
Enter atomic mass of element B (amu): 1.008
Enter fundamental frequency (cm⁻¹): 2900.0
Enter observed frequency (cm⁻¹): 8700.0
Enter overtone number (integer): 3


📐 Molecular Geometry Input
Choose input method:
1. Type coordinates directly
2. Load from file
Selection: 1


Enter molecular coordinates (Element x y z format, blank line to finish):
C 0.0 0.0 0.0
H 1.1 0.0 0.0
[blank line]


Enter spin multiplicity: 0
Enter bond atom indices (i,j format): 0,1


Advanced Options (press Enter for defaults)
Finite difference step size (Å) [0.005]:
Basis set [aug-cc-pVTZ]: aug-cc-pVQZ
```


### Geometry Input Options


#### Option 1: Direct molecular information input via the interactive CLI
```bash
# Example of how the inetractive prompt handles:
Enter molecular coordinates (Element x y z format):
C 0.000000 0.000000 0.000000
H 1.100000 0.000000 0.000000
O -1.200000 0.000000 0.000000
H -1.800000 0.800000 0.000000
[blank line to finish]
```


#### Option 2: File input (+ other necesary molecular parameters in interactive mode)
```bash
# Create coordinates file (e.g., molecule.xyz):
cat > molecule.xyz << EOF
C 0.000000 0.000000 0.000000
H 1.100000 0.000000 0.000000
O -1.200000 0.000000 0.000000
H -1.800000 0.800000 0.000000
EOF


# Then use in batch mode:
python3 run_morse_model.py compute --coords molecule.xyz [other parameters...]


```


### Option 3: Direct input of all coordinates and parameters (Advanced)


#### Examples:


#### C-H Stretch in Methane:
```bash
python3 run_morse_model.py compute \
 --m1 12.011 --m2 1.008 \
 --fundamental 2917 --observed 8750 --overtone 3 \
 --coords "C 0.0 0.0 0.0\nH 1.09 0.0 0.0\nH -0.36 1.03 0.0\nH -0.36 -0.51 0.89\nH -0.36 -0.51 -0.89" \
 --specified-spin 0 --bond "0,1"
```


#### O-H Stretch in Water:
```bash
python3 run_morse_model.py compute \
 --m1 15.999 --m2 1.008 \
 --fundamental 3657 --observed 10935 --overtone 3 \
 --coords "O 0.0 0.0 0.0\nH 0.757 0.587 0.0\nH -0.757 0.587 0.0" \
 --specified-spin 0 --bond "0,1"
```


#### Dual Bond System:
For molecules with symmetric stretching modes (the semicolon between bond axes is CRUCIAL):
```bash
python3 run_morse_model.py compute \
 --dual-bonds "(0,2);(1,2)" \
 --m1 12.011 --m2 15.999 \
 [other parameters...]
```


#### Fundamental peak ε values can also be determined with this software, if the overtone order is set to 0 and the observed frequency input is the same as the fundamental frequency:


```bash
python3 run_morse_model.py compute \
 --m1 15.999 --m2 1.008 \
 --fundamental 3657 --observed 3657 --overtone 0 \
 --coords "O 0.0 0.0 0.0\nH 0.757 0.587 0.0\nH -0.757 0.587 0.0" \
 --specified-spin 0  --bond "0,1"
```




#### *NOTE: ALL of the above examples are just arbitrary numbers, not actual valid data, including the hypothetical methane example. DO NOT USE THOSE DEMONSTRATION NUMBERS IN ACTUAL SCIENTIFIC RESEARCH!*


### What Is Basis Set Selection?


The `--basis` flag allows you to control the quantum chemistry basis set used for SCF calculations:


**Default (Recommended):** `aug-cc-pVTZ`
- High accuracy for most organic molecules
- Good balance of precision and computational cost
- Suitable for production calculations


**Higher Accuracy:** `aug-cc-pVQZ`
- Maximum precision for critical applications
- Significantly longer computation time
- Recommended for benchmarking or when highest accuracy is needed


**Faster Computation:** `cc-pVDZ`
- Reduced accuracy but much faster
- Useful for testing, debugging, or large systems
- Not recommended for final results


**Example with custom basis set:**
```bash
python3 run_morse_model.py compute --basis aug-cc-pVQZ [other parameters...]
```


### Output


The CLI provides detailed output including:
- SCF geometry optimization results
- Computed dipole derivatives
- Morse model parameters
- Final molar extinction coefficient


Example output:
```
Results Summary
================
Morse Parameters:
 λ (lambda): 25.43
 D_e: 4.52 eV
 a: 2.14 × 10¹⁰ m⁻¹


Transition Properties:
 S₁ overlap: 1.23 × 10⁻⁶
 S₂ overlap: 8.95 × 10⁻¹³
 Transition dipole: 2.34 × 10⁻³² C·m


Final Result:
 Peak molar extinction: 1.24 × 10⁻⁴² M⁻¹cm⁻¹ (× 10⁶⁴ scaling factor)
```