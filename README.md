
# Welcome to the Organic Morse Solver, a quantum computation for deriving molar extinction coefficients for IR/NIR peaks.

## This CLI performs the following quantum chemistry and Morse Model-based anharmonicty calulations to determine the molar extinction coefficent ε in $$\{M·cm}^{-1}\$$ for any organic molecule's IR (or NIR) overtone peak. Its algorithm can even compute ε values for fundamnetal peaks, at full anharmonic accuracy.


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

# High-Precision Arithmetic Note

The Morse overtone calculations involve extreme numerical challenges due to the alternating series nature of the overlap integrals. Individual terms in the sum can reach magnitudes as large as $e^{74691} \approx 10^{32000}$, but the alternating series produces final results in the range $10^{-50}$ to $10^{-200}$ C·m.

**Key Numerical Challenges:**
- **Catastrophic Cancellation**: Large positive and negative terms nearly cancel
- **Overflow Risk**: Individual terms exceed standard floating-point limits
- **Precision Loss**: Standard double precision (16 digits) is insufficient

**High-Precision Solution:**
The solver uses Python's `decimal` module with **1,000 decimal places of precision** to:
1. Compute each term using logarithmic arithmetic: $\ln|I_m| + \ln|c_m| - \ln(m!)$
2. Convert to high-precision `Decimal` objects before exponentiation
3. Apply alternating signs and sum with full precision
4. Handle Gamma, digamma, and trigamma functions for large arguments using asymptotic expansions

**Example Numerical Scale:**
```
Term 0: magnitude ≈ 10^32000
Term 1: magnitude ≈ -10^32000  
Term 2: magnitude ≈ 10^31995
...
Final Sum: ≈ 10^-85 C·m (after cancellation)
```

This technique enables accurate calculation of transition dipole moments for high overtones that would be impossible with standard numerical methods.

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
