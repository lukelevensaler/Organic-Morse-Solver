# We only test the modules that are called before dipole moment derivative computation and the subsequent morse anharmonicity calculations. 

* These will fail loudly if there is any issue with them. 
* Since these tested modules are somewhat buried at the bottom of the call stack, it is important to test them induvally to make sure they work.

## We use a sample H2O molecule coordinates for test tests given by:

''' python
mol_str = """
O 0.0 0.0 0.0
H 0.757 0.587 0.0
H -0.757 0.587 0.0
"""
'''

## Notes:

* STO-3G is used as the CCSD/CCSD(T) string in these tests because it is extremely minimal, just like the other parameters in these tests (such as the low upper limit for algorithmic steps)

* In running actual analysis, the actual main algotithm has very high-power hardcoded parameters for CCSD/CCSD(T) computations, so STO-3G will cause serious numerical problems ranging from weird zeroing, to convergence issues, to ridiculously predicting quantum singularities due to overflow/underflow errors.

    * This is because calculating integrals with STOs is computationally difficult [1].
    * As such, you **MUST** use Correlation-consistent basis set strings in the format cc-pVNZ where N = D,T,Q,5,6,... (D = double, T = triple, etc.). The 'cc-p', stands for 'correlation-consistent polarized' and the 'V' indicates that only basis sets for the valence orbitals are of multiple-zeta quality [1].
    * These Correlation-consistent strings are called Dunning basis sets, without augmentation (the strings starting with "aug-"), they only account for valence electrons. However, augmented strings can describe core electron correlations.

* If you want a quick computation basis string, try cc-pVDZ (no computationally costly molecular augmentation like with strings starting with "aug-")

### References:

[1] [text](https://en.wikipedia.org/wiki/Basis_set_(chemistry))
[2] [text](https://pubs.aip.org/aip/jcp/article-abstract/90/2/1007/91329/Gaussian-basis-sets-for-use-in-correlated?redirectedFrom=fulltext)