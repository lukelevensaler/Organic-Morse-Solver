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
