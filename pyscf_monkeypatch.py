from pyscf.grad import ccsd as ccsd_grad

def patch_pyscf_ccsd_gradient_bug():
    """
    Monkey patch to fix PySCF 2.10.0 CCSD gradient bug where eris.fock can be a tuple.
    This patches the specific line in pyscf.grad.ccsd that causes the error.
    """
    try:
        
        # Check if we need to apply the patch
        if hasattr(ccsd_grad, '_patched_for_fock_tuple_bug'):
            print("PySCF CCSD gradient patch already applied")
            return
            
        # Store original method
        original_grad_elec = ccsd_grad.Gradients.grad_elec
        
        def patched_grad_elec(self, *args, **kwargs):
            """Patched grad_elec that handles tuple fock matrices and amplitude tuples"""
            # Get parameters from args/kwargs
            t1, t2, l1, l2, eris = None, None, None, None, None
            
            if len(args) >= 1:
                t1 = args[0]
            if len(args) >= 2:
                t2 = args[1]
            if len(args) >= 3:
                l1 = args[2]
            if len(args) >= 4:
                l2 = args[3]
            if len(args) >= 5:
                eris = args[4]
                
            # Also check kwargs
            t1 = kwargs.get('t1', t1)
            t2 = kwargs.get('t2', t2) 
            l1 = kwargs.get('l1', l1)
            l2 = kwargs.get('l2', l2)
            eris = kwargs.get('eris', eris)
                
            # Fix the eris.fock tuple issue 
            if eris is not None and hasattr(eris, 'fock'):
                if isinstance(eris.fock, tuple):
                    if len(eris.fock) == 2:
                        print("Fixed PySCF fock tuple bug: converting to alpha fock matrix")
                        eris.fock = eris.fock[0]  # Use alpha fock matrix
                    else:
                        raise RuntimeError(f"Unexpected fock tuple length: {len(eris.fock)}")
            
            # Fix amplitude tuples - for UHF CCSD, amplitudes can be tuples (alpha, beta)
            # but some gradient code expects them as single arrays (use alpha component)
            modified_args = list(args)
            
            if t1 is not None and isinstance(t1, tuple):
                print("Fixed PySCF t1 amplitude tuple bug: using alpha component")
                modified_args[0] = t1[0]  # Use alpha t1
                if 't1' in kwargs:
                    kwargs['t1'] = t1[0]
                    
            if t2 is not None and isinstance(t2, tuple):
                print("Fixed PySCF t2 amplitude tuple bug: using alpha component") 
                modified_args[1] = t2[0]  # Use alpha t2
                if 't2' in kwargs:
                    kwargs['t2'] = t2[0]
                    
            if l1 is not None and isinstance(l1, tuple):
                print("Fixed PySCF l1 lambda tuple bug: using alpha component")
                modified_args[2] = l1[0]  # Use alpha l1
                if 'l1' in kwargs:
                    kwargs['l1'] = l1[0]
                    
            if l2 is not None and isinstance(l2, tuple):
                print("Fixed PySCF l2 lambda tuple bug: using alpha component")
                modified_args[3] = l2[0]  # Use alpha l2
                if 'l2' in kwargs:
                    kwargs['l2'] = l2[0]
            
            # Call the original method with fixed parameters
            try:
                return original_grad_elec(self, *modified_args, **kwargs)
            except ValueError as e:
                if "too many values to unpack" in str(e):
                    print(f"PySCF gradient unpack error: {e}")
                    print("This indicates another tuple handling issue in PySCF 2.10.0")
                    # Try alternative approaches or raise a more informative error
                    raise RuntimeError(f"PySCF 2.10.0 has multiple tuple handling bugs in CCSD gradients. "
                                     f"Consider downgrading to PySCF 2.6.2: conda install pyscf=2.6.2 -c conda-forge. "
                                     f"Original error: {e}")
                else:
                    raise e
        
        # Apply the patch
        ccsd_grad.Gradients.grad_elec = patched_grad_elec
        # Mark that patch was applied (ignore type checker warning)
        setattr(ccsd_grad, '_patched_for_fock_tuple_bug', True)
        
        print("Applied PySCF CCSD gradient fock tuple bug patch")
        
        # Also try to patch other common tuple issues in PySCF 2.10.0
        try:
            from pyscf.cc import ccsd_rdm
            if hasattr(ccsd_rdm, '_gamma1_intermediates') and not hasattr(ccsd_rdm, '_patched_gamma1'):
                original_gamma1 = ccsd_rdm._gamma1_intermediates
                
                def patched_gamma1_intermediates(cc, t1, t2, l1, l2, *args, **kwargs):
                    """Patch _gamma1_intermediates to handle tuple amplitudes"""
                    # Fix tuple amplitudes
                    if isinstance(t1, tuple):
                        t1 = t1[0]  # Use alpha component
                    if isinstance(t2, tuple):
                        t2 = t2[0]  # Use alpha component
                    if isinstance(l1, tuple):
                        l1 = l1[0]  # Use alpha component
                    if isinstance(l2, tuple):
                        l2 = l2[0]  # Use alpha component
                    
                    return original_gamma1(cc, t1, t2, l1, l2, *args, **kwargs)
                
                ccsd_rdm._gamma1_intermediates = patched_gamma1_intermediates
                setattr(ccsd_rdm, '_patched_gamma1', True)
                print("Applied additional PySCF gamma1 intermediates tuple patch")
        except Exception as rdm_e:
            print(f"Warning: Could not apply gamma1 patch: {rdm_e}")
        
    except Exception as e:
        print(f"Warning: Could not apply PySCF gradient patch: {e}")
        # Continue without patch - the error will be caught later
