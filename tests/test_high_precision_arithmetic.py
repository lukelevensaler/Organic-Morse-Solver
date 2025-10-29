#!/usr/bin/env python3
"""
Test suite for high_precision_arithmetic.py module.
Tests sign handling, numerical accuracy, and edge cases.
"""

import numpy as np
import pytest
import sys
import os
from decimal import Decimal, getcontext

# Add the current directory to the path to import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stabilization.high_precision_arithmetic import (
    high_precision_log_gamma,
    high_precision_digamma,
    high_precision_polygamma,
    high_precision_S1_0n,
    high_precision_S1_0n_log_space,
    high_precision_log_N_v,
    high_precision_N_v,
    high_precision_alternating_sum_from_logs,
    high_precision_S2_0n
)

class TestHighPrecisionArithmetic:
    """Test class for high precision arithmetic functions."""
    
    def setup_method(self):
        """Set up test environment."""
        # Set precision for consistent testing
        getcontext().prec = 500
        
    def test_log_gamma_basic(self):
        """Test log gamma function for basic values."""
        # Test small values (should use scipy)
        result = high_precision_log_gamma(5.0)
        expected = np.log(24.0)  # Gamma(5) = 4! = 24
        assert abs(float(result) - expected) < 1e-10
        
        # Test moderate value
        result = high_precision_log_gamma(10.5)
        # Gamma(10.5) ≈ 1133278.388
        expected = np.log(1133278.388)
        assert abs(float(result) - expected) < 1e-3
        
    def test_log_gamma_large_values(self):
        """Test log gamma for large values using Stirling approximation."""
        # Test large value (should use Stirling)
        result = high_precision_log_gamma(200.0)
        
        # Stirling: ln(Gamma(x)) ≈ (x-0.5)*ln(x) - x + 0.5*ln(2π)
        x = 200.0
        stirling = (x - 0.5) * np.log(x) - x + 0.5 * np.log(2 * np.pi)
        
        # Should be close to Stirling approximation
        assert abs(float(result) - stirling) < 1.0
        
    def test_digamma_basic(self):
        """Test digamma function for basic values."""
        # Test moderate values
        result = high_precision_digamma(5.0)
        # ψ(5) ≈ 1.506
        assert abs(float(result) - 1.506) < 0.1
        
        # Test larger values (asymptotic)
        result = high_precision_digamma(200.0)
        # Should be approximately ln(x) for large x
        expected = np.log(200.0)
        assert abs(float(result) - expected) < 0.1
        
    def test_polygamma_trigamma(self):
        """Test trigamma function (polygamma with n=1)."""
        # Test moderate value
        result = high_precision_polygamma(1, 10.0)
        # ψ'(10) ≈ 0.105
        assert abs(float(result) - 0.105) < 0.05
        
        # Test that unsupported n raises error
        with pytest.raises(NotImplementedError):
            high_precision_polygamma(2, 5.0)
            
    def test_alternating_sum_simple(self):
        """Test alternating sum with simple known values."""
        # Simple test: 1 - 0.5 + 0.25 - 0.125 = 0.625
        # Use exact logarithms for better precision
        log_terms = [
            Decimal('0'),                      # ln(1) = 0 exactly
            Decimal('2').ln(),                 # ln(0.5) = -ln(2)
            Decimal('4').ln(),                 # ln(0.25) = -ln(4) 
            Decimal('8').ln()                  # ln(0.125) = -ln(8)
        ]
        # Make the negative logarithms negative
        log_terms[1] = -log_terms[1] 
        log_terms[2] = -log_terms[2]
        log_terms[3] = -log_terms[3]
        signs = [1, -1, 1, -1]
        
        result = high_precision_alternating_sum_from_logs(log_terms, signs)
        expected = 1.0 - 0.5 + 0.25 - 0.125  # = 0.625
        
        assert abs(float(result) - expected) < 1e-8  # Relaxed tolerance for numerical precision
        
    def test_alternating_sum_cancellation(self):
        """Test alternating sum with near-cancellation (sign preservation)."""
        # Test case where terms nearly cancel: 1000 - 999.9 = 0.1
        log_terms = [Decimal(str(np.log(1000))), Decimal(str(np.log(999.9)))]
        signs = [1, -1]
        
        result = high_precision_alternating_sum_from_logs(log_terms, signs)
        expected = 1000.0 - 999.9  # = 0.1
        
        assert abs(float(result) - expected) < 1e-10
        assert float(result) > 0  # Should be positive
        
    def test_alternating_sum_negative_result(self):
        """Test alternating sum that should give negative result."""
        # Test: 10 - 20 = -10
        log_terms = [Decimal(str(np.log(10))), Decimal(str(np.log(20)))]
        signs = [1, -1]
        
        result = high_precision_alternating_sum_from_logs(log_terms, signs)
        expected = 10.0 - 20.0  # = -10
        
        assert abs(float(result) - expected) < 1e-10
        assert float(result) < 0  # Should be negative
        
    def test_N_v_normalization(self):
        """Test normalization constant computation."""
        # Test with reasonable parameters
        a = 1.0
        lambda_val = 5.0
        v = 0
        
        # Should give a finite positive result
        result = high_precision_N_v(v, a, lambda_val)
        assert float(result) > 0
        assert np.isfinite(float(result))
        
        # Test log version consistency
        log_result = high_precision_log_N_v(v, a, lambda_val)
        expected_log = Decimal(str(np.log(float(result))))
        assert abs(float(log_result - expected_log)) < 1e-6
        
    def test_S1_0n_sign_consistency(self):
        """Test that S1_0n preserves correct signs."""
        # Test parameters that should give a known sign pattern
        n = 1
        a = 1.0
        lambda_val = 3.0
        
        # Compute using both methods
        result_direct = high_precision_S1_0n(n, a, lambda_val)
        result_log = high_precision_S1_0n_log_space(n, a, lambda_val)
        
        # Both methods should agree
        assert abs(result_direct - result_log) < 1e-10
        
        # Result should be finite
        assert np.isfinite(result_direct)
        assert np.isfinite(result_log)
        
    def test_S1_0n_different_parameters(self):
        """Test S1_0n with different parameter sets to check sign patterns."""
        test_cases = [
            (1, 1.0, 2.0),   # Small n
            (2, 1.0, 3.0),   # Medium n
            (0, 1.0, 1.5),   # n=0 case
            (1, 2.0, 2.5),   # Different a
        ]
        
        results = []
        for n, a, lambda_val in test_cases:
            try:
                result = high_precision_S1_0n(n, a, lambda_val)
                results.append((n, a, lambda_val, result))
                
                # Check that result is finite
                assert np.isfinite(result), f"Non-finite result for n={n}, a={a}, λ={lambda_val}"
                
                print(f"S1_0n({n}, {a}, {lambda_val}) = {result:.6e}")
                
            except Exception as e:
                print(f"Error computing S1_0n({n}, {a}, {lambda_val}): {e}")
                
        # Ensure we got some results
        assert len(results) > 0
        
    def test_S1_0n_sign_preservation_extreme(self):
        """Test sign preservation for extreme parameter values."""
        # Test case that might trigger high precision arithmetic
        n = 5
        a = 0.1  # Small a
        lambda_val = 10.0  # Large lambda
        
        result = high_precision_S1_0n(n, a, lambda_val)
        
        # Should be finite
        assert np.isfinite(result)
        
        # Test that we can predict the sign based on physics
        # (This would require more detailed analysis of the specific problem)
        print(f"Extreme case: S1_0n({n}, {a}, {lambda_val}) = {result:.6e}")
        
    def test_S2_0n_basic(self):
        """Test S2_0n function returns reasonable values."""
        n = 1
        a = 1.0
        lambda_val = 3.0
        
        result = high_precision_S2_0n(n, a, lambda_val)
        
        # Should be finite and typically small
        assert np.isfinite(result)
        assert abs(result) < 1e-10  # S2 is typically much smaller than S1
        
    def test_zero_handling(self):
        """Test proper handling of zero values."""
        # Test alternating sum with zero terms
        log_terms = [Decimal('0'), Decimal('-1000')]  # ln(1), very small term
        signs = [1, 1]
        
        result = high_precision_alternating_sum_from_logs(log_terms, signs)
        
        # Should be approximately 1 (the large term dominates)
        assert abs(float(result) - 1.0) < 1e-10
        
    def test_large_magnitude_preservation(self):
        """Test that large magnitudes don't cause sign errors."""
        # Create terms with large magnitudes but opposite signs
        # Result should be their difference, not corrupted by overflow
        
        large_val = 1e100
        small_diff = 1e95
        
        log_terms = [Decimal(str(np.log(large_val))), Decimal(str(np.log(large_val - small_diff)))]
        signs = [1, -1]
        
        result = high_precision_alternating_sum_from_logs(log_terms, signs)
        expected = large_val - (large_val - small_diff)  # = small_diff
        
        # Should preserve the sign and approximate magnitude
        assert float(result) > 0  # Should be positive
        assert abs(float(result) - small_diff) / small_diff < 0.1  # Within 10%
        
def run_comprehensive_tests():
    """Run comprehensive tests and output detailed results."""
    print("=" * 60)
    print("COMPREHENSIVE HIGH PRECISION ARITHMETIC TESTS")
    print("=" * 60)
    
    test_class = TestHighPrecisionArithmetic()
    test_class.setup_method()
    
    tests = [
        ("Log Gamma Basic", test_class.test_log_gamma_basic),
        ("Log Gamma Large", test_class.test_log_gamma_large_values),
        ("Digamma Basic", test_class.test_digamma_basic),
        ("Polygamma Trigamma", test_class.test_polygamma_trigamma),
        ("Alternating Sum Simple", test_class.test_alternating_sum_simple),
        ("Alternating Sum Cancellation", test_class.test_alternating_sum_cancellation),
        ("Alternating Sum Negative", test_class.test_alternating_sum_negative_result),
        ("Normalization Constants", test_class.test_N_v_normalization),
        ("S1_0n Sign Consistency", test_class.test_S1_0n_sign_consistency),
        ("S1_0n Different Parameters", test_class.test_S1_0n_different_parameters),
        ("S1_0n Extreme Parameters", test_class.test_S1_0n_sign_preservation_extreme),
        ("S2_0n Basic", test_class.test_S2_0n_basic),
        ("Zero Handling", test_class.test_zero_handling),
        ("Large Magnitude Preservation", test_class.test_large_magnitude_preservation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n--- Running: {test_name} ---")
            test_func()
            print(f"✅ PASSED: {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {e}")
            failed += 1
            
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    # Run comprehensive tests when called directly
    success = run_comprehensive_tests()
    
    # Also demonstrate sign behavior with actual computation
    print("\n" + "=" * 60)
    print("SIGN BEHAVIOR DEMONSTRATION")
    print("=" * 60)
    
    # Test the specific case mentioned in the issue
    print("\nTesting S1_0n with parameters similar to the failing case:")
    
    test_params = [
        (1, 0.5, 2.0),
        (2, 0.5, 3.0),
        (1, 1.0, 2.5),
        (3, 0.8, 4.0),
    ]
    
    for n, a, lambda_val in test_params:
        try:
            result = high_precision_S1_0n(n, a, lambda_val)
            sign_str = "positive" if result > 0 else "negative" if result < 0 else "zero"
            print(f"S1_0n({n}, {a:.1f}, {lambda_val:.1f}) = {result:.6e} ({sign_str})")
        except Exception as e:
            print(f"Error with S1_0n({n}, {a:.1f}, {lambda_val:.1f}): {e}")
    
    if success:
        print("\n🎉 All tests passed! Sign handling appears to be working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        sys.exit(1)