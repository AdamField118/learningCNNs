"""
Mathematical foundation of convolution operations.

This module implements convolution from first principles to build
deep understanding of the mathematical operations underlying CNNs.
"""

import numpy as np
import matplotlib.pyplot as plt

def convolution_1d_simple(signal, kernel):
    """
    Implement 1D convolution from mathematical definition.
    
    Mathematical definition: (f * g)[n] = sum over m of f[m] * g[n - m]
    
    Args:
        signal: 1D numpy array (e.g., [1, 2, 3, 4, 5])
        kernel: 1D numpy array (e.g., [0.5, 1, 0.5])
        
    Returns:
        1D numpy array with convolution result
    """
    signal_len = len(signal)
    kernel_len = len(kernel)
    output_len = signal_len + kernel_len - 1  # Full convolution size
    
    # Initialize output array
    output = np.zeros(output_len)
    
    for n in range(output_len):
        for m in range(signal_len):
            # Check if kernel index (n-m) is valid
            kernel_idx = n - m
            if 0 <= kernel_idx < kernel_len:
                output[n] += signal[m] * kernel[kernel_idx]
    
    return output

def test_simple_convolution():
    """Test the 1D convolution with a simple example."""
    # Simple test case
    signal = np.array([1, 2, 3, 4, 5])
    kernel = np.array([0.5, 1, 0.5])  # Simple smoothing kernel
    
    result = convolution_1d_simple(signal, kernel)
    print(f"Signal: {signal}")
    print(f"Kernel: {kernel}")
    print(f"Result: {result}")
    
    # Compare with NumPy's implementation to verify correctness
    numpy_result = np.convolve(signal, kernel, mode='full')
    print(f"NumPy result: {numpy_result}")
    print(f"Match: {np.allclose(result, numpy_result)}")

if __name__ == "__main__":
    test_simple_convolution()