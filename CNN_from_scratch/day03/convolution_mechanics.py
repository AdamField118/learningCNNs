"""
Deep dive into convolution mechanics and variants.

This module explores stride, padding, dilation, and output dimension calculations
to build complete understanding of how convolution parameters affect results.
"""

import numpy as np
import sys
import os

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Centralized imports
from utils.test_data import create_test_pattern
from utils.visualization import plot_convolution_mechanics
from utils.mathematical_analysis import (
    calculate_conv_output_size, 
    analyze_computational_cost,
    print_dimension_analysis,
    print_cost_analysis
)
from day02.edge_detection_basics import manual_convolution_2d
from kernels.edge_detection_kernels import sobel_x_kernel

def manual_convolution_2d_extended(image, kernel, stride=1, padding=0, dilation=1):
    """
    Extended convolution with stride, padding, and dilation support.
    
    Args:
        image: 2D numpy array representing the input image
        kernel: 2D numpy array representing convolution kernel
        stride: Step size for convolution operation (default: 1)
        padding: Padding to add around image borders (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        
    Returns:
        Convolved image as 2D numpy array
        
    The dilation parameter controls the spacing between kernel elements:
    - dilation=1: Standard convolution (no spacing)
    - dilation=2: Insert one zero between each kernel element
    - dilation=3: Insert two zeros between each kernel element, etc.
    """
    # Add padding if specified
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    
    # Apply dilation to kernel if needed
    if dilation > 1:
        kernel = apply_dilation(kernel, dilation)
    
    # Get dimensions after padding and dilation
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    # Calculate output dimensions
    # Formula: (input_size - kernel_size) / stride + 1
    output_h = (image_h - kernel_h) // stride + 1
    output_w = (image_w - kernel_w) // stride + 1
    
    # Initialize output array
    output = np.zeros((output_h, output_w))
    
    # Implement convolution with stride support
    # Outer loops: iterate over each output position
    for i in range(output_h):
        for j in range(output_w):
            # Calculate starting position in input image
            # This is where stride comes into play - we jump by 'stride' pixels
            start_i = i * stride
            start_j = j * stride
            
            # Inner loops: iterate over kernel positions
            # We convolve the kernel with the current window of the image
            for ki in range(kernel_h):
                for kj in range(kernel_w):
                    # Calculate corresponding position in input image
                    img_i = start_i + ki
                    img_j = start_j + kj
                    
                    # Element-wise multiply and accumulate
                    # This is the core of convolution: sum of products
                    output[i, j] += image[img_i, img_j] * kernel[ki, kj]
    
    return output

def manual_convolution_2d_with_padding_types(image, kernel, stride=1, padding=0, padding_mode='constant'):
    """Extended convolution with different padding modes."""
    if padding > 0:
        if padding_mode == 'constant':
            padded_image = np.pad(image, padding, mode='constant', constant_values=0)
        elif padding_mode == 'reflect':
            padded_image = np.pad(image, padding, mode='reflect')
        elif padding_mode == 'edge':
            padded_image = np.pad(image, padding, mode='edge')
        elif padding_mode == 'wrap':
            padded_image = np.pad(image, padding, mode='wrap')
        else:
            raise ValueError(f"Unknown padding mode: {padding_mode}")
    else:
        padded_image = image
    
    image_h, image_w = padded_image.shape
    kernel_h, kernel_w = kernel.shape
    
    output_h = (image_h - kernel_h) // stride + 1
    output_w = (image_w - kernel_w) // stride + 1
    
    output = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            start_i = i * stride
            start_j = j * stride
            
            for ki in range(kernel_h):
                for kj in range(kernel_w):
                    img_i = start_i + ki
                    img_j = start_j + kj
                    output[i, j] += padded_image[img_i, img_j] * kernel[ki, kj]
    
    return output

def apply_dilation(kernel, dilation):
    """
    Apply dilation to a kernel by inserting zeros between elements.
    
    Args:
        kernel: 2D numpy array representing the original kernel
        dilation: Integer dilation factor (1 = no dilation)
        
    Returns:
        Dilated kernel with zeros inserted between original elements
        
    Example:
        Original 3x3 kernel with dilation=2 becomes 5x5:
        [1, 2, 3]     [1, 0, 2, 0, 3]
        [4, 5, 6] --> [0, 0, 0, 0, 0]
        [7, 8, 9]     [4, 0, 5, 0, 6]
                      [0, 0, 0, 0, 0]
                      [7, 0, 8, 0, 9]
    """
    if dilation == 1:
        return kernel  # No dilation needed
    
    # Get original kernel dimensions
    kernel_h, kernel_w = kernel.shape
    
    # Calculate new dimensions after dilation
    # New size = original_size + (original_size - 1) * (dilation - 1)
    new_h = kernel_h + (kernel_h - 1) * (dilation - 1)
    new_w = kernel_w + (kernel_w - 1) * (dilation - 1)
    
    # Create dilated kernel filled with zeros
    dilated_kernel = np.zeros((new_h, new_w))
    
    # Place original kernel values at dilated positions
    for i in range(kernel_h):
        for j in range(kernel_w):
            # Calculate position in dilated kernel
            dilated_i = i * dilation
            dilated_j = j * dilation
            dilated_kernel[dilated_i, dilated_j] = kernel[i, j]
    
    return dilated_kernel

def demonstrate_stride_effects():
    """Show how stride affects output size and feature sampling."""
    print("=== Stride Effects Demonstration ===")
    
    # Use centralized test pattern
    test_image = create_test_pattern(20, 20, 'mixed')
    kernel = sobel_x_kernel()
    
    results = {}
    for stride in [1, 2, 3]:
        result = manual_convolution_2d_extended(test_image, kernel, stride=stride)
        results[f'stride_{stride}'] = result
        print(f"Stride {stride}: Input {test_image.shape} -> Output {result.shape}")
    
    # Use centralized visualization
    plot_convolution_mechanics(test_image, results, "Stride Effects")
    
    return results

def demonstrate_padding_strategies():
    """Compare different padding approaches and their effects."""
    print("=== Padding Strategies Demonstration ===")
    
    # Use centralized test pattern
    test_image = create_test_pattern(10, 10, 'corner')
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])  # Laplacian
    
    results = {}
    
    # Test different padding amounts
    for padding in [0, 1, 2]:
        result = manual_convolution_2d_extended(test_image, kernel, padding=padding)
        results[f'pad_{padding}'] = result
        print(f"Padding {padding}: Input {test_image.shape} -> Output {result.shape}")
    
    # Use centralized visualization instead of the old function
    plot_convolution_mechanics(test_image, results, "Padding Effects")
    
    return results

def demonstrate_dilation():
    """Show how dilation increases receptive field."""
    print("=== Dilation Effects Demonstration ===")
    
    # Use centralized test pattern
    test_image = create_test_pattern(15, 15, 'diagonal')
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  # Simple edge detector
    
    results = {}
    for dilation in [1, 2, 3]:
        dilated_kernel = apply_dilation(kernel, dilation)
        result = manual_convolution_2d_extended(test_image, kernel, dilation=dilation)
        results[f'dilation_{dilation}'] = result
        
        print(f"Dilation {dilation}: Kernel {kernel.shape} -> Effective {dilated_kernel.shape}")
        print(f"Receptive field: {dilated_kernel.shape}")
    
    # Use centralized visualization
    plot_convolution_mechanics(test_image, results, "Dilation Effects")
    
    return results

def test_output_calculations():
    """Test output dimension calculations using centralized functions."""
    print("=== Output Dimension Analysis ===")
    
    test_cases = [
        (28, 3, 1, 0),  # MNIST-like: 28x28 image, 3x3 kernel, stride 1, no padding
        (28, 3, 1, 1),  # Same but with padding to preserve size
        (32, 5, 2, 2),  # Larger kernel with stride
        (100, 7, 3, 3), # Galaxy-sized image
    ]
    
    # Use centralized analysis
    print_dimension_analysis(test_cases, "Convolution Output Dimensions")

def test_computational_cost():
    """Test computational cost analysis using centralized functions."""
    print("=== Computational Cost Analysis ===")
    
    configs = [
        ("Base (3x3, stride=1)", ((224, 224), 3, 1, 1)),
        ("Larger kernel (5x5)", ((224, 224), 5, 1, 2)),  
        ("Stride=2", ((224, 224), 3, 2, 1)),
        ("No padding", ((224, 224), 3, 1, 0)),
    ]
    
    # Use centralized analysis
    print_cost_analysis(configs, "Computational Cost Comparison")

if __name__ == "__main__":
    print("=== Convolution Mechanics Testing ===")
    
    # Test all effects with updated functions
    demonstrate_stride_effects()
    demonstrate_padding_strategies() 
    demonstrate_dilation()
    test_output_calculations()
    test_computational_cost()
    
    print("All convolution mechanics tests completed!")