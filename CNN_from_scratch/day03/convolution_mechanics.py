"""
Deep dive into convolution mechanics and variants.

This module explores stride, padding, dilation, and output dimension calculations
to build complete understanding of how convolution parameters affect results.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Tuple, Optional, Union, Any

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Centralized imports
from utils.output_system import log_print, log_experiment_start, log_experiment_end
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

def manual_convolution_2d_extended(
    image: np.ndarray, 
    kernel: np.ndarray, 
    stride: int = 1, 
    padding: int = 0, 
    dilation: int = 1
) -> np.ndarray:
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

def manual_convolution_2d_with_padding_types(
    image: np.ndarray, 
    kernel: np.ndarray, 
    stride: int = 1, 
    padding: int = 0, 
    padding_mode: str = 'constant'
) -> np.ndarray:
    """
    Extended convolution with different padding modes.
    
    Args:
        image: Input image array
        kernel: Convolution kernel
        stride: Stride value
        padding: Amount of padding
        padding_mode: Type of padding ('constant', 'reflect', 'edge', 'wrap')
        
    Returns:
        Convolved image with specified padding mode
    """
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

def apply_dilation(kernel: np.ndarray, dilation: int) -> np.ndarray:
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

def demonstrate_stride_effects() -> Dict[str, np.ndarray]:
    """
    Show how stride affects output size and feature sampling.
    
    Returns:
        Dictionary mapping stride descriptions to results
    """
    log_print("=== Stride Effects Demonstration ===", level="SUBHEADER")
    
    # Use centralized test pattern
    test_image = create_test_pattern(20, 20, 'mixed')
    kernel = sobel_x_kernel()
    
    results = {}
    for stride in [1, 2, 3]:
        result = manual_convolution_2d_extended(test_image, kernel, stride=stride)
        results[f'stride_{stride}'] = result
        log_print(f"Stride {stride}: Input {test_image.shape} -> Output {result.shape}")
    
    # Use centralized visualization
    plot_convolution_mechanics(test_image, results, "Stride Effects")
    
    return results

def demonstrate_padding_strategies() -> Dict[str, np.ndarray]:
    """
    Compare different padding approaches and their effects.
    
    Returns:
        Dictionary mapping padding descriptions to results
    """
    log_print("=== Padding Strategies Demonstration ===", level="SUBHEADER")
    
    # Use centralized test pattern
    test_image = create_test_pattern(10, 10, 'corner')
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)  # Laplacian
    
    results = {}
    
    # Test different padding amounts
    for padding in [0, 1, 2]:
        result = manual_convolution_2d_extended(test_image, kernel, padding=padding)
        results[f'pad_{padding}'] = result
        log_print(f"Padding {padding}: Input {test_image.shape} -> Output {result.shape}")
    
    # Use centralized visualization instead of the old function
    plot_convolution_mechanics(test_image, results, "Padding Effects")
    
    return results

def demonstrate_dilation() -> Dict[str, np.ndarray]:
    """
    Show how dilation increases receptive field.
    
    Returns:
        Dictionary mapping dilation descriptions to results
    """
    log_print("=== Dilation Effects Demonstration ===", level="SUBHEADER")
    
    # Use centralized test pattern
    test_image = create_test_pattern(15, 15, 'diagonal')
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)  # Simple edge detector
    
    results = {}
    for dilation in [1, 2, 3]:
        dilated_kernel = apply_dilation(kernel, dilation)
        result = manual_convolution_2d_extended(test_image, kernel, dilation=dilation)
        results[f'dilation_{dilation}'] = result
        
        log_print(f"Dilation {dilation}: Kernel {kernel.shape} -> Effective {dilated_kernel.shape}")
        log_print(f"Receptive field: {dilated_kernel.shape}")
    
    # Use centralized visualization
    plot_convolution_mechanics(test_image, results, "Dilation Effects")
    
    return results

def test_output_calculations() -> None:
    """Test output dimension calculations using centralized functions."""
    log_print("=== Output Dimension Analysis ===", level="SUBHEADER")
    
    test_cases = [
        (28, 3, 1, 0),  # MNIST-like: 28x28 image, 3x3 kernel, stride 1, no padding
        (28, 3, 1, 1),  # Same but with padding to preserve size
        (32, 5, 2, 2),  # Larger kernel with stride
        (100, 7, 3, 3), # Galaxy-sized image
    ]
    
    # Use centralized analysis
    print_dimension_analysis(test_cases, "Convolution Output Dimensions")

def test_computational_cost() -> None:
    """Test computational cost analysis using centralized functions."""
    log_print("=== Computational Cost Analysis ===", level="SUBHEADER")
    
    configs = [
        ("Base (3x3, stride=1)", ((224, 224), 3, 1, 1)),
        ("Larger kernel (5x5)", ((224, 224), 5, 1, 2)),  
        ("Stride=2", ((224, 224), 3, 2, 1)),
        ("No padding", ((224, 224), 3, 1, 0)),
    ]
    
    # Use centralized analysis
    print_cost_analysis(configs, "Computational Cost Comparison")

def create_scale_sensitive_kernels(scale: str) -> Optional[np.ndarray]:
    """
    Create kernels that detect features at different spatial scales.
    
    For galaxies, we need to detect features at different scales:
    - Fine structure (3x3): Details like bright knots
    - Medium structure (5x5): Spiral arm segments  
    - Large structure (7x7): Overall galaxy shape
    
    Args:
        scale: Scale type ('fine', 'medium', 'large')
        
    Returns:
        Kernel array for the specified scale, or None if invalid scale
    """
    if scale == 'fine':
        # 3x3 - Good for sharp edges and fine details
        return np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    elif scale == 'medium':
        # 5x5 - Smoothed edge detector
        return np.array([[-1, -1, 0, 1, 1],
                        [-2, -2, 0, 2, 2],
                        [-2, -2, 0, 2, 2],
                        [-2, -2, 0, 2, 2],
                        [-1, -1, 0, 1, 1]], dtype=np.float32)
    elif scale == 'large':
        # 7x7 - Smoothed even more for large-scale gradients
        return np.array([[-1, -1, -1, 0, 1, 1, 1],
                        [-2, -2, -2, 0, 2, 2, 2],
                        [-2, -2, -2, 0, 2, 2, 2],
                        [-2, -2, -2, 0, 2, 2, 2],
                        [-2, -2, -2, 0, 2, 2, 2],
                        [-2, -2, -2, 0, 2, 2, 2],
                        [-1, -1, -1, 0, 1, 1, 1]], dtype=np.float32)
    else:
        log_print(f"Unknown scale: {scale}")
        return None

def test_kernel_comparison() -> Dict[str, np.ndarray]:
    """
    Compare different kernel types on the same galaxy image.
    
    Returns:
        Dictionary mapping kernel names to their responses
    """
    log_print("=== Kernel Comparison Test ===", level="SUBHEADER")
    
    from utils.test_data import create_synthetic_galaxy
    galaxy = create_synthetic_galaxy()
    
    # Test different scale kernels
    fine_response = manual_convolution_2d(galaxy, create_scale_sensitive_kernels('fine'))
    medium_response = manual_convolution_2d(galaxy, create_scale_sensitive_kernels('medium'))
    large_response = manual_convolution_2d(galaxy, create_scale_sensitive_kernels('large'))
    
    # Compare with standard Sobel
    sobel_x_response = manual_convolution_2d(galaxy, sobel_x_kernel())
    
    responses = {
        'Sobel X (Standard)': sobel_x_response,
        'Fine Scale (Custom)': fine_response,
        'Medium Scale': medium_response,
        'Large Scale': large_response
    }
    
    from utils.visualization import plot_feature_responses
    plot_feature_responses(galaxy, responses, title="Astronomical Kernel Comparison")
    
    return responses

def main() -> None:
    """Main execution function for Day 3 experiments."""
    log_experiment_start(3, "Deep Dive into Kernels and Convolution")
    
    # Test all effects with updated functions
    demonstrate_stride_effects()
    demonstrate_padding_strategies() 
    demonstrate_dilation()
    test_output_calculations()
    test_computational_cost()
    test_kernel_comparison()
    
    log_print("All convolution mechanics tests completed!")
    log_experiment_end(3)

if __name__ == "__main__":
    main()