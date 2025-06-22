"""
Core 2D convolution implementation from scratch.

Building understanding of how convolution operations work on images,
which is the foundation of all CNN operations.
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.test_data import create_simple_test_image
from utils.visualization import plot_convolution_demo

def manual_convolution_2d(image, kernel, stride=1, padding=0):
    """
    Implement 2D convolution from scratch.
    
    Args:
        image: 2D numpy array representing image
        kernel: 2D numpy array representing convolution kernel
        stride: Step size for convolution operation
        padding: Padding to add around image borders
        
    Returns:
        Convolved image as 2D numpy array
    """
    # Add padding if specified
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    
    # Get dimensions
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    # Calculate output dimensions
    output_h = (image_h - kernel_h) // stride + 1
    output_w = (image_w - kernel_w) // stride + 1
    
    # Initialize output
    output = np.zeros((output_h, output_w))
    
    # Implement the 2D convolution using nested loops
    # Outer loops: iterate over output positions
    for i in range(output_h):
        for j in range(output_w):
            # Calculate starting position in image for this output pixel
            start_i = i * stride
            start_j = j * stride
            
            # Inner loops: iterate over kernel positions
            for ki in range(kernel_h):
                for kj in range(kernel_w):
                    # Calculate corresponding image position
                    img_i = start_i + ki
                    img_j = start_j + kj
                    
                    # Multiply image pixel by kernel weight and accumulate
                    output[i, j] += image[img_i, img_j] * kernel[ki, kj]
    
    return output

def create_edge_kernels():
    """Create basic edge detection kernels."""
    # Vertical edge detector (detects horizontal changes)
    vertical_edge = np.array([[-1, 0, 1],
                             [-1, 0, 1], 
                             [-1, 0, 1]])
    
    # Horizontal edge detector (detects vertical changes)  
    horizontal_edge = np.array([[-1, -1, -1],
                               [ 0,  0,  0],
                               [ 1,  1,  1]])
    
    return vertical_edge, horizontal_edge

def test_convolution_2d():
    """Test 2D convolution with edge detection."""
    # Create test image
    image = create_simple_test_image()
    vertical_kernel, horizontal_kernel = create_edge_kernels()
    
    print("Original image:")
    print(image)
    
    # Test vertical edge detection
    vertical_result = manual_convolution_2d(image, vertical_kernel)
    print("\nVertical edge detection result:")
    print(vertical_result)
    
    # Test horizontal edge detection  
    horizontal_result = manual_convolution_2d(image, horizontal_kernel)
    print("\nHorizontal edge detection result:")
    print(horizontal_result)
    
    return image, vertical_result, horizontal_result

if __name__ == "__main__":
    # Use centralized test data
    test_image = create_simple_test_image()
    
    # Your existing kernel definitions stay the same
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    # Apply convolutions
    result_v = manual_convolution_2d(test_image, sobel_x)
    result_h = manual_convolution_2d(test_image, sobel_y)
    
    # Use centralized visualization
    plot_convolution_demo(test_image, sobel_x, result_v, result_h, 
                         title="Edge Detection Demo")
    
    print("Convolution test completed!")