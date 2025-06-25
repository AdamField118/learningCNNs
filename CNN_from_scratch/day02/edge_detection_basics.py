import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Tuple, Dict, List, Optional, Union

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import enhanced output system
from utils.output_system import log_print, log_experiment_start, log_experiment_end, save_plot
from utils.test_data import create_simple_test_image
from utils.visualization import plot_convolution_demo

def manual_convolution_2d(
    image: np.ndarray, 
    kernel: np.ndarray, 
    stride: int = 1, 
    padding: int = 0
) -> np.ndarray:
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

def create_edge_kernels() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create basic edge detection kernels.
    
    Returns:
        Tuple of (vertical_edge_kernel, horizontal_edge_kernel)
    """
    # Vertical edge detector (detects horizontal changes)
    vertical_edge = np.array([[-1, 0, 1],
                             [-1, 0, 1], 
                             [-1, 0, 1]], dtype=np.float32)
    
    # Horizontal edge detector (detects vertical changes)  
    horizontal_edge = np.array([[-1, -1, -1],
                               [ 0,  0,  0],
                               [ 1,  1,  1]], dtype=np.float32)
    
    return vertical_edge, horizontal_edge

def test_convolution_2d() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Test 2D convolution with edge detection.
    
    Returns:
        Tuple of (original_image, vertical_result, horizontal_result)
    """
    # Create test image
    image = create_simple_test_image()
    vertical_kernel, horizontal_kernel = create_edge_kernels()
    
    log_print("=== Testing 2D Convolution ===", level="SUBHEADER")
    log_print("Original image:")
    log_print(str(image))
    
    # Test vertical edge detection
    vertical_result = manual_convolution_2d(image, vertical_kernel)
    log_print("\nVertical edge detection result:")
    log_print(str(vertical_result))
    
    # Test horizontal edge detection  
    horizontal_result = manual_convolution_2d(image, horizontal_kernel)
    log_print("\nHorizontal edge detection result:")
    log_print(str(horizontal_result))
    
    return image, vertical_result, horizontal_result

def convolution_1d_simple(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
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

def test_simple_convolution() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Test the 1D convolution with a simple example.
    
    Returns:
        Tuple of (signal, kernel, result)
    """
    log_print("=== Testing 1D Convolution ===", level="SUBHEADER")
    
    # Simple test case
    signal = np.array([1, 2, 3, 4, 5])
    kernel = np.array([0.5, 1, 0.5])  # Simple smoothing kernel
    
    result = convolution_1d_simple(signal, kernel)
    log_print(f"Signal: {signal}")
    log_print(f"Kernel: {kernel}")
    log_print(f"Result: {result}")
    
    # Compare with NumPy's implementation to verify correctness
    numpy_result = np.convolve(signal, kernel, mode='full')
    log_print(f"NumPy result: {numpy_result}")
    log_print(f"Match: {np.allclose(result, numpy_result)}")
    
    return signal, kernel, result

def demonstrate_edge_detection() -> Dict[str, np.ndarray]:
    """
    Demonstrate edge detection on test patterns.
    
    Returns:
        Dictionary mapping test names to results
    """
    log_print("=== Edge Detection Demonstration ===", level="SUBHEADER")
    
    # Use centralized test data
    test_image = create_simple_test_image()
    
    # Get edge detection kernels
    from kernels.edge_detection_kernels import sobel_x_kernel, sobel_y_kernel
    sobel_x = sobel_x_kernel()
    sobel_y = sobel_y_kernel()
    
    # Apply convolutions
    result_x = manual_convolution_2d(test_image, sobel_x)
    result_y = manual_convolution_2d(test_image, sobel_y)
    
    # Use centralized visualization
    plot_convolution_demo(test_image, sobel_x, result_x, result_y, 
                         title="Edge Detection Demo")
    save_plot("edge_detection_demo.png")
    
    log_print("Edge detection completed!")
    
    return {
        'original': test_image,
        'sobel_x': result_x,
        'sobel_y': result_y
    }

def galaxy_edge_detection() -> Dict[str, np.ndarray]:
    """
    Apply edge detection to synthetic galaxy data.
    
    Returns:
        Dictionary with galaxy and edge detection results
    """
    log_print("=== Galaxy Edge Detection ===", level="SUBHEADER")
    
    # Create synthetic galaxy
    from utils.test_data import create_synthetic_galaxy
    galaxy = create_synthetic_galaxy(size=50, spiral_arms=True)
    
    # Apply edge detection
    from kernels.edge_detection_kernels import sobel_x_kernel, sobel_y_kernel
    sobel_x = sobel_x_kernel()
    sobel_y = sobel_y_kernel()
    
    edges_x = manual_convolution_2d(galaxy, sobel_x)
    edges_y = manual_convolution_2d(galaxy, sobel_y)
    
    # Combine edge responses
    edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
    
    # Visualize results
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(galaxy, cmap='hot')
    plt.title('Synthetic Galaxy')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(edges_x, cmap='gray')
    plt.title('Horizontal Edges (Sobel X)')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(edges_y, cmap='gray')
    plt.title('Vertical Edges (Sobel Y)')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(edge_magnitude, cmap='gray')
    plt.title('Edge Magnitude')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    save_plot("galaxy_edge_detection.png")
    
    log_print(f"Galaxy shape: {galaxy.shape}")
    log_print(f"Edge magnitude range: [{edge_magnitude.min():.3f}, {edge_magnitude.max():.3f}]")
    
    return {
        'galaxy': galaxy,
        'edges_x': edges_x,
        'edges_y': edges_y,
        'edge_magnitude': edge_magnitude
    }

def main() -> None:
    """Main execution function for Day 2 experiments."""
    log_experiment_start(2, "Convolution and Edge Detection")
    
    # Run all experiments
    test_simple_convolution()
    test_convolution_2d()
    demonstrate_edge_detection()
    galaxy_edge_detection()
    
    log_experiment_end(2)

if __name__ == "__main__":
    main()