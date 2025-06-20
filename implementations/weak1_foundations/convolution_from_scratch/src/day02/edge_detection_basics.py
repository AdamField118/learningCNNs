"""
Core 2D convolution implementation from scratch.

Building understanding of how convolution operations work on images,
which is the foundation of all CNN operations.
"""

import numpy as np
import matplotlib.pyplot as plt

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

def create_test_image():
    """Create a simple test image with clear patterns."""
    # Create a 7x7 image with a vertical line
    image = np.zeros((7, 7))
    image[:, 3] = 1  # Vertical line in middle
    return image

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
    image = create_test_image()
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

def visualize_convolution():
    """Visualize the convolution process."""
    image, vertical_result, horizontal_result = test_convolution_2d()
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image\n(Vertical Line)')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(vertical_result, cmap='gray')
    plt.title('Vertical Edge Detector\n(Should detect the line)')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(horizontal_result, cmap='gray')
    plt.title('Horizontal Edge Detector\n(Should show nothing)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_convolution_2d()
    visualize_convolution()