"""
Deep dive into convolution mechanics and variants.

This module explores stride, padding, dilation, and output dimension calculations
to build complete understanding of how convolution parameters affect results.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

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
    # Create test image
    test_image = create_test_pattern(20, 20)
    kernel = sobel_x_kernel()
    
    print("=== Stride Effects Demonstration ===")
    
    results = {}
    for stride in [1, 2, 3]:
        result = manual_convolution_2d_extended(test_image, kernel, stride=stride)
        results[f'stride_{stride}'] = result
        print(f"Stride {stride}: Input {test_image.shape} -> Output {result.shape}")
    
    # Visualize results
    visualize_stride_effects(test_image, results)
    
    return results

def demonstrate_padding_strategies():
    """Compare different padding approaches and their effects."""
    test_image = create_test_pattern(10, 10)
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])  # Laplacian
    
    print("=== Padding Strategies Demonstration ===")
    
    results = {}
    
    # Test different padding amounts
    for padding in [0, 1, 2]:
        result = manual_convolution_2d_extended(test_image, kernel, padding=padding)
        results[f'pad_{padding}'] = result
        print(f"Padding {padding}: Input {test_image.shape} -> Output {result.shape}")
    
    # Implement different padding TYPES (constant, reflect, etc.)
    print("\n=== Different Padding Types ===")
    padding_types = ['constant', 'reflect', 'edge', 'wrap']
    
    for padding_type in padding_types:
        result = manual_convolution_2d_with_padding_types(test_image, kernel, padding=1, padding_mode=padding_type)
        results[f'type_{padding_type}'] = result
        print(f"Padding type '{padding_type}': Output {result.shape}")
    
    visualize_padding_effects(test_image, results)
    return results

def demonstrate_dilation():
    """Show how dilation increases receptive field."""
    test_image = create_test_pattern(15, 15)
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  # Simple edge detector
    
    print("=== Dilation Effects Demonstration ===")
    
    results = {}
    for dilation in [1, 2, 3]:
        dilated_kernel = apply_dilation(kernel, dilation)
        result = manual_convolution_2d_extended(test_image, kernel, dilation=dilation)
        results[f'dilation_{dilation}'] = result
        
        print(f"Dilation {dilation}: Kernel {kernel.shape} -> Effective {dilated_kernel.shape}")
        print(f"Receptive field: {dilated_kernel.shape}")
    
    visualize_dilation_effects(test_image, results)
    return results

def calculate_output_dimensions_formula():
    """Practice and demonstrate output dimension calculations."""
    print("=== Output Dimension Calculations ===")
    
    def calc_output_size(input_size, kernel_size, stride, padding):
        """Calculate output size using the standard formula."""
        return (input_size + 2 * padding - kernel_size) // stride + 1
    
    # Test cases
    test_cases = [
        (28, 3, 1, 0),  # MNIST-like: 28x28 image, 3x3 kernel, stride 1, no padding
        (28, 3, 1, 1),  # Same but with padding to preserve size
        (32, 5, 2, 2),  # Larger kernel with stride
        (100, 7, 3, 3), # Galaxy-sized image
    ]
    
    print("Input Size | Kernel | Stride | Padding | Output Size")
    print("-" * 55)
    
    for input_size, kernel_size, stride, padding in test_cases:
        output_size = calc_output_size(input_size, kernel_size, stride, padding)
        print(f"{input_size:9d} | {kernel_size:6d} | {stride:6d} | {padding:7d} | {output_size:11d}")
    
    # Add receptive field calculations
    # Show how multiple layers compound the receptive field
    print("\n=== Receptive Field Calculations ===")
    
    def calculate_receptive_field(layers):
        """Calculate receptive field through multiple layers.
        
        Args:
            layers: List of (kernel_size, stride) tuples
            
        Returns:
            List of receptive field sizes after each layer
        """
        receptive_field = 1
        jump = 1
        receptive_fields = [receptive_field]
        
        for kernel_size, stride in layers:
            receptive_field = receptive_field + (kernel_size - 1) * jump
            jump = jump * stride
            receptive_fields.append(receptive_field)
        
        return receptive_fields
    
    # Example multi-layer networks
    networks = [
        ("3 layers, 3x3 kernels", [(3, 1), (3, 1), (3, 1)]),
        ("With stride=2 in middle", [(3, 1), (3, 2), (3, 1)]),
        ("Larger kernels", [(5, 1), (5, 1), (3, 1)]),
        ("Mixed configuration", [(3, 1), (5, 2), (3, 1), (3, 1)])
    ]
    
    print("\nReceptive Field Growth Through Layers:")
    print("Network | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Layer 4")
    print("-" * 65)
    
    for name, config in networks:
        rf_progression = calculate_receptive_field(config)
        rf_str = " | ".join([f"{rf:7d}" if i < len(rf_progression) else "       " 
                            for i, rf in enumerate(rf_progression[:5])])
        if len(rf_progression) > 5:
            rf_str += " | ..."
        print(f"{name:20s} | {rf_str}")
    
    return calc_output_size

def analyze_computational_cost():
    """Analyze how different parameters affect computational cost."""
    print("=== Computational Cost Analysis ===")
    
    def calculate_operations(input_h, input_w, kernel_h, kernel_w, stride, padding):
        """Calculate number of multiply-add operations."""
        output_h = (input_h + 2 * padding - kernel_h) // stride + 1
        output_w = (input_w + 2 * padding - kernel_w) // stride + 1
        
        # Each output pixel requires kernel_h * kernel_w operations
        total_ops = output_h * output_w * kernel_h * kernel_w
        return total_ops, (output_h, output_w)
    
    # Compare different configurations
    base_config = (224, 224, 3, 3, 1, 1)  # ImageNet-like
    
    configs = [
        ("Base (3x3, stride=1)", (224, 224, 3, 3, 1, 1)),
        ("Larger kernel (5x5)", (224, 224, 5, 5, 1, 2)),
        ("Stride=2", (224, 224, 3, 3, 2, 1)),
        ("No padding", (224, 224, 3, 3, 1, 0)),
    ]
    
    print("Configuration | Operations | Output Shape | Relative Cost")
    print("-" * 60)
    
    base_ops = None
    for name, config in configs:
        ops, output_shape = calculate_operations(*config)
        if base_ops is None:
            base_ops = ops
            relative = 1.0
        else:
            relative = ops / base_ops
            
        print(f"{name:20s} | {ops:10,d} | {output_shape!s:12s} | {relative:7.2f}x")

def create_test_pattern(height, width):
    """Create a test pattern for demonstrating convolution effects."""
    pattern = np.zeros((height, width))
    
    # Add some geometric shapes
    center_h, center_w = height // 2, width // 2
    
    # Vertical line
    pattern[:, center_w] = 1
    
    # Horizontal line  
    pattern[center_h, :] = 1
    
    # Diagonal corners
    for i in range(min(height//4, width//4)):
        if i < height and i < width:
            pattern[i, i] = 0.5
            pattern[height-1-i, width-1-i] = 0.5
    
    return pattern

def visualize_stride_effects(original, results):
    """Create visualization showing original + stride results"""
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(15, 4))
    
    # Show original image
    axes[0].imshow(original, cmap='gray', interpolation='nearest')
    axes[0].set_title(f'Original Image\n{original.shape}')
    axes[0].axis('off')
    
    # Show results for each stride
    for idx, (key, result) in enumerate(results.items()):
        stride_value = key.split('_')[1]
        
        axes[idx + 1].imshow(result, cmap='RdBu', interpolation='nearest')
        axes[idx + 1].set_title(f'Stride {stride_value}\nOutput: {result.shape}')
        axes[idx + 1].axis('off')
        
        plt.colorbar(axes[idx + 1].get_images()[0], ax=axes[idx + 1], fraction=0.046)
    
    plt.tight_layout()
    plt.show()

def visualize_padding_effects(original, results):
    """Create visualization showing padding effects"""
    # Separate padding amount results from padding type results
    amount_results = {k: v for k, v in results.items() if k.startswith('pad_')}
    type_results = {k: v for k, v in results.items() if k.startswith('type_')}
    
    if amount_results:
        # Visualize padding amounts
        fig, axes = plt.subplots(2, len(amount_results), figsize=(12, 8))
        
        for idx, (key, result) in enumerate(amount_results.items()):
            padding_value = int(key.split('_')[1])
            
            # Top row: show the padded input
            padded_input = np.pad(original, padding_value, mode='constant', constant_values=0)
            axes[0, idx].imshow(padded_input, cmap='gray', interpolation='nearest')
            axes[0, idx].set_title(f'Input + Padding {padding_value}\n{padded_input.shape}')
            axes[0, idx].axis('off')
            
            # Bottom row: show the convolution result
            axes[1, idx].imshow(result, cmap='RdBu', interpolation='nearest')
            axes[1, idx].set_title(f'Output: {result.shape}')
            axes[1, idx].axis('off')
            
            plt.colorbar(axes[1, idx].get_images()[0], ax=axes[1, idx], fraction=0.046)
        
        plt.tight_layout()
        plt.suptitle('Padding Amount Effects', y=1.02)
        plt.show()
    
    if type_results:
        # Visualize padding types
        fig, axes = plt.subplots(2, len(type_results), figsize=(16, 8))
        
        for idx, (key, result) in enumerate(type_results.items()):
            padding_type = key.split('_')[1]
            
            # Top row: show the padded input for each type
            if padding_type == 'constant':
                padded_input = np.pad(original, 1, mode='constant', constant_values=0)
            elif padding_type == 'reflect':
                padded_input = np.pad(original, 1, mode='reflect')
            elif padding_type == 'edge':
                padded_input = np.pad(original, 1, mode='edge')
            elif padding_type == 'wrap':
                padded_input = np.pad(original, 1, mode='wrap')
            
            axes[0, idx].imshow(padded_input, cmap='gray', interpolation='nearest')
            axes[0, idx].set_title(f'Input + {padding_type} padding\n{padded_input.shape}')
            axes[0, idx].axis('off')
            
            # Bottom row: show the convolution result
            axes[1, idx].imshow(result, cmap='RdBu', interpolation='nearest')
            axes[1, idx].set_title(f'Output: {result.shape}')
            axes[1, idx].axis('off')
            
            plt.colorbar(axes[1, idx].get_images()[0], ax=axes[1, idx], fraction=0.046)
        
        plt.tight_layout()
        plt.suptitle('Padding Type Effects', y=1.02)
        plt.show()

def visualize_dilation_effects(original, results):
    """Create visualization showing dilation effects"""
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  # Same kernel used in demonstration
    
    fig, axes = plt.subplots(2, len(results), figsize=(12, 8))
    
    for idx, (key, result) in enumerate(results.items()):
        dilation_value = int(key.split('_')[1])
        
        # Top row: show what the dilated kernel looks like
        dilated_kernel = apply_dilation(kernel, dilation_value)
        
        axes[0, idx].imshow(dilated_kernel, cmap='RdBu', interpolation='nearest')
        axes[0, idx].set_title(f'Dilation {dilation_value} Kernel\n{dilated_kernel.shape}')
        axes[0, idx].axis('off')
        
        # Bottom row: show the convolution result
        axes[1, idx].imshow(result, cmap='RdBu', interpolation='nearest')
        axes[1, idx].set_title(f'Output: {result.shape}')
        axes[1, idx].axis('off')
        
        plt.colorbar(axes[1, idx].get_images()[0], ax=axes[1, idx], fraction=0.046)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_stride_effects()
    demonstrate_padding_strategies() 
    demonstrate_dilation()
    calculate_output_dimensions_formula()
    analyze_computational_cost()