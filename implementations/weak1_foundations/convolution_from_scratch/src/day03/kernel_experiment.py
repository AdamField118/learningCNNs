"""
Systematic testing of different kernel designs for astronomical applications.

This module explores how different kernel designs affect feature detection
in galaxy images, with focus on scales and patterns relevant to astrophysics.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import cv2
from astropy.convolution import Gaussian2DKernel
from astropy import units as u

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from kernels.edge_detection_kernels import sobel_x_kernel, sobel_y_kernel
from day02.edge_detection_basics import manual_convolution_2d

def create_scale_sensitive_kernels(scale):
    """
    Create kernels that detect features at different spatial scales.
    
    For galaxies, we need to detect features at different scales:
    - Fine structure (3x3): Details like bright knots
    - Medium structure (5x5): Spiral arm segments  
    - Large structure (7x7): Overall galaxy shape
    """
    if (scale == 'fine') :
        # 3x3 - Good for sharp edges and fine details
        return np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    elif (scale == 'medium') :
        # 5x5 - Smoothed edge detector
        return np.array([[-1, -1, 0, 1, 1],
                        [-2, -2, 0, 2, 2],
                        [-2, -2, 0, 2, 2],
                        [-2, -2, 0, 2, 2],
                        [-1, -1, 0, 1, 1]], dtype=np.float32)
    elif (scale == 'large') :
        # 7x7 - Smoothed even more for large-scale gradients
        return np.array([[-1, -1, -1, 0, 1, 1, 1],
                        [-2, -2, -2, 0, 2, 2, 2],
                        [-2, -2, -2, 0, 2, 2, 2],
                        [-2, -2, -2, 0, 2, 2, 2],
                        [-2, -2, -2, 0, 2, 2, 2],
                        [-2, -2, -2, 0, 2, 2, 2],
                        [-1, -1, -1, 0, 1, 1, 1]], dtype=np.float32)
    else :
        print(f"Unknown scale: {scale}")
        return None

def create_orientation_sensitive_kernels(scale, orientation):
    """
    Create kernels that respond to specific orientations.
    
    Spiral galaxies have preferred orientations in their arms.
    We want kernels that can detect these patterns.
    """
    base_kernel = create_scale_sensitive_kernels(scale)
    
    if orientation == 'horizontal':  # 0 degrees
        return base_kernel
    elif orientation == 'vertical':  # 90 degrees  
        return base_kernel.T
    elif orientation == 'diagonal_1':  # 45 degrees
        return rotate_kernel_scipy(base_kernel, 45)
    elif orientation == 'diagonal_2':  # -45 degrees
        return rotate_kernel_scipy(base_kernel, -45)
    else:
        print(f"Unknown orientation: {orientation}")
        return None

def rotate_kernel_scipy(kernel, angle_degrees, preserve_properties=True):
    """Rotate kernel using SciPy with property preservation."""
    # Store original properties
    original_sum = np.sum(kernel)
    
    # Rotate with bicubic interpolation
    rotated = ndimage.rotate(
        kernel, 
        angle_degrees,
        reshape=False,  # Keep original size 
        order=3,        # Bicubic interpolation
        mode='constant',
        cval=0.0,
        prefilter=True
    )
    
    # Preserve sum (critical for detection kernels)
    if preserve_properties and original_sum != 0:
        current_sum = np.sum(rotated)
        if current_sum != 0:
            rotated = rotated * (original_sum / current_sum)
    
    return rotated

def rotate_kernel_opencv(kernel, angle_degrees, interpolation='cubic'):
    """High-performance rotation using OpenCV."""
    # Map interpolation methods
    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR, 
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    h, w = kernel.shape
    center = (w // 2, h // 2)
    
    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    
    # Apply rotation with boundary reflection
    rotated = cv2.warpAffine(
        kernel.astype(np.float32), 
        M, 
        (w, h),
        flags=interp_map[interpolation],
        borderMode=cv2.BORDER_REFLECT
    )
    
    return rotated

def create_rotated_astronomical_kernel(x_stddev, y_stddev, angle_degrees, size=21):
    """Create rotated astronomical kernel using Astropy."""
    kernel = Gaussian2DKernel(
        x_stddev=x_stddev,
        y_stddev=y_stddev, 
        theta=angle_degrees * u.deg,
        x_size=size,
        y_size=size
    )
    return kernel.array

def create_brightness_gradient_kernels(type):
    """
    Create kernels that detect subtle brightness variations.
    
    Galaxy shear measurement requires detecting very subtle
    shape distortions that appear as brightness gradients.
    """

    # Gentle X-direction gradient (very smooth transition)
    gentle_x_gradient = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]], dtype=np.float32) / 6
    
    # Gentle Y-direction gradient  
    gentle_y_gradient = gentle_x_gradient.T
    
    # Radial gradient detector (for circular/elliptical features)
    radial_gradient = np.array([[-1, -2, -1],
                               [-2,  8, -2],
                               [-1, -2, -1]], dtype=np.float32)
    
    # Second-order gradient (detects curvature changes)
    second_order = np.array([[ 1, -2,  1],
                            [-2,  4, -2],
                            [ 1, -2,  1]], dtype=np.float32)
    
    if (type == 'gentle_x') : 
        return gentle_x_gradient
    elif (type == 'gentle_y') : 
        return gentle_y_gradient
    elif (type == 'radial') : 
        return radial_gradient
    elif (type == 'curvature') : 
        return second_order
    else : 
        print(f"Unknown type: {type}")
        return None

def test_kernel_comparison():
    """Compare different kernel types on the same galaxy image."""
    
    # Create synthetic galaxy (copy from Day 2 or import)
    galaxy = create_synthetic_galaxy()
    
    # Test 1: Scale Comparison
    fine_response = manual_convolution_2d(galaxy, create_scale_sensitive_kernels('fine'))
    medium_response = manual_convolution_2d(galaxy, create_scale_sensitive_kernels('medium'))
    large_response = manual_convolution_2d(galaxy, create_scale_sensitive_kernels('large'))
    
    # Test 2: Your Custom vs Standard
    sobel_x_response = manual_convolution_2d(galaxy, sobel_x_kernel())
    fine_custom_response = manual_convolution_2d(galaxy, create_scale_sensitive_kernels('fine'))
    
    # Test 3: Orientation Sensitivity
    horizontal_response = manual_convolution_2d(galaxy, create_orientation_sensitive_kernels('fine', 'horizontal'))
    diagonal_response = manual_convolution_2d(galaxy, create_orientation_sensitive_kernels('fine', 'diagonal_1'))
    
    # Test 4: Gradient Detection
    gentle_x_response = manual_convolution_2d(galaxy, create_brightness_gradient_kernels('gentle_x'))
    radial_response = manual_convolution_2d(galaxy, create_brightness_gradient_kernels('radial'))
    
    # Visualization
    visualize_kernel_comparison(galaxy, {
        'Original': galaxy,
        'Sobel X (Standard)': sobel_x_response,
        'Fine Scale (Custom)': fine_custom_response,
        'Medium Scale': medium_response,
        'Diagonal Orientation': diagonal_response,
        'Gentle Gradient': gentle_x_response,
        'Radial Gradient': radial_response
    })

def visualize_kernel_comparison(original, responses):
    """Create comprehensive visualization of kernel responses."""
    n_plots = len(responses)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    
    # Much larger figure with explicit spacing
    fig, axes = plt.subplots(rows, cols, figsize=(18, 8 * rows))
    
    # Handle case where we only have one row
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easy indexing
    axes = axes.flatten()
    
    for i, (name, response) in enumerate(responses.items()):
        ax = axes[i]
        im = ax.imshow(response, cmap='gray')
        
        # Add colorbar with proper spacing
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Simple, short titles
        if name != 'Original':
            max_response = np.max(np.abs(response))
            # Very short titles
            short_names = {
                'Sobel X (Standard)': 'Sobel X',
                'Fine Scale (Custom)': 'Fine Scale', 
                'Medium Scale': 'Medium',
                'Diagonal Orientation': 'Diagonal',
                'Gentle Gradient': 'Gentle',
                'Radial Gradient': 'Radial'
            }
            display_name = short_names.get(name, name)
            ax.set_title(f'{display_name}\n{max_response:.1f}', fontsize=12)
        else:
            ax.set_title(name, fontsize=14)
    
    # Hide unused subplots
    for j in range(len(responses), len(axes)):
        axes[j].set_visible(False)
    
    # Critical: Much more spacing
    plt.subplots_adjust(
        left=0.05,      # Left margin
        bottom=0.05,    # Bottom margin  
        right=0.95,     # Right margin
        top=0.85,       # Top margin (leave room for titles!)
        wspace=0.3,     # Width spacing between plots
        hspace=0.5      # Height spacing between plots
    )
    
    plt.show()

def create_synthetic_galaxy():
    """Create synthetic galaxy (copy from Day 2 or create new version)."""
    # You can copy this from your Day 2 notebook
    # Or create a simpler version for testing
    size = 50
    center = size // 2
    galaxy = np.zeros((size, size))
    
    # Create circular galaxy base
    y, x = np.ogrid[:size, :size]
    mask = (x - center)**2 + (y - center)**2 <= (size//3)**2
    galaxy[mask] = 0.5
    
    # Add spiral arms (simplified)
    for i in range(size):
        for j in range(size):
            dx, dy = j - center, i - center
            angle = np.arctan2(dy, dx)
            radius = np.sqrt(dx**2 + dy**2)
            
            if radius < size//3 and radius > 5:
                spiral_condition = np.sin(angle * 3 + radius * 0.2) > 0.3
                if spiral_condition:
                    galaxy[i, j] = 1.0
    
    return galaxy

if __name__ == "__main__":
    test_kernel_comparison()