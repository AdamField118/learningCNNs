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

from utils.test_data import create_synthetic_galaxy
from utils.visualization import plot_feature_responses
from utils.output_system import log_print, log_experiment_start, log_experiment_end
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
    
    galaxy = create_synthetic_galaxy()
    
    fine_response = manual_convolution_2d(galaxy, create_scale_sensitive_kernels('fine'))
    medium_response = manual_convolution_2d(galaxy, create_scale_sensitive_kernels('medium'))
    large_response = manual_convolution_2d(galaxy, create_scale_sensitive_kernels('large'))
    
    sobel_x_response = manual_convolution_2d(galaxy, sobel_x_kernel())
    fine_custom_response = manual_convolution_2d(galaxy, create_scale_sensitive_kernels('fine'))
    
    horizontal_response = manual_convolution_2d(galaxy, create_orientation_sensitive_kernels('fine', 'horizontal'))
    diagonal_response = manual_convolution_2d(galaxy, create_orientation_sensitive_kernels('fine', 'diagonal_1'))
    
    gentle_x_response = manual_convolution_2d(galaxy, create_brightness_gradient_kernels('gentle_x'))
    radial_response = manual_convolution_2d(galaxy, create_brightness_gradient_kernels('radial'))
    
    plot_feature_responses(galaxy, {
        'Sobel X (Standard)': sobel_x_response,
        'Fine Scale (Custom)': fine_custom_response,
        'Medium Scale': medium_response,
        'Diagonal Orientation': diagonal_response,
        'Gentle Gradient': gentle_x_response,
        'Radial Gradient': radial_response
    }, title="Astronomical Kernel Comparison")

if __name__ == "__main__":
    log_experiment_start(3, "kernel designs in galaxy images")
    
    test_kernel_comparison()
    
    log_experiment_end(3)