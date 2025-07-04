"""
Edge detection kernels for convolution experiments.

This module contains standard edge detection kernels used in computer vision
and customized kernels for astronomical applications.
"""

import numpy as np
from typing import Tuple

def sobel_x_kernel() -> np.ndarray:
    """
    Sobel kernel for detecting horizontal edges (vertical gradients).
    
    Returns:
        3x3 numpy array representing the Sobel X kernel
        
    Notes:
        This kernel detects vertical edges by computing horizontal gradients.
        Positive responses indicate edges where intensity increases from left to right.
    """
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)

def sobel_y_kernel() -> np.ndarray:
    """
    Sobel kernel for detecting vertical edges (horizontal gradients).
    
    Returns:
        3x3 numpy array representing the Sobel Y kernel
    """
    return np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float32)

def prewitt_x_kernel() -> np.ndarray:
    """
    Prewitt kernel for horizontal edge detection.
    
    Returns:
        3x3 numpy array representing the Prewitt X kernel
    """
    return np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]], dtype=np.float32)

def prewitt_y_kernel() -> np.ndarray:
    """
    Prewitt kernel for vertical edge detection.
    
    Returns:
        3x3 numpy array representing the Prewitt Y kernel
    """
    return np.array([[-1, -1, -1],
                     [ 0,  0,  0],
                     [ 1,  1,  1]], dtype=np.float32)

def laplacian_kernel() -> np.ndarray:
    """
    Laplacian kernel for edge detection (detects all edge orientations).
    
    Returns:
        3x3 numpy array representing the Laplacian kernel
        
    Notes:
        This kernel detects edges in all directions by computing the second derivative.
        Good for finding where intensity changes rapidly in any direction.
    """
    return np.array([[ 0, -1,  0],
                     [-1,  4, -1],
                     [ 0, -1,  0]], dtype=np.float32)

def get_all_edge_kernels() -> dict[str, np.ndarray]:
    """
    Get all available edge detection kernels.
    
    Returns:
        Dictionary mapping kernel names to kernel arrays
    """
    return {
        'sobel_x': sobel_x_kernel(),
        'sobel_y': sobel_y_kernel(),
        'prewitt_x': prewitt_x_kernel(),
        'prewitt_y': prewitt_y_kernel(),
        'laplacian': laplacian_kernel()
    }