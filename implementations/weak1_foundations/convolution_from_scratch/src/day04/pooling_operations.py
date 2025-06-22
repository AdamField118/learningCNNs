"""
Pooling operations and their effects on feature maps.

This module explores how pooling affects spatial precision vs translation 
invariance - crucial for understanding CNN architecture trade-offs.
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
from day03.convolution_mechanics import manual_convolution_2d_extended

def max_pooling_2d(feature_map, pool_size=2, stride=None):
    """
    Implement max pooling using convolution windowing logic.
    
    Args:
        feature_map: 2D numpy array (output from convolution)
        pool_size: Size of pooling window (default 2x2)
        stride: Stride for pooling (default: same as pool_size)
    
    Returns:
        Pooled feature map (smaller spatial dimensions)
    """
    if stride is None:
        stride = pool_size
    
    # Create a dummy kernel of ones - we'll ignore the actual convolution math
    # and just use the windowing logic from manual_convolution_2d_extended
    dummy_kernel = np.ones((pool_size, pool_size))
    
    # Get dimensions for output calculation (same as convolution)
    input_h, input_w = feature_map.shape
    output_h = (input_h - pool_size) // stride + 1
    output_w = (input_w - pool_size) // stride + 1
    output = np.zeros((output_h, output_w))
    
    # Use the same windowing logic as convolution, but apply max instead
    for i in range(output_h):
        for j in range(output_w):
            start_i = i * stride
            start_j = j * stride
            
            # Extract the window (this is what convolution iterates over)
            window = feature_map[start_i:start_i + pool_size, 
                               start_j:start_j + pool_size]
            
            # Apply max operation instead of convolution
            output[i, j] = np.max(window)
    
    return output

def average_pooling_2d(feature_map, pool_size=2, stride=None):
    """
    Implement average pooling using convolution - this one's even cleaner!
    
    Average pooling is literally just convolution with a kernel of all 1s,
    followed by division by kernel size.
    """
    if stride is None:
        stride = pool_size
    
    # Create kernel of all ones
    ones_kernel = np.ones((pool_size, pool_size))
    
    # Use existing convolution function!
    conv_result = manual_convolution_2d_extended(
        feature_map, 
        ones_kernel, 
        stride=stride, 
        padding=0
    )
    
    # Divide by kernel size to get average
    return conv_result / (pool_size * pool_size)

if __name__ == "__main__":
    # Test with a simple feature map first
    test_feature_map = np.array([[1, 3, 2, 4],
                                [5, 6, 1, 2], 
                                [3, 2, 4, 1],
                                [1, 1, 3, 5]])
    
    print("Original feature map:")
    print(test_feature_map)
    
    # Test your implementations
    max_result = max_pooling_2d(test_feature_map, pool_size=2)
    avg_result = average_pooling_2d(test_feature_map, pool_size=2)
    
    print("\nMax pooled (2x2):")
    print(max_result)
    print("\nAverage pooled (2x2):")
    print(avg_result)