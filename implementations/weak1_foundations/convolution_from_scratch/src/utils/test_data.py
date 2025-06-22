"""
Test data generation utilities for CNN learning exercises.

Centralized test data generators - one source of truth for all experiments!
"""

import numpy as np

def create_synthetic_galaxy(size=50, spiral_arms=True, add_noise=False, noise_level=0.1):
    """
    Create synthetic galaxy with customizable features.
    
    THE canonical galaxy generator for all your experiments.
    Consolidates versions from kernel_experiment.py and galaxy_edge_detection.ipynb
    """
    center = size // 2
    galaxy = np.zeros((size, size))
    
    # Create circular galaxy base
    y, x = np.ogrid[:size, :size]
    mask = (x - center)**2 + (y - center)**2 <= (size//3)**2
    galaxy[mask] = 0.5
    
    # Add spiral arms if requested
    if spiral_arms:
        for i in range(size):
            for j in range(size):
                dx, dy = j - center, i - center
                angle = np.arctan2(dy, dx)
                radius = np.sqrt(dx**2 + dy**2)
                
                if radius < size//3 and radius > 5:
                    spiral_condition = np.sin(angle * 3 + radius * 0.2) > 0.3
                    if spiral_condition:
                        galaxy[i, j] = 1.0
    
    # Add noise if requested
    if add_noise:
        noise = np.random.normal(0, noise_level, galaxy.shape)
        galaxy = galaxy + noise
        galaxy = np.clip(galaxy, 0, 1)
    
    return galaxy

def create_test_pattern(height, width, pattern_type='mixed'):
    """
    Create test patterns for convolution experiments.
    
    Moved from convolution_mechanics.py
    """
    pattern = np.zeros((height, width))
    center_h, center_w = height // 2, width // 2
    
    if pattern_type == 'mixed':
        # Vertical line
        pattern[:, center_w] = 1
        # Horizontal line  
        pattern[center_h, :] = 1
        # Diagonal corners
        for i in range(min(height//4, width//4)):
            if i < height and i < width:
                pattern[i, i] = 0.5
                pattern[height-1-i, width-1-i] = 0.5
                
    elif pattern_type == 'vertical_line':
        pattern[:, center_w] = 1
        
    elif pattern_type == 'horizontal_line':
        pattern[center_h, :] = 1
        
    elif pattern_type == 'diagonal':
        np.fill_diagonal(pattern, 1)
        
    elif pattern_type == 'corner':
        pattern[:height//2, :width//2] = 1
        
    return pattern

def create_simple_test_image():
    """
    Create simple test image with vertical line.
    
    Moved from edge_detection_basics.py (was create_test_image)
    """
    image = np.zeros((7, 7))
    image[:, 3] = 1  # Vertical line in middle
    return image

def create_noise_variants(base_image, noise_levels=[0.01, 0.05, 0.1, 0.2]):
    """Create multiple noise variants of the same image."""
    variants = {'original': base_image}
    
    for noise_level in noise_levels:
        noise = np.random.normal(0, noise_level, base_image.shape)
        noisy = base_image + noise
        noisy = np.clip(noisy, 0, 1)
        variants[f'noise_{noise_level}'] = noisy
    
    return variants