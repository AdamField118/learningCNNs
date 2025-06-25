"""
Test data generation utilities for CNN learning exercises with type hints.

Centralized test data generators - one source of truth for all experiments!
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union

def create_synthetic_galaxy(
    size: int = 50, 
    spiral_arms: bool = True, 
    add_noise: bool = False, 
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Create synthetic galaxy with customizable features.
    
    THE canonical galaxy generator for all your experiments.
    Consolidates versions from kernel_experiment.py and galaxy_edge_detection.ipynb
    
    Args:
        size: Size of the square galaxy image
        spiral_arms: Whether to include spiral arm structure
        add_noise: Whether to add Gaussian noise
        noise_level: Standard deviation of noise (if add_noise=True)
        
    Returns:
        2D numpy array representing the synthetic galaxy
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

def create_test_pattern(
    height: int, 
    width: int, 
    pattern_type: str = 'mixed'
) -> np.ndarray:
    """
    Create test patterns for convolution experiments.
    
    Args:
        height: Height of the test pattern
        width: Width of the test pattern
        pattern_type: Type of pattern to create
            - 'mixed': Vertical and horizontal lines with diagonal corners
            - 'vertical_line': Single vertical line
            - 'horizontal_line': Single horizontal line
            - 'diagonal': Diagonal line from corner to corner
            - 'corner': Filled corner rectangle
            
    Returns:
        2D numpy array with the specified test pattern
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
        
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")
    
    return pattern

def create_simple_test_image() -> np.ndarray:
    """
    Create simple test image with vertical line.
    
    Returns:
        7x7 numpy array with a vertical line in the middle
    """
    image = np.zeros((7, 7))
    image[:, 3] = 1  # Vertical line in middle
    return image

def create_noise_variants(
    base_image: np.ndarray, 
    noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]
) -> Dict[str, np.ndarray]:
    """
    Create multiple noise variants of the same image.
    
    Args:
        base_image: Original image to add noise to
        noise_levels: List of noise standard deviations to apply
        
    Returns:
        Dictionary mapping noise descriptions to noisy images
    """
    variants = {'original': base_image}
    
    for noise_level in noise_levels:
        noise = np.random.normal(0, noise_level, base_image.shape)
        noisy = base_image + noise
        noisy = np.clip(noisy, 0, 1)
        variants[f'noise_{noise_level}'] = noisy
    
    return variants

def create_galaxy_batch(
    batch_size: int = 8,
    galaxy_size: int = 32,
    galaxy_types: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Create a batch of galaxies with different characteristics.
    
    Args:
        batch_size: Number of galaxies to create
        galaxy_size: Size of each galaxy image
        galaxy_types: List of galaxy types to create. If None, uses default mix.
        
    Returns:
        Tuple of (galaxy_batch, galaxy_type_labels)
    """
    if galaxy_types is None:
        galaxy_types = ['spiral', 'elliptical', 'noisy'] * (batch_size // 3 + 1)
        galaxy_types = galaxy_types[:batch_size]
    
    galaxies = []
    actual_types = []
    
    for i, galaxy_type in enumerate(galaxy_types[:batch_size]):
        if galaxy_type == 'spiral':
            galaxy = create_synthetic_galaxy(
                size=galaxy_size, 
                spiral_arms=True, 
                add_noise=False
            )
            # Vary brightness
            galaxy = galaxy * (1.5 + 0.5 * np.random.randn())
            
        elif galaxy_type == 'elliptical':
            galaxy = create_synthetic_galaxy(
                size=galaxy_size, 
                spiral_arms=False, 
                add_noise=True
            )
            # Make fainter
            galaxy = galaxy * (0.5 + 0.2 * np.random.randn())
            
        elif galaxy_type == 'noisy':
            galaxy = create_synthetic_galaxy(
                size=galaxy_size, 
                spiral_arms=True, 
                add_noise=True,
                noise_level=0.2
            )
            # Very faint
            galaxy = galaxy * (0.3 + 0.1 * np.random.randn())
            
        else:
            raise ValueError(f"Unknown galaxy_type: {galaxy_type}")
        
        # Ensure non-negative values
        galaxy = np.clip(galaxy, 0, None)
        galaxies.append(galaxy)
        actual_types.append(galaxy_type)
    
    return np.stack(galaxies, axis=0), actual_types

def create_edge_test_cases() -> Dict[str, np.ndarray]:
    """
    Create a set of test cases specifically for edge detection experiments.
    
    Returns:
        Dictionary mapping test case names to test images
    """
    test_cases = {}
    
    # Vertical edges
    test_cases['vertical_edge'] = create_test_pattern(20, 20, 'vertical_line')
    
    # Horizontal edges
    test_cases['horizontal_edge'] = create_test_pattern(20, 20, 'horizontal_line')
    
    # Diagonal edges
    test_cases['diagonal_edge'] = create_test_pattern(20, 20, 'diagonal')
    
    # Mixed patterns
    test_cases['mixed_pattern'] = create_test_pattern(20, 20, 'mixed')
    
    # Step edge (sharp transition)
    step_edge = np.zeros((20, 20))
    step_edge[:, :10] = 0.3
    step_edge[:, 10:] = 0.8
    test_cases['step_edge'] = step_edge
    
    # Gradient edge (smooth transition)
    gradient_edge = np.zeros((20, 20))
    for i in range(20):
        gradient_edge[:, i] = i / 20.0
    test_cases['gradient_edge'] = gradient_edge
    
    # Circular object
    circular = np.zeros((20, 20))
    center = 10
    y, x = np.ogrid[:20, :20]
    mask = (x - center)**2 + (y - center)**2 <= 6**2
    circular[mask] = 1.0
    test_cases['circular_object'] = circular
    
    return test_cases

def create_convolution_test_suite() -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """
    Create comprehensive test suite for convolution parameter experiments.
    
    Returns:
        Dictionary mapping test names to test configurations
    """
    test_suite = {}
    
    # Base test image
    base_image = create_synthetic_galaxy(size=40, spiral_arms=True)
    
    # Stride tests
    for stride in [1, 2, 3]:
        test_suite[f'stride_{stride}'] = {
            'image': base_image,
            'stride': stride,
            'padding': 0,
            'dilation': 1
        }
    
    # Padding tests
    for padding in [0, 1, 2, 3]:
        test_suite[f'padding_{padding}'] = {
            'image': base_image,
            'stride': 1,
            'padding': padding,
            'dilation': 1
        }
    
    # Dilation tests
    for dilation in [1, 2, 3]:
        test_suite[f'dilation_{dilation}'] = {
            'image': base_image,
            'stride': 1,
            'padding': 1,
            'dilation': dilation
        }
    
    return test_suite