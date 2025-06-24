"""
Utilities package for CNN learning exercises.

Centralized functions that eliminate code duplication and provide
consistent interfaces across all experiments.
"""

# Test data generation
from .test_data import (
    create_synthetic_galaxy,
    create_test_pattern, 
    create_simple_test_image,
    create_noise_variants
)

# Visualization with proper spacing
from .visualization import (
    plot_activation_comparison,
    plot_convolution_demo,
    plot_feature_responses,
    plot_convolution_mechanics
)

# Mathematical analysis
from .mathematical_analysis import (
    calculate_conv_output_size,
    calculate_receptive_field_multilayer,
    analyze_computational_cost,
    print_dimension_analysis,
    print_cost_analysis
)