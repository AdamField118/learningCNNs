"""
Mathematical analysis utilities for CNN operations.

Centralized functions for dimension calculations, performance analysis, etc.
"""

import numpy as np

def calculate_conv_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    """
    Calculate convolution output dimensions.
    
    Moved from convolution_mechanics.py
    """
    effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
    return (input_size + 2 * padding - effective_kernel) // stride + 1

def calculate_receptive_field_multilayer(kernel_sizes, strides, dilations=None):
    """
    Calculate receptive field through multiple CNN layers.
    
    Moved from convolution_mechanics.py
    """
    if dilations is None:
        dilations = [1] * len(kernel_sizes)
    
    rf = 1
    jump = 1
    rf_progression = [rf]
    
    for kernel_size, stride, dilation in zip(kernel_sizes, strides, dilations):
        effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
        rf = rf + (effective_kernel - 1) * jump
        jump = jump * stride
        rf_progression.append(rf)
    
    return rf_progression

def analyze_computational_cost(input_shape, kernel_size, stride=1, padding=0):
    """
    Analyze computational cost of convolution operations.
    
    Moved from convolution_mechanics.py
    """
    h, w = input_shape
    output_h = calculate_conv_output_size(h, kernel_size, stride, padding)
    output_w = calculate_conv_output_size(w, kernel_size, stride, padding)
    
    total_ops = output_h * output_w * kernel_size * kernel_size
    return {
        'operations': total_ops,
        'output_shape': (output_h, output_w),
        'reduction_factor': (h * w) / (output_h * output_w)
    }

def print_dimension_analysis(test_cases, title="Output Dimension Analysis"):
    """Print formatted table of dimension calculations."""
    print(f"=== {title} ===")
    print("Input Size | Kernel | Stride | Padding | Output Size")
    print("-" * 55)
    
    for input_size, kernel_size, stride, padding in test_cases:
        output_size = calculate_conv_output_size(input_size, kernel_size, stride, padding)
        print(f"{input_size:9d} | {kernel_size:6d} | {stride:6d} | {padding:7d} | {output_size:11d}")

def print_cost_analysis(configs, title="Computational Cost Analysis"):
    """Print formatted table of computational cost comparisons."""
    print(f"=== {title} ===")
    print("Configuration | Operations | Output Shape | Relative Cost")
    print("-" * 60)
    
    base_ops = None
    for name, config in configs:
        ops_data = analyze_computational_cost(*config)
        ops = ops_data['operations']
        output_shape = ops_data['output_shape']
        
        if base_ops is None:
            base_ops = ops
            relative = 1.0
        else:
            relative = ops / base_ops
            
        print(f"{name:20s} | {ops:10,d} | {output_shape!s:12s} | {relative:7.2f}x")