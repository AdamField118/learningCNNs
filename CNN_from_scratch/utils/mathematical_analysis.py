"""
Mathematical analysis utilities for CNN operations.

Centralized functions for dimension calculations, performance analysis, etc.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Union

def calculate_conv_output_size(
    input_size: int, 
    kernel_size: int, 
    stride: int = 1, 
    padding: int = 0, 
    dilation: int = 1
) -> int:
    """
    Calculate convolution output dimensions.
    
    Args:
        input_size: Size of input dimension
        kernel_size: Size of kernel dimension
        stride: Convolution stride
        padding: Amount of padding
        dilation: Dilation factor
        
    Returns:
        Output size after convolution
    """
    effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
    return (input_size + 2 * padding - effective_kernel) // stride + 1

def calculate_receptive_field_multilayer(
    kernel_sizes: List[int], 
    strides: List[int], 
    dilations: List[int] = None
) -> List[int]:
    """
    Calculate receptive field through multiple CNN layers.
    
    Args:
        kernel_sizes: List of kernel sizes for each layer
        strides: List of stride values for each layer
        dilations: List of dilation values for each layer (optional)
        
    Returns:
        List of receptive field sizes after each layer
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

def analyze_computational_cost(
    input_shape: Tuple[int, int], 
    kernel_size: int, 
    stride: int = 1, 
    padding: int = 0
) -> Dict[str, Union[int, Tuple[int, int], float]]:
    """
    Analyze computational cost of convolution operations.
    
    Args:
        input_shape: (height, width) of input
        kernel_size: Size of convolution kernel
        stride: Convolution stride
        padding: Amount of padding
        
    Returns:
        Dictionary with operation count, output shape, and reduction factor
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

def print_dimension_analysis(
    test_cases: List[Tuple[int, int, int, int]], 
    title: str = "Output Dimension Analysis"
) -> None:
    """
    Print formatted table of dimension calculations.
    
    Args:
        test_cases: List of (input_size, kernel_size, stride, padding) tuples
        title: Title for the analysis table
    """
    from utils.output_system import log_print
    
    log_print(f"=== {title} ===", level="SUBHEADER")
    log_print("Input Size | Kernel | Stride | Padding | Output Size", level="CODE")
    log_print("-" * 55, level="CODE")
    
    for input_size, kernel_size, stride, padding in test_cases:
        output_size = calculate_conv_output_size(input_size, kernel_size, stride, padding)
        log_print(f"{input_size:9d} | {kernel_size:6d} | {stride:6d} | {padding:7d} | {output_size:11d}", level="CODE")

def print_cost_analysis(
    configs: List[Tuple[str, Tuple[Tuple[int, int], int, int, int]]], 
    title: str = "Computational Cost Analysis"
) -> None:
    """
    Print formatted table of computational cost comparisons.
    
    Args:
        configs: List of (name, config) tuples where config is 
                (input_shape, kernel_size, stride, padding)
        title: Title for the analysis table
    """
    from utils.output_system import log_print
    
    log_print(f"=== {title} ===", level="SUBHEADER")
    log_print("Configuration | Operations | Output Shape | Relative Cost", level="CODE")
    log_print("-" * 60, level="CODE")
    
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
            
        log_print(f"{name:20s} | {ops:10,d} | {output_shape!s:12s} | {relative:7.2f}x", level="CODE")

def calculate_parameter_count(
    layer_configs: List[Dict[str, Any]]
) -> Dict[str, int]:
    """
    Calculate parameter count for CNN architecture.
    
    Args:
        layer_configs: List of layer configuration dictionaries
            Each dict should have: 'type', 'kernel_size', 'in_channels', 'out_channels'
            
    Returns:
        Dictionary with parameter counts per layer and total
    """
    param_counts = {}
    total_params = 0
    
    for i, config in enumerate(layer_configs):
        layer_name = f"layer_{i+1}_{config['type']}"
        
        if config['type'] == 'conv':
            # Conv layer: kernel_size^2 * in_channels * out_channels + out_channels (bias)
            kernel_params = (config['kernel_size'] ** 2) * config['in_channels'] * config['out_channels']
            bias_params = config['out_channels']
            layer_params = kernel_params + bias_params
            
        elif config['type'] == 'fc':
            # Fully connected: in_features * out_features + out_features (bias)
            layer_params = config['in_features'] * config['out_features'] + config['out_features']
            
        else:
            layer_params = 0  # Pooling, activation layers have no parameters
        
        param_counts[layer_name] = layer_params
        total_params += layer_params
    
    param_counts['total'] = total_params
    return param_counts

def analyze_memory_usage(
    input_shape: Tuple[int, int, int],  # (channels, height, width)
    layer_configs: List[Dict[str, Any]],
    batch_size: int = 1
) -> Dict[str, Dict[str, Union[int, float]]]:
    """
    Analyze memory usage through CNN layers.
    
    Args:
        input_shape: Input tensor shape (C, H, W)
        layer_configs: List of layer configurations
        batch_size: Batch size for memory calculation
        
    Returns:
        Dictionary with memory usage statistics per layer
    """
    memory_analysis = {}
    current_shape = input_shape
    
    for i, config in enumerate(layer_configs):
        layer_name = f"layer_{i+1}_{config['type']}"
        
        if config['type'] == 'conv':
            c, h, w = current_shape
            new_h = calculate_conv_output_size(h, config['kernel_size'], 
                                             config.get('stride', 1), 
                                             config.get('padding', 0))
            new_w = calculate_conv_output_size(w, config['kernel_size'], 
                                             config.get('stride', 1), 
                                             config.get('padding', 0))
            current_shape = (config['out_channels'], new_h, new_w)
            
        elif config['type'] == 'pool':
            c, h, w = current_shape
            new_h = h // config.get('pool_size', 2)
            new_w = w // config.get('pool_size', 2)
            current_shape = (c, new_h, new_w)
        
        # Calculate memory usage (assuming float32 = 4 bytes)
        elements = batch_size * np.prod(current_shape)
        memory_mb = (elements * 4) / (1024 * 1024)
        
        memory_analysis[layer_name] = {
            'shape': current_shape,
            'elements': elements,
            'memory_mb': memory_mb
        }
    
    return memory_analysis

def compare_architectures(
    architectures: Dict[str, List[Dict[str, Any]]],
    input_shape: Tuple[int, int, int] = (3, 224, 224)
) -> None:
    """
    Compare different CNN architectures in terms of parameters and memory.
    
    Args:
        architectures: Dictionary mapping architecture names to layer configs
        input_shape: Input shape for comparison
    """
    from utils.output_system import log_print
    
    log_print("=== Architecture Comparison ===", level="SUBHEADER")
    log_print("Architecture | Parameters | Memory (MB) | Depth", level="CODE")
    log_print("-" * 50, level="CODE")
    
    for arch_name, layers in architectures.items():
        params = calculate_parameter_count(layers)
        memory = analyze_memory_usage(input_shape, layers)
        
        total_params = params['total']
        total_memory = sum(layer['memory_mb'] for layer in memory.values())
        depth = len(layers)
        
        log_print(f"{arch_name:12s} | {total_params:10,d} | {total_memory:10.2f} | {depth:5d}", level="CODE")