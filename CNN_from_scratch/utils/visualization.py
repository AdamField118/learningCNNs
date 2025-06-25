"""
Visualization utilities for CNN learning and analysis.

Centralized plotting functions with consistent styling - no more overlaps!
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import os
import sys

# Import our output system
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.output_system import save_plot, log_print

def plot_activation_comparison(
    x_values: np.ndarray, 
    activations_dict: Dict[str, np.ndarray], 
    title: str = "Activation Function Comparison",
    filename_prefix: str = "activation_comparison"
) -> str:
    """
    Plot activation function comparison and save to file.
    
    Args:
        x_values: Input values for activation functions
        activations_dict: Dictionary mapping activation names to their outputs
        title: Plot title
        filename_prefix: Prefix for saved filename
        
    Returns:
        Path to saved plot file
    """
    n_funcs = len(activations_dict)
    fig, axes = plt.subplots(1, n_funcs + 1, figsize=(4 * (n_funcs + 1), 4))
    
    # Individual plots
    for i, (name, values) in enumerate(activations_dict.items()):
        axes[i].plot(x_values, values, linewidth=2)
        axes[i].set_title(f'{name} Activation')
        axes[i].set_xlabel('Input')
        axes[i].set_ylabel('Output')
        axes[i].grid(True, alpha=0.3)
    
    # Comparison plot
    for name, values in activations_dict.items():
        axes[-1].plot(x_values, values, linewidth=2, label=name)
    axes[-1].set_title('Comparison')
    axes[-1].set_xlabel('Input')
    axes[-1].set_ylabel('Output')
    axes[-1].legend()
    axes[-1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    filename = f"{filename_prefix}.png"
    return save_plot(filename)

def plot_convolution_demo(
    image: np.ndarray, 
    kernel: np.ndarray, 
    result_v: np.ndarray, 
    result_h: np.ndarray, 
    title: str = "Convolution Demo",
    filename_prefix: str = "convolution_demo"
) -> str:
    """
    Standard convolution demonstration plot.
    
    Args:
        image: Original input image
        kernel: Convolution kernel used
        result_v: Vertical detection result
        result_h: Horizontal detection result
        title: Plot title
        filename_prefix: Prefix for saved filename
        
    Returns:
        Path to saved plot file
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image\n(Test Pattern)')
    plt.colorbar()
    
    plt.subplot(1, 4, 2)
    kernel_max = np.max(np.abs(kernel))
    plt.imshow(kernel, cmap='RdBu', vmin=-kernel_max, vmax=kernel_max)
    plt.title('Kernel')
    plt.colorbar()
    
    plt.subplot(1, 4, 3)
    plt.imshow(result_v, cmap='gray')
    plt.title('Vertical Detection')
    plt.colorbar()
    
    plt.subplot(1, 4, 4)
    plt.imshow(result_h, cmap='gray')
    plt.title('Horizontal Detection')
    plt.colorbar()
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    filename = f"{filename_prefix}.png"
    return save_plot(filename)

def plot_feature_responses(
    original: np.ndarray, 
    responses_dict: Dict[str, np.ndarray], 
    title: str = "Feature Response Analysis",
    filename_prefix: str = "feature_responses"
) -> str:
    """
    Plot original + multiple feature responses with perfect spacing.
    
    Args:
        original: Original input image
        responses_dict: Dictionary mapping response names to feature maps
        title: Plot title
        filename_prefix: Prefix for saved filename
        
    Returns:
        Path to saved plot file
    """
    n_plots = len(responses_dict) + 1
    cols = 3
    rows = (n_plots + cols - 1) // cols
    
    # Much larger figure with explicit spacing
    fig, axes = plt.subplots(rows, cols, figsize=(18, 8 * rows))
    
    # Handle single row case
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easy indexing
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(original, cmap='hot')
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    
    # Feature responses
    for i, (name, response) in enumerate(responses_dict.items(), 1):
        if i < len(axes):
            im = axes[i].imshow(response, cmap='gray')
            plt.colorbar(im, ax=axes[i], shrink=0.8)
            
            # Clean up long names
            if name != 'Original':
                max_response = np.max(np.abs(response))
                short_names = {
                    'Sobel X (Standard)': 'Sobel X',
                    'Fine Scale (Custom)': 'Fine Scale', 
                    'Medium Scale': 'Medium',
                    'Diagonal Orientation': 'Diagonal',
                    'Gentle Gradient': 'Gentle',
                    'Radial Gradient': 'Radial'
                }
                display_name = short_names.get(name, name)
                axes[i].set_title(f'{display_name}\n{max_response:.1f}', fontsize=12)
            else:
                axes[i].set_title(name, fontsize=14)
    
    # Hide unused subplots
    for j in range(len(responses_dict) + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Perfect spacing - no more overlaps!
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.95, top=0.85,
        wspace=0.3, hspace=0.5
    )
    plt.suptitle(title, fontsize=16)
    
    filename = f"{filename_prefix}.png"
    return save_plot(filename)

def plot_convolution_mechanics(
    original: np.ndarray, 
    results_dict: Dict[str, np.ndarray], 
    effect_type: str = "Convolution Effects",
    filename_prefix: str = "convolution_mechanics"
) -> str:
    """
    Generic plotter for stride/padding/dilation effects.
    
    Args:
        original: Original input image
        results_dict: Dictionary mapping parameter names to results
        effect_type: Type of effect being demonstrated
        filename_prefix: Prefix for saved filename
        
    Returns:
        Path to saved plot file
    """
    n_results = len(results_dict)
    fig, axes = plt.subplots(2, n_results, figsize=(4 * n_results, 8))
    
    if n_results == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (key, result) in enumerate(results_dict.items()):
        # Top row: show parameter effect
        axes[0, idx].imshow(original, cmap='gray', interpolation='nearest')
        axes[0, idx].set_title(f'Original\n{original.shape}')
        axes[0, idx].axis('off')
        
        # Bottom row: show result
        im = axes[1, idx].imshow(result, cmap='RdBu', interpolation='nearest')
        param_value = key.split('_')[1]
        axes[1, idx].set_title(f'{effect_type}\nParam: {param_value}\nOutput: {result.shape}')
        axes[1, idx].axis('off')
        plt.colorbar(im, ax=axes[1, idx], fraction=0.046)
    
    plt.tight_layout()
    
    filename = f"{filename_prefix}_{effect_type.lower().replace(' ', '_')}.png"
    return save_plot(filename)

def plot_training_curves(
    curves_dict: Dict[str, List[float]], 
    title: str = "Training Curves",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    filename_prefix: str = "training_curves"
) -> str:
    """
    Plot training curves comparison.
    
    Args:
        curves_dict: Dictionary mapping curve names to loss values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        filename_prefix: Prefix for saved filename
        
    Returns:
        Path to saved plot file
    """
    plt.figure(figsize=(10, 6))
    
    for name, values in curves_dict.items():
        plt.plot(values, linewidth=2, label=name)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"{filename_prefix}.png"
    return save_plot(filename)

def plot_gradient_analysis(
    gradients_dict: Dict[str, np.ndarray],
    title: str = "Gradient Analysis",
    filename_prefix: str = "gradient_analysis"
) -> str:
    """
    Plot gradient flow analysis.
    
    Args:
        gradients_dict: Dictionary mapping method names to gradient values
        title: Plot title
        filename_prefix: Prefix for saved filename
        
    Returns:
        Path to saved plot file
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot gradient flow over layers
    axes[0].set_title('Gradient Flow Through Layers')
    for name, gradients in gradients_dict.items():
        axes[0].plot(gradients, linewidth=2, label=name)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Gradient Magnitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot final gradient comparison
    final_gradients = [gradients[-1] for gradients in gradients_dict.values()]
    names = list(gradients_dict.keys())
    
    axes[1].bar(names, final_gradients, alpha=0.7)
    axes[1].set_title('Final Gradient Comparison')
    axes[1].set_ylabel('Final Gradient Magnitude')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_yscale('log')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    filename = f"{filename_prefix}.png"
    return save_plot(filename)