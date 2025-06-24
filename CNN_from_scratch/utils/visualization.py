"""
Visualization utilities for CNN learning and analysis.

Centralized plotting functions with consistent styling - no more overlaps!
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_activation_comparison(x_values, activations_dict, title="Activation Function Comparison"):
    """
    Plot activation function comparison.
    
    Moved from neural_network_foundations.py (was compare_activation_functions)
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
    plt.show()

def plot_convolution_demo(image, kernel, result_v, result_h, title="Convolution Demo"):
    """
    Standard convolution demonstration plot.
    
    Moved from edge_detection_basics.py (was visualize_convolution)
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
    plt.show()

def plot_feature_responses(original, responses_dict, title="Feature Response Analysis"):
    """
    Plot original + multiple feature responses with PERFECT spacing.
    
    Moved from kernel_experiment.py (was visualize_kernel_comparison)
    FIXES the title overlap issues!
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
    
    # PERFECT spacing - no more overlaps!
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.95, top=0.85,
        wspace=0.3, hspace=0.5
    )
    plt.suptitle(title, fontsize=16)
    plt.show()

def plot_convolution_mechanics(original, results_dict, effect_type="Convolution Effects"):
    """
    Generic plotter for stride/padding/dilation effects.
    
    Consolidates visualize_stride_effects, visualize_padding_effects, visualize_dilation_effects
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
    plt.show()