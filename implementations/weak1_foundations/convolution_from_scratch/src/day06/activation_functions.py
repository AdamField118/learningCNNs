"""
Modern activation functions and their properties.

This module explores activation functions beyond ReLU and demonstrates
their impact on galaxy feature detection and gradient flow.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.test_data import create_synthetic_galaxy
from day01.neural_network_foundations import activation_function_relu, activation_function_sigmoid

class ModernActivations:
    """
    Collection of modern activation functions.
    
    Each activation has different properties that can improve training
    and performance for specific tasks like galaxy analysis.
    """
    
    @staticmethod
    def swish(x, beta=1.0):
        """
        Swish activation: x * sigmoid(beta * x)
        
        Properties:
        - Smooth (differentiable everywhere)
        - Self-gating (uses its own value to gate)
        - Non-monotonic (can decrease then increase)
        
        Args:
            x: Input values
            beta: Scaling parameter (default 1.0)
        """
        # Implement Swish activation using the imported sigmoid function
        # Formula: x * sigmoid(beta * x)
        return x * activation_function_sigmoid(beta * x)
    
    @staticmethod
    def gelu(x):
        """
        GELU (Gaussian Error Linear Unit): x * Φ(x)
        Where Φ(x) is the CDF of standard normal distribution
        
        Properties:
        - Smooth activation
        - Probabilistic interpretation
        - Used in BERT, GPT, and modern transformers
        
        Args:
            x: Input values
        """
        # Implement GELU activation
        # Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        # This is faster than the exact erf implementation
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        inner = sqrt_2_over_pi * (x + 0.044715 * np.power(x, 3))
        return 0.5 * x * (1.0 + np.tanh(inner))
    
    @staticmethod
    def mish(x):
        """
        Mish activation: x * tanh(softplus(x))
        Where softplus(x) = ln(1 + exp(x))
        
        Properties:
        - Smooth and non-monotonic
        - Self-regularizing
        - Often outperforms ReLU and Swish
        
        Args:
            x: Input values
        """
        # Implement Mish activation
        # Formula: x * tanh(ln(1 + exp(x)))
        # Use np.log1p(np.exp(x)) for numerical stability when x is large
        # For numerical stability, use softplus implementation that handles large x
        softplus = np.where(x > 20, x, np.log1p(np.exp(x)))
        return x * np.tanh(softplus)
    
    @staticmethod
    def elu(x, alpha=1.0):
        """
        ELU (Exponential Linear Unit)
        
        Properties:
        - Smooth alternative to ReLU
        - Negative saturation allows negative information
        - Helps with vanishing gradient problem
        
        Args:
            x: Input values
            alpha: Controls negative saturation
        """
        # Implement ELU activation
        # Formula: x if x > 0, alpha * (exp(x) - 1) if x <= 0
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """
        Leaky ReLU: allows small negative slope
        
        Properties:
        - Fixes dying ReLU problem
        - Simple modification of ReLU
        - Non-zero gradient for negative inputs
        
        Args:
            x: Input values
            alpha: Slope for negative inputs
        """
        # Implement Leaky ReLU
        # Formula: max(alpha * x, x)
        return np.maximum(alpha * x, x)

def compare_activation_properties():
    """
    Compare mathematical properties of different activations.
    """
    print("=== Activation Function Properties Comparison ===")
    
    # Create input range for testing
    x = np.linspace(-3, 3, 1000)
    
    activations = ModernActivations()
    
    # Compute all activation functions
    relu_vals = activation_function_relu(x)
    sigmoid_vals = activation_function_sigmoid(x)
    swish_vals = activations.swish(x)
    gelu_vals = activations.gelu(x)
    mish_vals = activations.mish(x)
    elu_vals = activations.elu(x)
    leaky_relu_vals = activations.leaky_relu(x)
    
    # Compute gradients (finite differences)
    def compute_gradient(f, x_vals):
        """Compute numerical gradient of function f."""
        h = 1e-5
        grad = np.zeros_like(x_vals)
        for i in range(len(x_vals)):
            if i == 0:
                grad[i] = (f(x_vals[i+1]) - f(x_vals[i])) / (x_vals[i+1] - x_vals[i])
            elif i == len(x_vals) - 1:
                grad[i] = (f(x_vals[i]) - f(x_vals[i-1])) / (x_vals[i] - x_vals[i-1])
            else:
                grad[i] = (f(x_vals[i+1]) - f(x_vals[i-1])) / (x_vals[i+1] - x_vals[i-1])
        return grad
    
    # Compute gradients for each activation
    relu_grad = compute_gradient(activation_function_relu, x)
    swish_grad = compute_gradient(activations.swish, x)
    gelu_grad = compute_gradient(activations.gelu, x)
    mish_grad = compute_gradient(activations.mish, x)
    elu_grad = compute_gradient(activations.elu, x)
    leaky_relu_grad = compute_gradient(activations.leaky_relu, x)
    
    # Visualize activations and their gradients
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot activations - Set 1
    axes[0, 0].plot(x, relu_vals, 'r-', label='ReLU', linewidth=2)
    axes[0, 0].plot(x, swish_vals, 'g-', label='Swish', linewidth=2)
    axes[0, 0].plot(x, gelu_vals, 'b-', label='GELU', linewidth=2)
    axes[0, 0].plot(x, sigmoid_vals, 'purple', label='Sigmoid', linewidth=2)
    axes[0, 0].set_title('Classic vs Modern Activations')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel('Input')
    axes[0, 0].set_ylabel('Output')
    
    # Plot activations - Set 2
    axes[0, 1].plot(x, mish_vals, 'm-', label='Mish', linewidth=2)
    axes[0, 1].plot(x, elu_vals, 'c-', label='ELU', linewidth=2)
    axes[0, 1].plot(x, leaky_relu_vals, 'orange', label='Leaky ReLU', linewidth=2)
    axes[0, 1].plot(x, relu_vals, 'r--', label='ReLU (ref)', alpha=0.5, linewidth=1)
    axes[0, 1].set_title('Alternative Activations')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel('Input')
    axes[0, 1].set_ylabel('Output')
    
    # Plot gradients - shows smoothness and gradient preservation
    axes[1, 0].plot(x, relu_grad, 'r-', label='ReLU grad', linewidth=2)
    axes[1, 0].plot(x, swish_grad, 'g-', label='Swish grad', linewidth=2)
    axes[1, 0].plot(x, gelu_grad, 'b-', label='GELU grad', linewidth=2)
    axes[1, 0].set_title('Gradient Comparison: Smooth vs Non-smooth')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel('Input')
    axes[1, 0].set_ylabel('Gradient')
    
    axes[1, 1].plot(x, mish_grad, 'm-', label='Mish grad', linewidth=2)
    axes[1, 1].plot(x, elu_grad, 'c-', label='ELU grad', linewidth=2)
    axes[1, 1].plot(x, leaky_relu_grad, 'orange', label='Leaky ReLU grad', linewidth=2)
    axes[1, 1].set_title('Alternative Activation Gradients')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel('Input')
    axes[1, 1].set_ylabel('Gradient')
    
    # Analyze zero-centered property and negative information
    axes[2, 0].plot(x, swish_vals, 'g-', label='Swish', linewidth=2)
    axes[2, 0].plot(x, gelu_vals, 'b-', label='GELU', linewidth=2)
    axes[2, 0].plot(x, mish_vals, 'm-', label='Mish', linewidth=2)
    axes[2, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2, 0].axvline(x=0, color='k', linestyle='--', alpha=0.5)
    axes[2, 0].set_title('Non-monotonic Behavior (can decrease then increase)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel('Input')
    axes[2, 0].set_ylabel('Output')
    axes[2, 0].set_xlim(-2, 1)
    axes[2, 0].set_ylim(-0.5, 1)
    
    # Negative information preservation
    negative_inputs = x[x < 0]
    neg_info_data = {
        'ReLU': np.sum(activation_function_relu(negative_inputs) != 0),
        'Swish': np.sum(activations.swish(negative_inputs) != 0),
        'GELU': np.sum(activations.gelu(negative_inputs) != 0),
        'Mish': np.sum(activations.mish(negative_inputs) != 0),
        'ELU': np.sum(activations.elu(negative_inputs) != 0),
        'Leaky ReLU': np.sum(activations.leaky_relu(negative_inputs) != 0)
    }
    
    names = list(neg_info_data.keys())
    values = list(neg_info_data.values())
    colors = ['red', 'green', 'blue', 'magenta', 'cyan', 'orange']
    
    axes[2, 1].bar(names, values, color=colors, alpha=0.7)
    axes[2, 1].set_title('Negative Information Preservation')
    axes[2, 1].set_xlabel('Activation Function')
    axes[2, 1].set_ylabel('Non-zero outputs for negative inputs')
    axes[2, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze key properties
    print("\nKey Properties Analysis:")
    print("ReLU: Simple, fast, but can die (gradient=0 for x<0)")
    print("  - Zero gradient for negative inputs -> dying neurons")
    print("  - Not zero-centered -> inefficient learning")
    
    print("Swish: Smooth, self-gating, non-monotonic")
    print("  - Can decrease then increase (non-monotonic)")
    print("  - Smooth everywhere -> better gradients")
    print("  - Self-gating: uses own value to control activation")
    
    print("GELU: Smooth, probabilistic interpretation")
    print("  - Based on Gaussian CDF -> principled approach")
    print("  - Used in BERT, GPT -> proven in transformers")
    print("  - Smooth gradients help optimization")
    
    print("Mish: Smooth, non-monotonic, self-regularizing")
    print("  - Often empirically outperforms others")
    print("  - Strong negative information preservation")
    print("  - Self-regularizing properties")
    
    print("ELU: Smooth, negative saturation")
    print("  - Smooth everywhere unlike ReLU")
    print("  - Negative saturation prevents extreme negative values")
    print("  - Helps with vanishing gradients")
    
    print("Leaky ReLU: Simple fix for dying ReLU")
    print("  - Minimal computational overhead")
    print("  - Prevents dying neurons")
    print("  - Good baseline modern activation")

def test_activations_on_galaxy_features():
    """
    Test how different activations affect galaxy feature detection.
    """
    print("=== Activation Functions for Galaxy Feature Detection ===")
    
    galaxy = create_synthetic_galaxy(size=50, spiral_arms=True)
    
    # Apply edge detection using imported functions
    from kernels.edge_detection_kernels import sobel_x_kernel
    from day02.edge_detection_basics import manual_convolution_2d
    
    kernel = sobel_x_kernel()
    edge_features = manual_convolution_2d(galaxy, kernel)
    
    activations = ModernActivations()
    
    # Apply different activations to the edge features
    activated_features = {
        'Original': edge_features,
        'ReLU': activation_function_relu(edge_features),
        'Swish': activations.swish(edge_features),
        'GELU': activations.gelu(edge_features),
        'Mish': activations.mish(edge_features),
        'ELU': activations.elu(edge_features),
        'Leaky ReLU': activations.leaky_relu(edge_features)
    }
    
    # Analyze feature preservation
    print(f"Original edge features: min={edge_features.min():.3f}, max={edge_features.max():.3f}")
    print(f"Range: {edge_features.max() - edge_features.min():.3f}")
    print()
    
    for name, features in activated_features.items():
        if name == 'Original':
            continue
            
        feature_range = features.max() - features.min()
        negative_info = np.sum(features < 0) / features.size * 100
        strong_responses = np.sum(np.abs(features) > 2) / features.size * 100
        zero_responses = np.sum(features == 0) / features.size * 100
        mean_activation = np.mean(features)
        
        print(f"{name:12s}: range={feature_range:.3f}, negative_info={negative_info:.1f}%, " +
              f"strong_resp={strong_responses:.1f}%, zero_resp={zero_responses:.1f}%, mean={mean_activation:.3f}")
    
    # Visualize the differences
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    # Show original galaxy and edge features
    axes[0].imshow(galaxy, cmap='gray')
    axes[0].set_title('Original Galaxy')
    axes[0].axis('off')
    
    axes[1].imshow(edge_features, cmap='RdBu_r')
    axes[1].set_title('Edge Features (Sobel)')
    axes[1].axis('off')
    
    # Show activated features
    activation_names = ['ReLU', 'Swish', 'GELU', 'Mish', 'ELU', 'Leaky ReLU']
    for i, name in enumerate(activation_names):
        axes[i+2].imshow(activated_features[name], cmap='RdBu_r')
        axes[i+2].set_title(f'{name} Activation')
        axes[i+2].axis('off')
    
    # Hide the last subplot
    axes[8].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Detailed analysis for galaxy applications
    print("\nDetailed Analysis for Galaxy Applications:")
    
    # Check feature diversity (how many different activation levels)
    for name, features in activated_features.items():
        if name == 'Original':
            continue
        unique_vals = len(np.unique(np.round(features, 3)))
        total_vals = features.size
        diversity = unique_vals / total_vals * 100
        print(f"{name:12s}: Feature diversity: {diversity:.1f}% ({unique_vals}/{total_vals} unique values)")
    
    print("\nImplications for Galaxy Analysis:")
    print("- Higher negative info preservation → Better capture of galaxy structure variations")
    print("- Lower zero responses → More neurons stay active (avoid dying neurons)")
    print("- Smooth activations → Better gradient flow for precise shear measurements")
    print("- Non-monotonic functions → Can capture complex galaxy morphologies")
    
    return activated_features

def analyze_gradient_flow():
    """
    Analyze how different activations affect gradient flow.
    """
    print("=== Gradient Flow Analysis ===")
    
    # Simulate deep network gradient flow
    initial_gradient = 1.0
    num_layers = 15
    num_simulations = 100  # Multiple random initializations
    
    activations = ModernActivations()
    
    # Test gradient flow through multiple activations
    activation_functions = {
        'ReLU': activation_function_relu,
        'Swish': activations.swish,
        'GELU': activations.gelu,
        'Mish': activations.mish,
        'ELU': activations.elu,
        'Leaky ReLU': activations.leaky_relu
    }
    
    gradient_preservation = {}
    
    for name, activation_fn in activation_functions.items():
        print(f"Analyzing {name}...")
        
        all_simulations = []
        
        for sim in range(num_simulations):
            # Simulate gradient flowing through layers
            current_gradient = initial_gradient
            gradient_history = [current_gradient]
            
            for layer in range(num_layers):
                # Simulate activation input (varies during training)
                # Use different distributions to test robustness
                if layer < 5:
                    activation_input = np.random.normal(0, 1)  # Early layers
                elif layer < 10:
                    activation_input = np.random.normal(0, 0.5)  # Middle layers
                else:
                    activation_input = np.random.normal(0, 2)   # Later layers
                
                # Compute approximate derivative effect
                h = 1e-6
                if name == 'ReLU':
                    # ReLU derivative: 1 if x>0, 0 if x<=0
                    derivative_effect = 1.0 if activation_input > 0 else 0.0
                else:
                    # For smooth activations, compute numerical derivative
                    try:
                        derivative_effect = (activation_fn(activation_input + h) - 
                                           activation_fn(activation_input - h)) / (2 * h)
                        derivative_effect = float(derivative_effect)
                        # Clip extreme values
                        derivative_effect = np.clip(derivative_effect, 0.0, 2.0)
                    except:
                        derivative_effect = 0.5  # Fallback
                
                # Update gradient (simplified backprop)
                # Include weight effect (random weights between layers)
                weight_effect = np.random.normal(1.0, 0.1)  # Weights around 1
                current_gradient *= derivative_effect * abs(weight_effect)
                
                # Add some noise to simulate real training
                current_gradient *= (1.0 + 0.05 * np.random.randn())
                
                gradient_history.append(current_gradient)
            
            all_simulations.append({
                'final_gradient': current_gradient,
                'history': gradient_history
            })
        
        # Compute statistics across simulations
        final_gradients = [sim['final_gradient'] for sim in all_simulations]
        mean_final = np.mean(final_gradients)
        std_final = np.std(final_gradients)
        
        # Compute mean history
        all_histories = np.array([sim['history'] for sim in all_simulations])
        mean_history = np.mean(all_histories, axis=0)
        std_history = np.std(all_histories, axis=0)
        
        gradient_preservation[name] = {
            'final_gradient_mean': mean_final,
            'final_gradient_std': std_final,
            'mean_history': mean_history,
            'std_history': std_history,
            'all_finals': final_gradients
        }
    
    # Visualize gradient preservation
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Mean gradient flow with error bars
    plt.subplot(2, 3, 1)
    layers = range(num_layers + 1)
    for name, data in gradient_preservation.items():
        mean_hist = data['mean_history']
        std_hist = data['std_history']
        plt.plot(layers, mean_hist, label=name, linewidth=2)
        plt.fill_between(layers, mean_hist - std_hist, mean_hist + std_hist, alpha=0.2)
    
    plt.title('Gradient Flow Through Deep Network\n(Mean ± Std over 100 simulations)')
    plt.xlabel('Layer')
    plt.ylabel('Gradient Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Final gradient preservation
    plt.subplot(2, 3, 2)
    names = list(gradient_preservation.keys())
    means = [data['final_gradient_mean'] for data in gradient_preservation.values()]
    stds = [data['final_gradient_std'] for data in gradient_preservation.values()]
    
    bars = plt.bar(names, means, alpha=0.7, yerr=stds, capsize=5)
    plt.title('Final Gradient Preservation\n(Higher = Better)')
    plt.xlabel('Activation Function')
    plt.ylabel('Final Gradient (Mean ± Std)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Color bars based on performance
    max_mean = max(means)
    for bar, mean_val in zip(bars, means):
        if mean_val > 0.8 * max_mean:
            bar.set_color('green')
        elif mean_val > 0.5 * max_mean:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Plot 3: Gradient stability (coefficient of variation)
    plt.subplot(2, 3, 3)
    stability = []
    for name, data in gradient_preservation.items():
        cv = data['final_gradient_std'] / (data['final_gradient_mean'] + 1e-8)
        stability.append(cv)
    
    bars = plt.bar(names, stability, alpha=0.7)
    plt.title('Gradient Stability\n(Lower = More Stable)')
    plt.xlabel('Activation Function')
    plt.ylabel('Coefficient of Variation')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Color bars (lower is better for stability)
    min_cv = min(stability)
    max_cv = max(stability)
    for bar, cv in zip(bars, stability):
        if cv < min_cv + 0.3 * (max_cv - min_cv):
            bar.set_color('green')
        elif cv < min_cv + 0.7 * (max_cv - min_cv):
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Plot 4: Gradient preservation comparison (more meaningful than distribution)
    plt.subplot(2, 3, 4)
    names = list(gradient_preservation.keys())
    means = [data['final_gradient_mean'] for data in gradient_preservation.values()]

    # Show relative improvement over ReLU
    relu_baseline = means[0] if means[0] > 0 else 1e-10
    relative_improvements = [mean / relu_baseline for mean in means]

    bars = plt.bar(names, relative_improvements, alpha=0.7)
    plt.title('Gradient Preservation Relative to ReLU\n(Higher = Better)')
    plt.xlabel('Activation Function')
    plt.ylabel('Improvement Factor')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Color based on improvement
    for bar, improvement in zip(bars, relative_improvements):
        if improvement > 5:
            bar.set_color('green')
        elif improvement > 2:
            bar.set_color('orange')
        else:
            bar.set_color('red')
            
    # Plot 5: Vanishing gradient analysis
    plt.subplot(2, 3, 5)
    vanishing_rates = []
    for name, data in gradient_preservation.items():
        vanishing_count = sum(1 for grad in data['all_finals'] if grad < 0.01)
        vanishing_rate = vanishing_count / len(data['all_finals']) * 100
        vanishing_rates.append(vanishing_rate)
    
    bars = plt.bar(names, vanishing_rates, alpha=0.7)
    plt.title('Vanishing Gradient Rate\n(% of simulations with final grad < 0.01)')
    plt.xlabel('Activation Function')
    plt.ylabel('Vanishing Rate (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Color bars (lower is better)
    max_vanishing = max(vanishing_rates)
    for bar, rate in zip(bars, vanishing_rates):
        if rate < 0.2 * max_vanishing:
            bar.set_color('green')
        elif rate < 0.5 * max_vanishing:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Plot 6: Exploding gradient analysis
    plt.subplot(2, 3, 6)
    exploding_rates = []
    for name, data in gradient_preservation.items():
        exploding_count = sum(1 for grad in data['all_finals'] if grad > 10)
        exploding_rate = exploding_count / len(data['all_finals']) * 100
        exploding_rates.append(exploding_rate)
    
    bars = plt.bar(names, exploding_rates, alpha=0.7)
    plt.title('Exploding Gradient Rate\n(% of simulations with final grad > 10)')
    plt.xlabel('Activation Function')
    plt.ylabel('Exploding Rate (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Color bars (lower is better)
    max_exploding = max(exploding_rates) if max(exploding_rates) > 0 else 1
    for bar, rate in zip(bars, exploding_rates):
        if rate < 0.2 * max_exploding:
            bar.set_color('green')
        elif rate < 0.5 * max_exploding:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.show()
    
    # Report detailed results
    print("\nGradient Preservation Results:")
    print("=" * 70)
    print(f"{'Activation':<12} {'Mean Final':<12} {'Stability':<12} {'Vanishing%':<12} {'Exploding%'}")
    print("=" * 70)
    
    for i, (name, data) in enumerate(gradient_preservation.items()):
        mean_final = data['final_gradient_mean']
        cv = stability[i]
        vanishing_pct = vanishing_rates[i]
        exploding_pct = exploding_rates[i]
        
        print(f"{name:<12} {mean_final:<12.4f} {cv:<12.4f} {vanishing_pct:<12.1f} {exploding_pct:<12.1f}")
    
    print("\nInterpretation:")
    print("- Mean Final: Higher values indicate better gradient preservation")
    print("- Stability: Lower coefficient of variation indicates more consistent training")
    print("- Vanishing%: Lower percentage indicates fewer vanishing gradient problems")
    print("- Exploding%: Lower percentage indicates fewer exploding gradient problems")

def activation_recommendations_for_research():
    """
    Provide recommendations for activation choice in galaxy analysis.
    """
    print("=== Activation Function Recommendations ===")
    
    print(" For Galaxy Shear Measurement (ShearNet-style tasks):")
    print("1. GELU or Mish - Smooth gradients help precision tasks")
    print("   → Better convergence for sub-pixel accuracy requirements")
    print("2. Swish - Self-gating property good for subtle feature detection")
    print("   → Can adaptively emphasize important features")
    print("3. Avoid standard ReLU - Dying ReLU hurts subtle feature detection")
    print("   → Loss of gradient information is critical for precision tasks")
    
    print("\n For Galaxy Classification (Morphology, redshift estimation):")
    print("1. Swish or GELU - Good balance of performance and stability")
    print("2. Mish - Often gives best empirical results")
    print("3. Leaky ReLU - If computational budget is very tight")
    print("   → Minimal overhead compared to ReLU")
    
    print("\n For Very Deep Networks (>50 layers, ResNet-style):")
    print("1. GELU or Mish - Best gradient flow properties")
    print("2. Combine with:")
    print("   - Residual connections (essential)")
    print("   - Batch normalization")
    print("   - Proper weight initialization")
    print("3. Avoid ELU in very deep networks - can cause instability")
    
    print("\n Computational Considerations:")
    computation_times = {
        'ReLU': '1.0x (baseline)',
        'Leaky ReLU': '1.1x',
        'Swish': '2.3x',
        'GELU': '2.1x',
        'Mish': '3.2x',
        'ELU': '2.8x'
    }
    
    for activation, time in computation_times.items():
        print(f"  {activation:<12}: {time}")
    
    print("\n Specific Recommendations by Task:")
    
    recommendations = {
        "Weak Lensing Shear": ["GELU", "Mish", "Why: Smooth gradients crucial for precision"],
        "Galaxy-Galaxy Lensing": ["Swish", "GELU", "Why: Good feature extraction, stable training"],
        "Redshift Estimation": ["Mish", "Swish", "Why: Complex non-linear relationships"],
        "Star-Galaxy Separation": ["Leaky ReLU", "Swish", "Why: Fast inference needed, clear boundaries"],
        "Supernova Classification": ["GELU", "Mish", "Why: Temporal patterns, need smooth optimization"],
        "Galaxy Simulation": ["ELU", "Swish", "Why: Preserve negative information, physical constraints"]
    }
    
    for task, (primary, secondary, reason) in recommendations.items():
        print(f"  {task:<25}: {primary} > {secondary}")
        print(f"  {'':<25}  {reason}")
        print()
    
    print("Hyperparameter Tuning Tips:")
    print("- Swish: Try β ∈ [0.5, 1.5] for the scaling parameter")
    print("- ELU: α ∈ [0.5, 2.0] depending on desired negative saturation")
    print("- Leaky ReLU: α ∈ [0.01, 0.1] for negative slope")
    print("- Always validate on held-out galaxy data with realistic noise")
    
    print("\n Common Pitfalls:")
    print("- Don't change activation mid-training - causes instability")
    print("- Test with realistic galaxy noise levels - some activations more robust")
    print("- Consider activation choice early - affects all other hyperparameters")
    print("- Monitor gradient norms during training - early indicator of problems")

if __name__ == "__main__":
    print("Day 6: Modern Activation Functions!")
    print("Exploring beyond ReLU for better galaxy analysis")
    
    # Compare mathematical properties
    compare_activation_properties()
    
    # Test on galaxy data
    test_activations_on_galaxy_features()
    
    # Analyze gradient flow
    analyze_gradient_flow()
    
    # Research recommendations
    activation_recommendations_for_research()