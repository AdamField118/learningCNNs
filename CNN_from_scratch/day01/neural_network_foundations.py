"""
Mathematical foundations of neural networks with type hints and output logging.

Building understanding from first principles of how neural networks
perform computations, with focus on the mathematical operations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Tuple, Dict, Any, Optional, Union, List

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import our enhanced utilities
from utils.output_system import log_print, log_experiment_start, log_experiment_end, log_array_stats
from utils.visualization import plot_activation_comparison

def linear_transformation(
    input_vector: np.ndarray, 
    weight_matrix: np.ndarray, 
    bias_vector: np.ndarray
) -> np.ndarray:
    """
    The fundamental operation of neural networks: linear transformation.
    
    This is what happens at every layer: y = Wx + b
    
    Args:
        input_vector: numpy array, shape (n_features,)
        weight_matrix: numpy array, shape (n_outputs, n_features)  
        bias_vector: numpy array, shape (n_outputs,)
        
    Returns:
        output_vector: numpy array, shape (n_outputs,)
        
    Mathematical insight: This is just matrix multiplication + addition.
    Every complex neural network is built from these simple operations.
    """
    bias_vector = np.asarray(bias_vector)
    y = np.matmul(weight_matrix, input_vector) + bias_vector
    return y

def activation_function_sigmoid(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Sigmoid activation: squashes any real number to (0,1)
    
    Mathematical form: σ(x) = 1 / (1 + e^(-x))
    
    Why this matters: Without activation functions, neural networks 
    would just be linear algebra - they couldn't learn complex patterns.
    
    Args:
        x: Input value(s) to apply sigmoid to
        
    Returns:
        Sigmoid-activated values in range (0, 1)
    """
    # Handle numerical stability for very negative x values
    # When x is very negative, e^(-x) becomes very large, causing overflow
    # We use a numerically stable implementation
    
    # Convert to numpy array to handle both scalars and arrays
    x = np.asarray(x)
    
    # For positive x: use standard formula
    # For negative x: use equivalent formula that avoids overflow
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)),           # Standard formula for x >= 0
                    np.exp(x) / (1 + np.exp(x)))    # Stable formula for x < 0

def activation_function_relu(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    ReLU (Rectified Linear Unit): f(x) = max(0, x)
    
    Mathematical form: 
        f(x) = x if x > 0
        f(x) = 0 if x ≤ 0
    
    Why ReLU is important:
    1. Computationally simple (just thresholding)
    2. Helps with vanishing gradient problem
    3. Sparse activation (many neurons output 0)
    4. Most common activation in modern deep learning
    
    For CNNs: ReLU helps detect features by keeping positive responses
    and eliminating negative ones, which is perfect for edge detection.
    
    Args:
        x: Input value(s) to apply ReLU to
        
    Returns:
        ReLU-activated values (negative values become 0)
    """
    return np.maximum(0, x)

def compare_activation_functions() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Visualize different activation functions to understand their behavior.
    
    This helps build intuition about how different activations affect
    the network's ability to learn different types of patterns.
    
    Returns:
        Tuple of (x_values, sigmoid_values, relu_values)
    """
    log_print("=== Comparing Activation Functions ===", level="SUBHEADER")
    
    # Create range of input values
    x_values = np.linspace(-5, 5, 100)
    
    # Calculate activation functions
    sigmoid_values = activation_function_sigmoid(x_values)
    relu_values = activation_function_relu(x_values)
    
    # Log statistics
    log_array_stats("Sigmoid output", sigmoid_values)
    log_array_stats("ReLU output", relu_values)
    
    # Plot comparison
    plot_activation_comparison(
        x_values, 
        {'Sigmoid': sigmoid_values, 'ReLU': relu_values},
        title="Activation Function Analysis",
        filename_prefix="day01_activation_comparison"
    )

    return x_values, sigmoid_values, relu_values

def test_basic_operations() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Test our fundamental operations with concrete examples.
    
    Returns:
        Tuple of (linear_output, sigmoid_output, relu_output)
    """
    log_print("=== Testing Basic Operations ===", level="SUBHEADER")

    # Simple example: 2 inputs -> 1 output
    inputs = np.array([0.5, 0.3])  # Two input features
    weights = np.array([[0.4, 0.7]])  # One output neuron, two weights
    bias = np.array([0.1])
    
    # Test linear transformation
    linear_output = linear_transformation(inputs, weights, bias)
    log_print(f"Inputs: {inputs}")
    log_print(f"Weights: {weights}")
    log_print(f"Bias: {bias}")
    log_print(f"Linear output (Wx + b): {linear_output}")
    
    # Test both activation functions
    sigmoid_output = activation_function_sigmoid(linear_output)
    relu_output = activation_function_relu(linear_output)
    
    log_print(f"After sigmoid: {sigmoid_output}")
    log_print(f"After ReLU: {relu_output}")
    
    # Test with negative input to see ReLU behavior
    negative_input = np.array([-0.5])
    log_print(f"\nTesting with negative input: {negative_input}")
    log_print(f"Sigmoid(-0.5): {activation_function_sigmoid(negative_input)}")
    log_print(f"ReLU(-0.5): {activation_function_relu(negative_input)}")
    
    return linear_output, sigmoid_output, relu_output

def simple_perceptron(
    inputs: np.ndarray, 
    weights: np.ndarray, 
    bias: np.ndarray, 
    activation: str = 'sigmoid'
) -> np.ndarray:
    """
    A complete perceptron: linear transformation + activation.
    
    This combines the building blocks into a complete "neuron".
    
    Args:
        inputs: Input feature vector
        weights: Weight matrix
        bias: Bias vector
        activation: Activation function to use ('sigmoid' or 'relu')
        
    Returns:
        Activated output from perceptron
    """
    linear_out = linear_transformation(inputs, weights, bias)
    
    if activation == 'sigmoid':
        return activation_function_sigmoid(linear_out)
    elif activation == 'relu':
        return activation_function_relu(linear_out)
    else:
        raise ValueError(f"Unknown activation: {activation}")

def test_perceptron_decision_boundary() -> None:
    """
    Test the perceptron on a simple classification problem.
    This will show how neural networks make decisions!
    """
    log_print("=== Testing Perceptron Decision Making ===", level="SUBHEADER")
    
    # Simple 2D classification problem
    # Try to separate points into two classes
    
    # Test points in 2D space
    test_points = np.array([
        [0, 0],    # Should be class 0
        [0, 1],    # Should be class 1  
        [1, 0],    # Should be class 1
        [1, 1]     # Should be class 1
    ])
    
    # Weights for a simple decision boundary
    weights = np.array([[1, 1]])  # Sum the inputs
    bias = np.array([-0.5])       # Threshold at 0.5

    log_print("Testing Perceptron Decision Making:")
    log_print(f"Weights: {weights}")
    log_print(f"Bias: {bias}")
    log_print("\nPoint\t\tSigmoid\t\tReLU")
    log_print("-" * 40)
    
    for point in test_points:
        sig_out = simple_perceptron(point, weights, bias, 'sigmoid')
        relu_out = simple_perceptron(point, weights, bias, 'relu')
        log_print(f"{point}\t\t{sig_out[0]:.3f}\t\t{relu_out[0]:.3f}")

def demonstrate_parameter_explosion() -> None:
    """
    Show why fully connected layers don't scale to images.
    This motivates the need for convolution!
    """
    log_print("=== Parameter Explosion Demonstration ===", level="SUBHEADER")
    log_print("Parameter count for different approaches:")
    
    # Small 28x28 image (like MNIST digits)
    image_pixels = 28 * 28
    hidden_layer_size = 100
    
    fully_connected_params = image_pixels * hidden_layer_size
    log_print(f"28x28 image -> 100 hidden neurons:")
    log_print(f"Parameters needed: {fully_connected_params:,}")
    
    # Larger image like galaxy images might be
    galaxy_image_pixels = 200 * 200
    fully_connected_galaxy = galaxy_image_pixels * hidden_layer_size
    log_print(f"\n200x200 galaxy image -> 100 hidden neurons:")
    log_print(f"Parameters needed: {fully_connected_galaxy:,}")
    
    log_print(f"\nWhy this is a problem:")
    log_print(f"- Each neuron needs to learn about EVERY pixel")
    log_print(f"- No sharing of knowledge between similar patterns")
    log_print(f"- Massive memory requirements")
    log_print(f"- Easy to overfit")

def sigmoid_derivative(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Derivative of sigmoid function.
    
    Mathematical insight: σ'(x) = σ(x) * (1 - σ(x))
    
    This is why sigmoid can cause vanishing gradients - when σ(x) is near 0 or 1,
    the derivative becomes very small, making learning slow.
    
    Args:
        x: Input values
        
    Returns:
        Derivative of sigmoid at x
    """
    sigmoid_x = activation_function_sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

def relu_derivative(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Derivative of ReLU function.
    
    Mathematical insight: 
        f'(x) = 1 if x > 0
        f'(x) = 0 if x ≤ 0
    
    This is why ReLU helps with vanishing gradients - the derivative is either 0 or 1,
    never getting smaller than 1 for positive inputs.
    
    Args:
        x: Input values
        
    Returns:
        Derivative of ReLU at x
    """
    return np.where(x > 0, 1.0, 0.0)

def forward_pass_single_neuron(
    inputs: np.ndarray, 
    weights: np.ndarray, 
    bias: np.ndarray, 
    activation: str = 'sigmoid'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Forward pass with detailed intermediate values saved for backprop.
    
    Returns both the output AND the intermediate values needed for backward pass.
    This is how real neural network frameworks work internally.
    
    Args:
        inputs: Input feature vector
        weights: Weight matrix
        bias: Bias vector
        activation: Activation function to use
        
    Returns:
        Tuple of (activated_output, cache_dict)
    """
    # Linear transformation
    linear_output = linear_transformation(inputs, weights, bias)
    
    # Activation
    if activation == 'sigmoid':
        activated_output = activation_function_sigmoid(linear_output)
    elif activation == 'relu':
        activated_output = activation_function_relu(linear_output)
    else:
        raise ValueError(f"Unknown activation: {activation}")
    
    # Save intermediate values for backprop
    cache = {
        'inputs': inputs,
        'weights': weights,
        'bias': bias,
        'linear_output': linear_output,
        'activated_output': activated_output,
        'activation': activation
    }
    
    return activated_output, cache

def backward_pass_single_neuron(
    output_gradient: np.ndarray, 
    cache: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Backward pass: compute gradients with respect to inputs, weights, and bias.
    
    This implements the chain rule of calculus to figure out how much each
    parameter contributed to the final error.
    
    Args:
        output_gradient: How much the output should change (∂Loss/∂output)
        cache: Intermediate values from forward pass
        
    Returns:
        Dictionary with gradients for weights, bias, and inputs
    """
    # Extract cached values
    inputs = cache['inputs']
    weights = cache['weights']
    linear_output = cache['linear_output']
    activation = cache['activation']
    
    # Step 1: Gradient through activation function
    if activation == 'sigmoid':
        activation_grad = sigmoid_derivative(linear_output)
    elif activation == 'relu':
        activation_grad = relu_derivative(linear_output)
    
    # Chain rule: ∂Loss/∂linear = ∂Loss/∂output * ∂output/∂linear
    linear_gradient = output_gradient * activation_grad
    
    # Step 2: Gradients with respect to weights, bias, inputs
    # Remember: linear_output = weights @ inputs + bias
    
    # ∂Loss/∂weights = ∂Loss/∂linear * ∂linear/∂weights = linear_gradient * inputs
    weights_gradient = np.outer(linear_gradient, inputs)
    
    # ∂Loss/∂bias = ∂Loss/∂linear * ∂linear/∂bias = linear_gradient * 1
    bias_gradient = linear_gradient
    
    # ∂Loss/∂inputs = ∂Loss/∂linear * ∂linear/∂inputs = linear_gradient * weights
    inputs_gradient = weights.T @ linear_gradient
    
    gradients = {
        'weights': weights_gradient,
        'bias': bias_gradient,
        'inputs': inputs_gradient
    }
    
    return gradients

def demonstrate_learning(activation: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Show how backpropagation enables learning by updating weights.
    
    This demonstrates the complete learning cycle:
    1. Forward pass (make prediction)
    2. Calculate error
    3. Backward pass (compute gradients)
    4. Update parameters
    5. Repeat
    
    Args:
        activation: Activation function to use for demonstration
        
    Returns:
        Tuple of (final_weights, final_bias)
    """
    log_print("=== Learning Through Backpropagation ===", level="SUBHEADER")

    # Simple learning task: learn to output 1.0 when input is [1, 1]
    target_output = 1.0
    inputs = np.array([1.0, 1.0])
    
    # Initialize random weights and bias
    np.random.seed(42)  # For reproducible results
    weights = np.random.randn(1, 2) * 0.5  # Small random weights
    bias = np.random.randn(1) * 0.5
    
    learning_rate = 0.1
    log_print(f"Target output: {target_output}")
    log_print(f"Input: {inputs}")
    log_print(f"Initial weights: {weights}")
    log_print(f"Initial bias: {bias}")
    log_print("")
    
    # Training loop
    for epoch in range(10):
        # Forward pass
        output, cache = forward_pass_single_neuron(inputs, weights, bias, activation)
        
        # Calculate error (simple squared error)
        error = target_output - output[0]
        loss = 0.5 * error**2
        
        # Calculate output gradient (∂Loss/∂output)
        output_gradient = np.array([-error])  # Derivative of 0.5*(target-output)^2
        
        # Backward pass
        gradients = backward_pass_single_neuron(output_gradient, cache)
        
        # Update parameters
        weights = weights - learning_rate * gradients['weights']
        bias = bias - learning_rate * gradients['bias']
        
        if epoch % 2 == 0:  # Print every 2 epochs
            log_print(f"Epoch {epoch:2d}: Output={output[0]:.4f}, Error={error:.4f}, Loss={loss:.4f}")
    
    log_print(f"\nFinal weights: {weights}")
    log_print(f"Final bias: {bias}")
    log_print(f"Final output: {output[0]:.4f} (target: {target_output})")
    
    return weights, bias

def main() -> None:
    """Main execution function for Day 1 experiments."""
    log_experiment_start(1, "Neural Network Foundations")
    
    # Run all experiments
    test_basic_operations()
    compare_activation_functions()
    test_perceptron_decision_boundary()
    demonstrate_parameter_explosion()
    demonstrate_learning('sigmoid')
    
    log_experiment_end(1)

if __name__ == "__main__":
    main()