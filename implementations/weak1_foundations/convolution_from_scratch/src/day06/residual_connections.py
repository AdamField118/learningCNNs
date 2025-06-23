"""
Residual connections and solving the vanishing gradient problem.

This module explores how residual connections enable training of very deep networks
and why they're crucial for complex tasks like galaxy shape measurement.
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
from utils.visualization import plot_feature_responses
from day02.edge_detection_basics import manual_convolution_2d
from day01.neural_network_foundations import activation_function_relu

class ResidualBlock:
    """
    A residual block implementation to understand skip connections.

    The key insight: instead of learning H(x), learn F(x) = H(x) - x
    Then H(x) = F(x) + x (the residual connection)
    """

    def __init__(self, input_channels, output_channels):
        """
        Initialize a residual block.

        Args:
            input_channels: Number of input feature channels
            output_channels: Number of output feature channels
        """
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Initialize weights for two convolution layers
        self.conv1_weights = None  
        self.conv2_weights = None  

        # Handle channel mismatch with a 1x1 convolution
        self.skip_weights = None   

        # Initialize all weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with proper scaling for gradient flow."""
        # Xavier/He initialization for better gradient flow
        # For ReLU networks, He initialization works better: std = sqrt(2/fan_in)

        # Initialize conv1_weights as 3x3 filter
        fan_in_1 = 3 * 3 * self.input_channels
        self.conv1_weights = np.random.randn(3, 3) * np.sqrt(2.0 / fan_in_1)

        # Initialize conv2_weights as 3x3 filter
        fan_in_2 = 3 * 3 * self.output_channels
        self.conv2_weights = np.random.randn(3, 3) * np.sqrt(2.0 / fan_in_2)

        # Initialize skip_weights if input/output channels differ
        # This should be a 1x1 convolution to match dimensions
        if self.input_channels != self.output_channels:
            # For simplicity in this educational example, we'll use a simple scaling
            # In real implementations, this would be a 1x1 conv
            self.skip_weights = np.random.randn(1, 1) * np.sqrt(2.0 / self.input_channels)

    def forward(self, x):
        """
        Forward pass through residual block.

        Args:
            x: Input feature map

        Returns:
            Output with residual connection: F(x) + x
        """
        # Store the input for the skip connection
        identity = x.copy()

        # Apply first convolution + ReLU
        conv1_output = manual_convolution_2d(x, self.conv1_weights)
        conv1_activated = activation_function_relu(conv1_output)

        # Apply second convolution (NO activation yet!)
        conv2_output = manual_convolution_2d(conv1_activated, self.conv2_weights)

        # Handle dimension mismatch for skip connection
        # If channels don't match, apply 1x1 conv to identity
        if self.input_channels != self.output_channels and self.skip_weights is not None:
            identity = manual_convolution_2d(identity, self.skip_weights)

        # Ensure spatial dimensions match after convolutions
        # Crop identity to match conv2_output size if needed
        if identity.shape != conv2_output.shape:
            # Calculate cropping needed
            h_diff = identity.shape[0] - conv2_output.shape[0]
            w_diff = identity.shape[1] - conv2_output.shape[1]

            if h_diff > 0 or w_diff > 0:
                h_start = h_diff // 2
                w_start = w_diff // 2
                h_end = h_start + conv2_output.shape[0]
                w_end = w_start + conv2_output.shape[1]
                identity = identity[h_start:h_end, w_start:w_end]

        # Add skip connection (THE KEY INSIGHT!)
        output = conv2_output + identity

        # Apply activation AFTER adding skip connection
        final_output = activation_function_relu(output)

        return final_output

def demonstrate_vanishing_gradients():
    """
    Demonstrate the vanishing gradient problem with deep networks.

    This shows why we need residual connections for deep networks.
    """
    print("=== Demonstrating Vanishing Gradient Problem ===")

    # Create a simple deep network simulation
    # Simulate gradients flowing backward through many layers

    initial_gradient = 1.0
    num_layers = 20

    # Simulate gradient flow through layers WITHOUT residual connections
    gradients_without_residual = []
    current_gradient = initial_gradient

    for layer in range(num_layers):
        # Typical gradient in deep networks gets multiplied by small weights
        # Use a multiplication factor like 0.8 to simulate this
        current_gradient = current_gradient * 0.8  # Gradients get smaller each layer
        gradients_without_residual.append(current_gradient)

    # Simulate gradient flow WITH residual connections
    gradients_with_residual = []
    current_gradient = initial_gradient

    for layer in range(num_layers):
        # With residual connections, gradients flow more directly
        # The gradient gets added (skip connection) rather than just multiplied
        # Use a factor like 0.8 * current + 0.2 * initial to simulate this
        residual_contribution = 0.2 * initial_gradient  # Direct path from skip connection
        main_path = 0.8 * current_gradient  # Diminished main path
        current_gradient = main_path + residual_contribution
        gradients_with_residual.append(current_gradient)

    # Plot the comparison
    # Show how gradients vanish without residual connections
    # but remain strong with residual connections

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_layers + 1), gradients_without_residual, 'r-', linewidth=2, label='Without Residual')
    plt.plot(range(1, num_layers + 1), gradients_with_residual, 'b-', linewidth=2, label='With Residual')
    plt.xlabel('Layer Depth')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Flow Through Deep Network')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to see the dramatic difference

    plt.subplot(1, 2, 2)
    plt.bar(['Without Residual', 'With Residual'], 
            [gradients_without_residual[-1], gradients_with_residual[-1]],
            color=['red', 'blue'], alpha=0.7)
    plt.ylabel('Final Gradient Magnitude')
    plt.title('Gradient at Layer 20')
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

    print("Without residual connections:")
    print(f"Initial gradient: {initial_gradient}")
    print(f"Final gradient: {gradients_without_residual[-1]:.6f}")

    print("\nWith residual connections:")
    print(f"Final gradient: {gradients_with_residual[-1]:.6f}")

    improvement_factor = gradients_with_residual[-1] / gradients_without_residual[-1]
    print(f"Improvement factor: {improvement_factor:.2f}x")

def test_residual_vs_plain_network():
    """
    Compare a plain deep network vs one with residual connections.

    This demonstrates why ResNet was revolutionary.
    """
    print("=== Residual vs Plain Network Comparison ===")

    galaxy = create_synthetic_galaxy(size=50, spiral_arms=True)

    # Create a plain deep network (multiple conv layers)
    def plain_deep_network(image):
        """Apply multiple convolutions without skip connections."""
        # Apply 5 convolution layers in sequence
        # Each layer should make the representation more abstract
        # But gradients will have trouble flowing back through all layers

        # Simple edge detection kernels of decreasing strength
        kernels = [
            np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),  # Strong Sobel
            np.array([[-0.5, 0, 0.5], [-1, 0, 1], [-0.5, 0, 0.5]], dtype=np.float32),  # Weaker
            np.array([[-0.25, 0, 0.25], [-0.5, 0, 0.5], [-0.25, 0, 0.25]], dtype=np.float32),  # Even weaker
            np.array([[-0.1, 0, 0.1], [-0.2, 0, 0.2], [-0.1, 0, 0.1]], dtype=np.float32),  # Very weak
            np.array([[-0.05, 0, 0.05], [-0.1, 0, 0.1], [-0.05, 0, 0.05]], dtype=np.float32)  # Barely there
        ]

        current = image
        for i, kernel in enumerate(kernels):
            try:
                current = manual_convolution_2d(current, kernel)
                current = activation_function_relu(current)
                print(f"  Plain Layer {i+1}: {current.shape}, max={np.max(current):.3f}")
            except:
                print(f"  Plain Layer {i+1}: Failed (too small)")
                break

        return current

    # Create a residual network
    def residual_network(image):
        """Apply multiple convolutions WITH skip connections."""
        # Apply the same number of layers but with residual blocks
        # Each block should maintain the skip connection

        print("  Residual Network:")
        residual_block = ResidualBlock(input_channels=1, output_channels=1)

        current = image
        for i in range(3):  # Fewer layers due to size constraints
            try:
                current = residual_block.forward(current)
                print(f"  Residual Block {i+1}: {current.shape}, max={np.max(current):.3f}")
            except Exception as e:
                print(f"  Residual Block {i+1}: Failed - {str(e)}")
                break

        return current

    # Compare the outputs
    print("Plain Network:")
    plain_result = plain_deep_network(galaxy)

    print("\nResidual Network:")
    residual_result = residual_network(galaxy)

    # Visualize the difference
    plot_feature_responses(galaxy, {
        'Plain Deep Network': plain_result,
        'Residual Network': residual_result
    }, title="Plain vs Residual Network Comparison")

    print(f"\nPlain network final response: {np.max(np.abs(plain_result)):.3f}")
    print(f"Residual network final response: {np.max(np.abs(residual_result)):.3f}")
    print("Residual network maintains stronger signal through deeper layers!")

def residual_connections_for_galaxies():
    """
    Explore how residual connections help with galaxy feature detection.

    This connects residual connections to your astronomical applications.
    """
    print("=== Residual Connections for Galaxy Analysis ===")

    galaxy = create_synthetic_galaxy(size=48, spiral_arms=True)

    # Show how deep networks can extract complex galaxy features
    # Layer 1: Basic edges (your Day 2 work)
    edge_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    layer1_features = manual_convolution_2d(galaxy, edge_kernel)
    layer1_features = activation_function_relu(layer1_features)

    # Layer 2: Texture patterns (your Day 3 work)  
    texture_kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)

    # Without residual connections:
    # - Deep networks lose low-level features (edges get lost)
    # - Can't combine multiple scales of information effectively
    layer2_without_residual = manual_convolution_2d(layer1_features, texture_kernel)
    layer2_without_residual = activation_function_relu(layer2_without_residual)

    # With residual connections:
    # - Preserve edge information through skip connections
    # - Combine low-level and high-level features
    # - Better galaxy shape measurement
    residual_block = ResidualBlock(input_channels=1, output_channels=1)
    layer2_with_residual = residual_block.forward(layer1_features)

    plot_feature_responses(galaxy, {
        'Layer 1: Edges': layer1_features,
        'Layer 2: Without Residual': layer2_without_residual,
        'Layer 2: With Residual': layer2_with_residual
    }, title="Residual Connections for Galaxy Feature Preservation")

    print("Key insight: Galaxy shear measurement needs BOTH:")
    print("1. Low-level edge information (precise shape)")
    print("2. High-level structural information (galaxy type)")
    print("Residual connections preserve both!")

    # Analyze signal preservation
    edge_signal = np.max(np.abs(layer1_features))
    without_residual_signal = np.max(np.abs(layer2_without_residual))
    with_residual_signal = np.max(np.abs(layer2_with_residual))

    print(f"\nSignal preservation analysis:")
    print(f"Layer 1 (edges): {edge_signal:.3f}")
    print(f"Layer 2 without residual: {without_residual_signal:.3f} ({without_residual_signal/edge_signal:.1%} preserved)")
    print(f"Layer 2 with residual: {with_residual_signal:.3f} ({with_residual_signal/edge_signal:.1%} preserved)")

def why_resnet_revolutionized_deep_learning():
    """
    Explain the historical impact and key insights of ResNet.
    """
    print("=== Why ResNet Changed Everything ===")

    print("Before ResNet (2015):")
    print("- Networks deeper than ~20 layers performed WORSE")
    print("- Not due to overfitting - even training error increased!")
    print("- Vanishing gradients made deep networks untrainable")

    print("\nResNet's insight:")
    print("- Let layers learn residual function F(x) = H(x) - x")
    print("- Identity mapping is easier to learn than complex mapping")
    print("- Skip connections provide gradient superhighway")

    print("\nImpact:")
    print("- Enabled 50, 101, even 1000+ layer networks")
    print("- Dramatic improvements in image recognition")
    print("- Foundation for modern architectures (Transformers use similar ideas)")

    # Show the mathematical insight
    print("\nMathematical insight:")
    print("Traditional layer: H(x) = F(weight*x + bias)")
    print("Residual layer: H(x) = F(weight*x + bias) + x")
    print("If optimal mapping is close to identity, F can learn small corrections")

def visualize_residual_flow():
    """
    Create a visualization showing how information flows
    through residual connections vs plain connections.
    """
    print("=== Visualizing Residual Information Flow ===")

    # Create a simple demonstration
    galaxy = create_synthetic_galaxy(size=40, spiral_arms=True)

    # Simulate information flow through layers
    layer_outputs = {'Input': galaxy}

    # Plain network - information degrades
    current = galaxy
    for i in range(4):
        # Apply a kernel that reduces signal strength
        weak_kernel = np.array([[0.1, 0.2, 0.1], [0.2, 0.2, 0.2], [0.1, 0.2, 0.1]], dtype=np.float32)
        current = manual_convolution_2d(current, weak_kernel)
        current = activation_function_relu(current)
        layer_outputs[f'Plain Layer {i+1}'] = current

    # Residual network - information preserved  
    residual_block = ResidualBlock(input_channels=1, output_channels=1)
    current = galaxy
    for i in range(2):  # Fewer layers due to size constraints
        current = residual_block.forward(current)
        layer_outputs[f'Residual Layer {i+1}'] = current

    plot_feature_responses(galaxy, layer_outputs, 
                         title="Information Flow: Plain vs Residual Networks")

    print("Notice how residual connections maintain signal strength!")

if __name__ == "__main__":
    # Start with the problem
    demonstrate_vanishing_gradients()

    # Show the solution
    test_residual_vs_plain_network()

    # Connect to your research
    residual_connections_for_galaxies()

    # Historical context
    why_resnet_revolutionized_deep_learning()

    # Visualization
    visualize_residual_flow()