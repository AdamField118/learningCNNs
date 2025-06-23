"""
Batch normalization and training stabilization.

This module explores how batch normalization stabilizes training and enables
faster convergence in deep networks, particularly important for galaxy CNNs.
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
from day01.neural_network_foundations import activation_function_relu

class BatchNormLayer:
    """
    Batch normalization layer implementation.
    
    Key insight: Normalize inputs to each layer to have mean=0, std=1
    Then learn optimal mean and std via learnable parameters.
    """
    
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        """
        Initialize batch normalization layer.
        
        Args:
            num_features: Number of features/channels to normalize
            epsilon: Small constant for numerical stability
            momentum: Momentum for running statistics update
        """
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Initialize learnable parameters
        self.gamma = np.ones(num_features)  # Scale parameter (learnable)
        self.beta = np.zeros(num_features)  # Shift parameter (learnable)
        
        # Initialize running statistics for inference
        self.running_mean = np.zeros(num_features)  # Running mean for test time
        self.running_var = np.ones(num_features)    # Running variance for test time
        
    def forward(self, x, training=True):
        """
        Forward pass of batch normalization.
        
        Args:
            x: Input features [batch_size, height, width, channels] or [batch_size, features]
            training: Whether in training mode
            
        Returns:
            Normalized and scaled features
        """
        # Handle both 2D and 4D inputs
        original_shape = x.shape
        if len(x.shape) == 4:
            # Reshape to [batch_size * height * width, channels]
            batch_size, height, width, channels = x.shape
            x = x.reshape(-1, channels)
        
        if training:
            # Compute batch statistics along batch dimension (axis=0)
            batch_mean = np.mean(x, axis=0)     # Mean across batch dimension
            batch_var = np.var(x, axis=0)       # Variance across batch dimension
            
            # Normalize using batch statistics
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
            # Update running statistics for inference using exponential moving average
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
        else:
            # Use running statistics for inference
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Apply learnable scale and shift
        output = self.gamma * x_normalized + self.beta
        
        # Reshape back to original shape if needed
        if len(original_shape) == 4:
            output = output.reshape(original_shape)
            
        return output

def demonstrate_internal_covariate_shift():
    """
    Show the problem batch normalization solves.
    """
    print("=== Internal Covariate Shift Problem ===")
    
    # Simulate how activations change during training
    # Create synthetic "before" and "after" activation distributions
    
    # Layer 1 activations at start of training
    layer1_early = np.random.normal(0, 1, 1000)
    
    # Layer 1 activations after some training (distribution has shifted)
    layer1_later = np.random.normal(2, 3, 1000)  # Mean and std changed
    
    # Show how this affects subsequent layers
    print(f"Early training - Layer 1: mean={layer1_early.mean():.2f}, std={layer1_later.std():.2f}")
    print(f"Later training - Layer 1: mean={layer1_later.mean():.2f}, std={layer1_later.std():.2f}")
    
    # Show impact on next layer (using ReLU activation)
    layer2_early = activation_function_relu(layer1_early)
    layer2_later = activation_function_relu(layer1_later)
    
    print(f"Early training - Layer 2 (post-ReLU): mean={layer2_early.mean():.2f}, std={layer2_early.std():.2f}")
    print(f"Later training - Layer 2 (post-ReLU): mean={layer2_later.mean():.2f}, std={layer2_later.std():.2f}")
    
    # Visualize the distribution shift
    plt.figure(figsize=(15, 8))
    
    # Original distributions
    plt.subplot(2, 3, 1)
    plt.hist(layer1_early, alpha=0.7, label='Early Training', bins=30, color='blue')
    plt.hist(layer1_later, alpha=0.7, label='Later Training', bins=30, color='red')
    plt.title('Layer 1: Internal Covariate Shift Problem')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # After ReLU
    plt.subplot(2, 3, 2)
    plt.hist(layer2_early, alpha=0.7, label='Early Training', bins=30, color='blue')
    plt.hist(layer2_later, alpha=0.7, label='Later Training', bins=30, color='red')
    plt.title('Layer 2: After ReLU Activation')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Apply batch normalization to fix this
    normalized_early = (layer1_early - layer1_early.mean()) / (layer1_early.std() + 1e-5)
    normalized_later = (layer1_later - layer1_later.mean()) / (layer1_later.std() + 1e-5)
    
    plt.subplot(2, 3, 3)
    plt.hist(normalized_early, alpha=0.7, label='Normalized Early', bins=30, color='green')
    plt.hist(normalized_later, alpha=0.7, label='Normalized Later', bins=30, color='orange')
    plt.title('After Batch Normalization')
    plt.xlabel('Normalized Activation Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Show the effect on gradients (simplified)
    plt.subplot(2, 3, 4)
    gradient_early = np.random.normal(0, 2, 1000)  # Unstable gradients
    gradient_later = np.random.normal(0, 0.1, 1000)  # Vanishing gradients
    plt.hist(gradient_early, alpha=0.7, label='Early (Unstable)', bins=30, color='blue')
    plt.hist(gradient_later, alpha=0.7, label='Later (Vanishing)', bins=30, color='red')
    plt.title('Gradient Problems Without BatchNorm')
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Stable gradients with batch norm
    plt.subplot(2, 3, 5)
    stable_gradients = np.random.normal(0, 1, 1000)
    plt.hist(stable_gradients, alpha=0.7, label='With BatchNorm', bins=30, color='green')
    plt.title('Stable Gradients With BatchNorm')
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nKey Problem: As training progresses, input distributions to each layer change,")
    print("making it harder for subsequent layers to learn effectively.")
    print("BatchNorm Solution: Keep input distributions normalized and stable.")

def compare_training_stability():
    """
    Compare training with and without batch normalization.
    """
    print("=== Training Stability Comparison ===")
    
    # Simulate training curves
    epochs = 50
    np.random.seed(42)  # For reproducible results
    
    # Without batch norm: unstable training
    learning_rate_no_bn = 0.01
    loss_no_bn = []
    current_loss = 1.0
    
    for epoch in range(epochs):
        # Simulate unstable training (high variance)
        # Without batch norm, training is sensitive to learning rate
        noise_factor = 0.3 * np.random.randn()  # High variance
        gradient_scale = np.random.uniform(0.5, 2.0)  # Inconsistent gradients
        
        # Simulate learning step with instability
        step = learning_rate_no_bn * gradient_scale * (0.02 + noise_factor)
        current_loss -= step
        
        # Add occasional gradient explosions
        if epoch > 10 and np.random.random() < 0.1:
            current_loss += 0.1 * np.random.randn()
            
        # Prevent negative loss
        current_loss = max(0.01, current_loss)
        loss_no_bn.append(current_loss)
    
    # With batch norm: stable training
    learning_rate_bn = 0.1  # Can use higher learning rate!
    loss_bn = []
    current_loss = 1.0
    
    for epoch in range(epochs):
        # Simulate stable training
        # Batch norm allows higher learning rates and more stable convergence
        noise_factor = 0.05 * np.random.randn()  # Low variance
        gradient_scale = 1.0  # Consistent gradients due to normalization
        
        # Stable learning step
        step = learning_rate_bn * gradient_scale * (0.02 + noise_factor)
        current_loss -= step
        
        # Prevent negative loss
        current_loss = max(0.01, current_loss)
        loss_bn.append(current_loss)
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_no_bn, 'r-', linewidth=2, label='Without BatchNorm')
    plt.plot(loss_bn, 'g-', linewidth=2, label='With BatchNorm')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show learning rate sensitivity
    plt.subplot(1, 2, 2)
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    final_losses_no_bn = [0.3, 0.15, 0.4, 0.8]  # Sensitive to LR
    final_losses_bn = [0.08, 0.05, 0.03, 0.04]   # Robust to LR
    
    x = np.arange(len(learning_rates))
    width = 0.35
    
    plt.bar(x - width/2, final_losses_no_bn, width, label='Without BatchNorm', color='red', alpha=0.7)
    plt.bar(x + width/2, final_losses_bn, width, label='With BatchNorm', color='green', alpha=0.7)
    
    plt.title('Learning Rate Sensitivity')
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Loss')
    plt.xticks(x, learning_rates)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Without Batch Norm:")
    print("  - Unstable training with high variance")
    print("  - Requires careful learning rate tuning")
    print("  - Prone to gradient explosion/vanishing")
    print("  - Final loss: {:.3f}".format(loss_no_bn[-1]))
    
    print("\nWith Batch Norm:")
    print("  - Stable, smooth convergence")
    print("  - Allows higher learning rates")
    print("  - Robust to hyperparameter choices")
    print("  - Final loss: {:.3f}".format(loss_bn[-1]))

def batch_norm_for_galaxy_networks():
    """
    Apply batch normalization concepts to galaxy analysis.
    """
    print("=== Batch Normalization for Galaxy Networks ===")
    
    # Create multiple galaxy images with different characteristics
    galaxies = []
    galaxy_types = []
    
    for i in range(8):
        # Create galaxies with varying properties
        if i < 3:
            # Bright spiral galaxies
            galaxy = create_synthetic_galaxy(size=32, spiral_arms=True, add_noise=False)
            galaxy = galaxy * (1.5 + 0.5 * np.random.randn())  # Vary brightness
            galaxy_types.append("Bright Spiral")
        elif i < 6:
            # Faint elliptical galaxies
            galaxy = create_synthetic_galaxy(size=32, spiral_arms=False, add_noise=True)
            galaxy = galaxy * (0.5 + 0.2 * np.random.randn())  # Fainter
            galaxy_types.append("Faint Elliptical")
        else:
            # Noisy distant galaxies
            galaxy = create_synthetic_galaxy(size=32, spiral_arms=True, add_noise=True)
            galaxy = galaxy * (0.3 + 0.1 * np.random.randn())  # Very faint
            galaxy_types.append("Distant Noisy")
            
        galaxies.append(galaxy)
    
    batch = np.stack(galaxies, axis=0)  # Create batch [8, 32, 32]
    
    print(f"Created batch of {len(galaxies)} galaxies")
    print(f"Galaxy brightness ranges:")
    for i, (galaxy, gtype) in enumerate(zip(galaxies, galaxy_types)):
        print(f"  {gtype}: min={galaxy.min():.3f}, max={galaxy.max():.3f}, mean={galaxy.mean():.3f}")
    
    # Show activation statistics without batch norm
    # Apply edge detection (convolution) to the batch
    from kernels.edge_detection_kernels import sobel_x_kernel
    kernel = sobel_x_kernel()
    
    activations = []
    for galaxy in galaxies:
        from day02.edge_detection_basics import manual_convolution_2d
        activation = manual_convolution_2d(galaxy, kernel)
        activations.append(activation)
    
    # Analyze activation statistics
    all_activations = np.concatenate([act.flatten() for act in activations])
    
    print(f"\nBefore Batch Norm (after edge detection):")
    print(f"  Mean: {all_activations.mean():.3f}")
    print(f"  Std: {all_activations.std():.3f}")
    print(f"  Min: {all_activations.min():.3f}")
    print(f"  Max: {all_activations.max():.3f}")
    print(f"  Range: {all_activations.max() - all_activations.min():.3f}")
    
    # Apply batch normalization
    # First, let's simulate this would be applied in a CNN layer
    batch_size, height, width = batch.shape
    
    # Reshape for batch norm (treat each pixel as a feature)
    batch_reshaped = batch.reshape(batch_size, -1)  # [8, 1024]
    
    # Initialize and apply batch normalization
    bn_layer = BatchNormLayer(num_features=batch_reshaped.shape[1])
    normalized_batch = bn_layer.forward(batch_reshaped, training=True)
    
    # Apply edge detection to normalized galaxies
    normalized_galaxies = normalized_batch.reshape(batch_size, height, width)
    normalized_activations = []
    
    for galaxy in normalized_galaxies:
        activation = manual_convolution_2d(galaxy, kernel)
        normalized_activations.append(activation)
    
    all_normalized_activations = np.concatenate([act.flatten() for act in normalized_activations])
    
    print(f"\nAfter Batch Norm (after edge detection):")
    print(f"  Mean: {all_normalized_activations.mean():.3f}")
    print(f"  Std: {all_normalized_activations.std():.3f}")
    print(f"  Min: {all_normalized_activations.min():.3f}")
    print(f"  Max: {all_normalized_activations.max():.3f}")
    print(f"  Range: {all_normalized_activations.max() - all_normalized_activations.min():.3f}")
    
    # Visualize the effect
    plt.figure(figsize=(15, 10))
    
    # Show sample galaxies
    for i in range(4):
        plt.subplot(3, 4, i+1)
        plt.imshow(galaxies[i], cmap='gray')
        plt.title(f'Original: {galaxy_types[i]}')
        plt.axis('off')
        
        plt.subplot(3, 4, i+5)
        plt.imshow(normalized_galaxies[i], cmap='gray')
        plt.title(f'Normalized: {galaxy_types[i]}')
        plt.axis('off')
    
    # Show activation histograms
    plt.subplot(3, 2, 5)
    plt.hist(all_activations, bins=50, alpha=0.7, color='red', label='Before BatchNorm')
    plt.title('Edge Detection Activations: Before BatchNorm')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(3, 2, 6)
    plt.hist(all_normalized_activations, bins=50, alpha=0.7, color='green', label='After BatchNorm')
    plt.title('Edge Detection Activations: After BatchNorm')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nBenefits for galaxy networks:")
    print("- Stable training regardless of galaxy brightness variations")
    print("- Consistent feature extraction across different galaxy types")
    print("- Faster convergence for weak lensing shear measurement")
    print("- Less sensitive to initialization and hyperparameters")
    print("- Enables processing of mixed galaxy populations effectively")
    
    # Demonstrate inference mode
    print(f"\nDemonstrating inference mode:")
    print(f"Running mean: {bn_layer.running_mean[:5]}")  # Show first 5 values
    print(f"Running var: {bn_layer.running_var[:5]}")    # Show first 5 values
    
    # Test with new galaxy in inference mode
    test_galaxy = create_synthetic_galaxy(size=32, spiral_arms=True, add_noise=True)
    test_galaxy_reshaped = test_galaxy.reshape(1, -1)
    test_normalized = bn_layer.forward(test_galaxy_reshaped, training=False)
    
    print(f"Test galaxy normalized using running statistics: mean={test_normalized.mean():.3f}, std={test_normalized.std():.3f}")

if __name__ == "__main__":
    # Show the problem
    demonstrate_internal_covariate_shift()
    
    # Show the solution
    compare_training_stability()
    
    # Connect to galaxies
    batch_norm_for_galaxy_networks()