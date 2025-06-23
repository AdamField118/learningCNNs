"""
ShearNet Research Bridge - Module 1A: JAX/Flax Foundations
Translate NumPy CNN implementations to JAX/Flax for real ShearNet analysis.

Learning Objectives:
- Convert Day 2 manual convolution to JAX
- Implement Day 6 modern activations in Flax 
- Build Day 5 CNN architecture as trainable Flax model
- Bridge to actual ShearNet analysis workflow

Prerequisites: Days 1-6 complete, JAX/Flax installed
Time Estimate: 2-3 hours
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import flax.linen as nn
from flax.training import train_state
from flax import struct
import optax
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence

# ============================================================================
# Custom TrainState for Batch Normalization Support
# ============================================================================

@struct.dataclass
class TrainStateWithBN(train_state.TrainState):
    """
    Custom TrainState that includes batch normalization statistics.
    
    This extends the standard TrainState to handle batch norm running averages,
    which are needed for proper batch normalization during training and inference.
    """
    batch_stats: dict = None

# ============================================================================
# Manual Convolution Implementation
# ============================================================================

def manual_convolution_jax(image, kernel, stride=1, padding=0):
    """
    JAX version of Day 2 manual_convolution_2d function.
    
    This should work identically to NumPy version, but with JAX arrays.
    Compare to: day02/edge_detection_basics.py manual_convolution_2d()
    
    Args:
        image: JAX array (height, width)
        kernel: JAX array (kernel_h, kernel_w) 
        stride: Step size (same as Day 3 work)
        padding: Padding amount (same as Day 3 work)
    
    Returns:
        JAX array with convolution result
    """
    # Add padding if specified
    if padding > 0:
        pad_width = ((padding, padding), (padding, padding))
        image = jnp.pad(image, pad_width, mode='constant', constant_values=0)
    
    # Get dimensions
    image_h, image_w = image.shape[0], image.shape[1]  # Extract image dimensions
    kernel_h, kernel_w = kernel.shape[0], kernel.shape[1]  # Extract kernel dimensions
    
    # Calculate output dimensions using Day 3 formula
    output_h = (image_h - kernel_h) // stride + 1  # Use convolution output size formula
    output_w = (image_w - kernel_w) // stride + 1  # Use convolution output size formula
    
    # Initialize output array
    output = jnp.zeros((output_h, output_w))  # Create zeros array with output dimensions
    
    # Implement convolution loops (exact same as Day 2)
    for i in range(output_h):
        for j in range(output_w):
           
            start_i = i * stride 
            start_j = j * stride  
            
            window = image[start_i:start_i + kernel_h, start_j:start_j + kernel_w]
            conv_value = jnp.sum(window * kernel)
            
            output = output.at[i, j].set(conv_value)
    
    return output

def fast_convolution_jax(image, kernel, stride=1, padding=0):
    """
    Fast JAX convolution using built-in primitives.
    
    This shows you the "production" version after understanding the mechanics.
    """
    from jax import lax
    
    # Reshape for JAX convolution format
    # JAX expects [batch, height, width, channels] format
    image = image[None, :, :, None]  # Add batch and channel dimensions  
    kernel = kernel[:, :, None, None]  # Add input_channel and output_channel dimensions
    
    # Apply fast convolution
    result = lax.conv_general_dilated(
        image, kernel,
        window_strides=[stride, stride],
        padding=[(padding, padding), (padding, padding)],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    
    # Remove batch and channel dimensions to match manual version
    return result[0, :, :, 0]  # Extract [0, :, :, 0] to get 2D result

class ModernActivationsFlax(nn.Module):
    """
    Flax version of Day 6 ModernActivations class.
    
    Convert activation discoveries to differentiable Flax functions.
    Compare to: day06/activation_functions.py ModernActivations class
    """
    
    def swish(self, x, beta=1.0):
        """
        Convert Day 6 Swish implementation to JAX.
        
     implementation: x * sigmoid(beta * x)
        """
        # Implement Swish using JAX operations
        return x * nn.sigmoid(beta * x)
    
    def gelu(self, x):
        """
        Convert Day 6 GELU implementation to JAX.
        
     implementation used tanh approximation - keep the same math!
        """
        # Implement GELU using Day 6 approximation
        sqrt_2_over_pi = jnp.sqrt(2.0 / jnp.pi)
        inner = sqrt_2_over_pi * (x + 0.044715 * jnp.power(x, 3))
        return 0.5 * x * (1.0 + jnp.tanh(inner))
    
    def mish(self, x):
        """
        Convert Day 6 Mish implementation to JAX.
        
     implementation: x * tanh(softplus(x))
        """
        # Implement Mish with numerical stability
        return x * jnp.tanh(jax.nn.softplus(x))
    
    def elu(self, x, alpha=1.0):
        """Convert Day 6 ELU implementation to JAX."""
        # Implement ELU using jnp.where
        # formula: x if x > 0, alpha * (exp(x) - 1) if x <= 0
        return jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1))

class GalaxyCNN(nn.Module):
    """
    Flax implementation of Day 5 SimpleCNN for galaxy shear measurement.
    
    This bridges architectural understanding to a trainable ShearNet-style model.
    Compare to: day05/feature_map_interpretation.py SimpleCNN class
    """
    features: Sequence[int] = (32, 64, 128)
    activation: str = 'gelu' 
    use_batch_norm: bool = True 
    
    def setup(self):
        """Initialize network components using 6-day insights."""
        # Initialize modern activations module
        self.activations = ModernActivationsFlax()  # ModernActivationsFlax()
        
        # Create convolutional layers (Day 2-3 mechanics)
        self.conv_layers = [nn.Conv(features=feat, kernel_size=(3,3), padding='SAME') 
                           for feat in self.features]  # List of nn.Conv layers
        
        # Create batch normalization layers (Day 6 stabilization)
        self.batch_norms = [nn.BatchNorm() for _ in self.features]  # List of nn.BatchNorm() layers
        
        # Create final layers for g1, g2 prediction (ShearNet's goal)
        self.final_conv = nn.Conv(features=16, kernel_size=(3,3), padding='SAME')  # nn.Conv for final feature extraction
        self.dense = nn.Dense(features=2)  # nn.Dense(features=2) for [g1, g2] output
    
    def __call__(self, x, training=True):
        """
        Forward pass implementing 6 days of CNN learning.
        
        Input: Galaxy images [batch, height, width, channels]
        Output: Shear measurements [batch, 2] for [g1, g2]
        """
        # Process through convolutional layers
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            
            # Apply convolution Day 2-3 understanding)
            x = conv(x)  # conv(x)
            
            # Apply batch normalization Day 6 stabilization)
            if self.use_batch_norm:
                x = bn(x, use_running_average=not training)
            
            # Apply modern activation Day 6 analysis)
            if self.activation == 'gelu':
                x = self.activations.gelu(x)
            elif self.activation == 'swish':
                x = self.activations.swish(x)
            elif self.activation == 'mish':
                x = self.activations.mish(x)
            else:
                x = nn.relu(x)
            
            # Apply pooling except for last layer Day 4 analysis)
            if i < len(self.conv_layers) - 1:
                x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))
        
        # Final processing for shear measurement
        x = self.final_conv(x)  # self.final_conv(x)
        x = self.activations.gelu(x)  # Apply activation
        
        # Global pooling (reduce spatial dimensions)
        x = jnp.mean(x, axis=(1, 2))  # global average pooling
        
        # Final prediction
        x = self.dense(x)  # outputs [g1, g2]
        
        return x

def create_train_state(model, learning_rate, input_shape):
    """
    Create training state for galaxy CNN.
    
    This sets up the infrastructure to actually train model!
    Note: We use TrainStateWithBN to handle batch normalization statistics.
    """
    # Initialize model parameters
    key = random.PRNGKey(42)
    dummy_input = jnp.ones([1] + list(input_shape))
    variables = model.init(key, dummy_input, training=True)
    
    # Create optimizer
    # Use Day 6 insights about learning rates and modern optimizers
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
    
    # Create train state with batch normalization support
    return TrainStateWithBN.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables.get('batch_stats', {}),
        tx=tx
    )

@jit  # JAX magic - compiles function for speed!
def train_step(state, batch, labels):
    """
    Single training step implementing Day 1 backpropagation understanding.
    
    This is where 6 days of learning becomes actual gradient descent!
    """
    def loss_fn(params):
        # Forward pass with current parameters
        if state.batch_stats and len(state.batch_stats) > 0:
            # With batch normalization
            predictions, new_batch_stats = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats}, 
                batch, training=True, mutable=['batch_stats']
            )  # Apply model with mutable batch_stats
        else:
            # Without batch normalization
            predictions = state.apply_fn({'params': params}, batch, training=True)  # Apply model
            new_batch_stats = state.batch_stats
        
        # Compute loss for g1, g2 regression
        loss = jnp.mean((predictions - labels) ** 2)  # MSE loss
        
        return loss, (predictions, new_batch_stats)
    
    # Compute gradients (automatic differentiation!)
    (loss, (predictions, new_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Update parameters using gradients
    state = state.apply_gradients(grads=grads)
    
    # Update batch normalization statistics
    if new_batch_stats and len(new_batch_stats) > 0:
        state = state.replace(batch_stats=new_batch_stats)
    
    return state, loss, predictions

def test_convolution_conversion():
    """
    Test that JAX convolution matches NumPy Day 2 implementation.
    """
    print("=== Testing Convolution Conversion ===")
    
    # Create test data
    np.random.seed(42)
    galaxy = np.random.rand(10, 10).astype(np.float32)  # Create random test image
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)  # Create Sobel X kernel from Day 2 work
    
    # Convert to JAX arrays
    galaxy_jax = jnp.array(galaxy)
    kernel_jax = jnp.array(kernel)
    
    # Test manual convolution
    result_manual = manual_convolution_jax(galaxy_jax, kernel_jax) 
    
    # Test fast convolution
    result_fast = fast_convolution_jax(galaxy_jax, kernel_jax)
    
    # Compare results
    print(f"Manual result shape: {result_manual.shape}")
    print(f"Fast result shape: {result_fast.shape}")
    
    # Check if they match
    matches = jnp.allclose(result_manual, result_fast, atol=1e-6) 
    print(f"Results match: {matches}")
    
    if not matches:
        print("Debug: Check implementations!")
        print(f"Max difference: {jnp.max(jnp.abs(result_manual - result_fast))}")
    
    return result_manual, result_fast

def test_activation_conversion():
    """
    Test that JAX activations match NumPy Day 6 implementations.
    """
    print("=== Testing Activation Conversion ===")
    
    # Create test data
    x = jnp.linspace(-3, 3, 100)
    
    # Initialize Flax activations
    activations = ModernActivationsFlax()
    
    # Test each activation function
    swish_result = activations.swish(x)
    gelu_result = activations.gelu(x)
    mish_result = activations.mish(x)
    elu_result = activations.elu(x)
    
    # Print results and compare to Day 6 analysis
    print(f"Swish range: [{swish_result.min():.3f}, {swish_result.max():.3f}]")
    print(f"GELU range: [{gelu_result.min():.3f}, {gelu_result.max():.3f}]")
    print(f"Mish range: [{mish_result.min():.3f}, {mish_result.max():.3f}]")
    print(f"ELU range: [{elu_result.min():.3f}, {elu_result.max():.3f}]")
    
    # Visualize to confirm they match Day 6 plots
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.plot(x, swish_result, label='Swish JAX')
    plt.title('Swish')
    plt.grid(True)
    
    plt.subplot(1, 4, 2)
    plt.plot(x, gelu_result, label='GELU JAX', color='green')
    plt.title('GELU')
    plt.grid(True)
    
    plt.subplot(1, 4, 3)
    plt.plot(x, mish_result, label='Mish JAX', color='red')
    plt.title('Mish')
    plt.grid(True)
    
    plt.subplot(1, 4, 4)
    plt.plot(x, elu_result, label='ELU JAX', color='purple')
    plt.title('ELU')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return swish_result, gelu_result, mish_result, elu_result

def test_galaxy_cnn():
    """
    Test complete GalaxyCNN on synthetic data.
    """
    print("=== Testing Galaxy CNN ===")
    
    # Create model with optimal Day 6 settings
    model = GalaxyCNN(features=(16, 32, 64), activation='gelu')
    print("✓ Model created successfully")
    
    # Create synthetic galaxy batch
    key = random.PRNGKey(42)
    batch_size, height, width, channels = 4, 48, 48, 1
    
    # Generate synthetic galaxies (use Day 2-3 insights)
    galaxies = random.normal(key, (batch_size, height, width, channels))
    print(f"✓ Generated synthetic galaxies: {galaxies.shape}")
    
    # Generate synthetic g1, g2 labels (small shear values)
    g1_g2_labels = random.normal(key, (batch_size, 2)) * 0.1
    print(f"✓ Generated shear labels: {g1_g2_labels.shape}")
    
    # Create training state
    print("Creating training state...")
    state = create_train_state(model, learning_rate=1e-3, input_shape=(height, width, channels))
    print("✓ Training state created successfully")
    
    # Run one training step
    print("Running training step...")
    new_state, loss, predictions = train_step(state, galaxies, g1_g2_labels)
    print("✓ Training step completed successfully")
    
    # Analyze results
    print(f"\nResults:")
    print(f"Model works: {predictions.shape == (batch_size, 2)}")
    print(f"Input shape: {galaxies.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Initial loss: {loss:.4f}")
    print(f"g1, g2 predictions:\n{predictions}")
    
    print("\nSuccess! 6-day CNN understanding now runs in JAX!")
    
    return model, state, predictions

# ============================================================================
# MAIN EXECUTION - Run Tests
# ============================================================================

if __name__ == "__main__":
    print("ShearNet Research Bridge - Module 1A")
    print("Converting 6-day CNN mastery to JAX/Flax!")
    
    # Run all tests
    print("\n1. Testing convolution conversion...")
    test_convolution_conversion()
    
    print("\n2. Testing activation conversion...")
    test_activation_conversion()
    
    print("\n3. Testing complete Galaxy CNN...")
    test_galaxy_cnn()
    
    print("\nImplementation Tasks:")
    print("✓ Complete all TODOs in order")
    print("✓ Test each component before moving to the next")
    print("✓ Compare results to Day 2-6 NumPy implementations")
    print("✓ Verify the Galaxy CNN produces reasonable g1, g2 predictions")
    
    print("\nAfter completing this module:")
    print("→ Module 1B: Your custom astronomical kernels in Flax")
    print("→ Module 2: Load and analyze actual ShearNet architecture")
    print("→ Module 3: Performance benchmarking and improvements")