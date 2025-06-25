"""
Complete CNN Architecture incorporating all concepts from Days 1-6.

This module builds a state-of-the-art CNN that demonstrates:
- Modern activation functions (Day 6)
- Attention mechanisms (Day 6) 
- Batch normalization (Day 6)
- Residual connections (Day 6)
- Advanced pooling strategies (Day 4)
- Dilated convolutions (Day 3)
- Proper initialization and training (Day 1-2)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Tuple, Optional, Any

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.output_system import log_print, log_experiment_start, log_experiment_end, save_plot
from utils.test_data import create_synthetic_galaxy
from day06.activation_functions import ModernActivations
from day06.attention_mechanisms import SpatialAttention, ChannelAttention
from day06.batch_normalization import BatchNormLayer
from day06.residual_connections import ResidualBlock
from day04.pooling_operations import max_pooling_2d
from day03.convolution_mechanics import manual_convolution_2d_extended
from day02.edge_detection_basics import manual_convolution_2d

class ModernCNNBlock:
    """
    A modern CNN block that incorporates all Day 1-6 concepts.
    
    Architecture: Conv -> BatchNorm -> Modern Activation -> Attention -> Residual
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dilation: int = 1,
                 activation: str = 'swish', use_attention: bool = True):
        """
        Initialize modern CNN block.
        
        TODO: Implement the __init__ method
        - Store all the parameters
        - Initialize weight matrices with proper shapes using He initialization
        - Create batch norm layer
        - Create attention mechanisms if use_attention=True
        - Choose activation function based on activation parameter
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels  
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            dilation: Convolution dilation
            activation: Activation function name ('swish', 'gelu', 'mish', etc.)
            use_attention: Whether to apply attention mechanisms
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.activation_name = activation
        self.use_attention = use_attention
        
        # TODO: Initialize convolution weights with He initialization
        # fan_in = kernel_size * kernel_size * in_channels
        # std = np.sqrt(2.0 / fan_in)
        # self.conv_weights = np.random.randn(kernel_size, kernel_size) * std
        
        # TODO: Initialize batch normalization
        # self.batch_norm = BatchNormLayer(out_channels)
        
        # TODO: Set up activation function
        # self.activations = ModernActivations()
        
        # TODO: Set up attention if requested
        # if self.use_attention:
        #     self.spatial_attention = SpatialAttention(out_channels)
        #     self.channel_attention = ChannelAttention(out_channels)
        
        # TODO: Set up residual connection weights if needed
        # if in_channels != out_channels:
        #     self.residual_projection = ...
        
        pass
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the modern CNN block.
        
        TODO: Implement the forward pass with this order:
        1. Store input for residual connection
        2. Apply convolution (use manual_convolution_2d_extended for dilation support)
        3. Apply batch normalization 
        4. Apply modern activation function (swish, gelu, mish, etc.)
        5. Apply attention mechanisms (spatial + channel) if enabled
        6. Add residual connection if input/output shapes match
        7. Return result
        
        Args:
            x: Input feature map [batch, height, width, channels] or [height, width]
            training: Whether in training mode
            
        Returns:
            Output feature map after all operations
        """
        # TODO: Implement complete forward pass
        
        # Store input for residual connection
        identity = x.copy()
        
        # Step 1: Convolution
        # TODO: Apply convolution with dilation support
        # conv_out = manual_convolution_2d_extended(x, self.conv_weights, 
        #                                          stride=self.stride, 
        #                                          padding=self.padding, 
        #                                          dilation=self.dilation)
        
        # Step 2: Batch Normalization
        # TODO: Apply batch normalization
        # bn_out = self.batch_norm.forward(conv_out, training=training)
        
        # Step 3: Activation
        # TODO: Apply chosen activation function
        # if self.activation_name == 'swish':
        #     activated = self.activations.swish(bn_out)
        # elif self.activation_name == 'gelu':
        #     activated = self.activations.gelu(bn_out)
        # elif self.activation_name == 'mish':
        #     activated = self.activations.mish(bn_out)
        # else:
        #     activated = self.activations.leaky_relu(bn_out)  # fallback
        
        # Step 4: Attention (if enabled)
        # TODO: Apply attention mechanisms
        # if self.use_attention:
        #     # Apply spatial attention
        #     spatial_attended, spatial_weights = self.spatial_attention.forward(activated)
        #     # Apply channel attention  
        #     final_attended, channel_weights = self.channel_attention.forward(spatial_attended)
        #     activated = final_attended
        
        # Step 5: Residual connection
        # TODO: Add residual connection with proper dimension handling
        # if identity.shape == activated.shape:
        #     output = activated + identity
        # elif hasattr(self, 'residual_projection'):
        #     projected_identity = manual_convolution_2d(identity, self.residual_projection)
        #     if projected_identity.shape == activated.shape:
        #         output = activated + projected_identity
        #     else:
        #         output = activated  # Skip connection if shapes don't match
        # else:
        #     output = activated
        
        # For now, return input unchanged
        return x

class GalaxyShearCNN:
    """
    Complete CNN for galaxy shear estimation using all modern techniques.
    
    This demonstrates a production-ready architecture that could compete
    with ShearNet while showing all the concepts we've learned.
    """
    
    def __init__(self):
        """
        Initialize the complete galaxy shear estimation CNN.
        
        TODO: Design the architecture with these components:
        - Input processing layer (1 -> 16 channels)
        - ModernCNNBlock 1: 16 -> 32 channels, basic features
        - ModernCNNBlock 2: 32 -> 64 channels, with pooling
        - ModernCNNBlock 3: 64 -> 128 channels, with attention
        - Global average pooling 
        - Dense layers for regression (g1, g2, sigma, flux)
        - Initialize all weights properly with He initialization
        """
        log_print("Initializing Galaxy Shear CNN with modern architecture...")
        
        # TODO: Initialize the architecture
        # Architecture design:
        # Input (53x53x1) -> Block1 (53x53x16) -> Block2 (26x26x32) -> 
        # Block3 (13x13x64) -> Block4 (6x6x128) -> GlobalPool (128) -> 
        # Dense (64) -> Output (4: g1, g2, sigma, flux)
        
        # self.input_conv = ModernCNNBlock(1, 16, kernel_size=3, activation='swish')
        # self.block1 = ModernCNNBlock(16, 32, kernel_size=3, activation='swish')  
        # self.block2 = ModernCNNBlock(32, 64, kernel_size=3, activation='gelu', use_attention=True)
        # self.block3 = ModernCNNBlock(64, 128, kernel_size=3, activation='mish', use_attention=True)
        
        # TODO: Initialize dense layers for final prediction
        # self.dense1_weights = He initialization for (128, 64)
        # self.dense1_bias = zeros (64,)
        # self.output_weights = He initialization for (64, 4) 
        # self.output_bias = zeros (4,)
        
        log_print("Architecture: Input -> 16 -> 32 -> 64 -> 128 -> 64 -> 4 output")
        pass
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the complete CNN.
        
        TODO: Implement the complete forward pass:
        1. Ensure input is proper shape [H, W] or [H, W, 1]
        2. Process through input_conv layer
        3. Apply Block1 with pooling 
        4. Apply Block2 with pooling
        5. Apply Block3 with pooling
        6. Global average pooling to get feature vector
        7. Dense layer 1 with activation
        8. Output layer for [g1, g2, sigma, flux]
        9. Return predictions
        
        Args:
            x: Input galaxy image(s) - shape (H, W) or (H, W, 1)
            training: Whether in training mode
            
        Returns:
            Predictions [g1, g2, sigma, flux] - shape (4,)
        """
        # TODO: Implement complete forward pass
        
        # Ensure proper input shape
        if x.ndim == 2:
            x = np.expand_dims(x, axis=-1)  # Add channel dimension
        
        log_print(f"Input shape: {x.shape}")
        
        # TODO: Forward through all blocks
        # x = self.input_conv.forward(x, training)
        # x = max_pooling_2d(x, pool_size=2)  # Downsample
        # 
        # x = self.block1.forward(x, training) 
        # x = max_pooling_2d(x, pool_size=2)  # Downsample
        #
        # x = self.block2.forward(x, training)
        # x = max_pooling_2d(x, pool_size=2)  # Downsample
        #
        # x = self.block3.forward(x, training)
        # 
        # # Global average pooling
        # features = np.mean(x, axis=(0, 1))  # Average over spatial dimensions
        #
        # # Dense layers
        # dense1 = np.dot(features, self.dense1_weights) + self.dense1_bias
        # dense1_activated = self.activations.swish(dense1)
        #
        # output = np.dot(dense1_activated, self.output_weights) + self.output_bias
        
        # For now, return dummy predictions
        return np.array([0.1, -0.05, 1.2, 2.5])  # [g1, g2, sigma, flux]
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute loss with proper weighting for different parameters.
        
        TODO: Implement sophisticated loss computation:
        - Higher weight for g1/g2 (shear parameters - most important)
        - Lower weight for sigma/flux (shape parameters - less critical)
        - Consider using Huber loss for robustness to outliers
        - Maybe add L2 regularization on weights
        
        Args:
            predictions: Model predictions [g1, g2, sigma, flux]
            targets: Ground truth values [g1, g2, sigma, flux]
            
        Returns:
            Weighted loss value
        """
        # TODO: Implement sophisticated loss
        # weights = np.array([10.0, 10.0, 1.0, 0.1])  # High weight on g1, g2
        # 
        # # Weighted MSE
        # mse_per_param = (predictions - targets) ** 2
        # weighted_loss = np.sum(weights * mse_per_param)
        #
        # # Add L2 regularization if desired
        # l2_reg = 0.0001 * np.sum([np.sum(w**2) for w in self.get_weights()])
        # 
        # return weighted_loss + l2_reg
        
        # For now, simple MSE
        return np.mean((predictions - targets) ** 2)

def demonstrate_architecture_evolution() -> None:
    """
    Show how our architecture evolved from Day 1 simple networks to Day 7 modern CNN.
    """
    log_print("=== CNN Architecture Evolution: Days 1-7 ===", level="SUBHEADER")
    
    # TODO: Implement comparison showing parameter counts and capabilities
    
    architectures = {
        "Day 1 - Simple MLP": {
            "params": 1000,
            "layers": "Input(2809) -> Dense(128) -> Dense(64) -> Output(4)",
            "concepts": ["Basic neural network", "ReLU activation"]
        },
        "Day 2 - Basic CNN": {
            "params": 5000, 
            "layers": "Input -> Conv(16) -> Pool -> Conv(32) -> Pool -> Dense -> Output",
            "concepts": ["Convolution", "Pooling", "Feature extraction"]
        },
        "Day 3 - Advanced Conv": {
            "params": 8000,
            "layers": "Same as Day 2 + Stride/Padding/Dilation options",
            "concepts": ["Dilation", "Receptive fields", "Parameter efficiency"]
        },
        "Day 4 - Smart Pooling": {
            "params": 6000,
            "layers": "Optimized pooling strategies",
            "concepts": ["Max vs Average pooling", "Translation invariance"]
        },
        "Day 5 - Multi-layer": {
            "params": 15000,
            "layers": "Deeper network with feature progression",
            "concepts": ["Feature hierarchies", "Multi-scale processing"]
        },
        "Day 6 - Modern Components": {
            "params": 25000,
            "layers": "Added BatchNorm + Attention + Residuals + Modern activations",
            "concepts": ["Training stability", "Attention", "Skip connections", "Swish/GELU"]
        },
        "Day 7 - Complete Modern CNN": {
            "params": 50000,
            "layers": "Integrated all concepts into production-ready architecture",
            "concepts": ["All of the above", "Production considerations", "Real applications"]
        }
    }
    
    for name, info in architectures.items():
        log_print(f"\n{name}:")
        log_print(f"  Parameters: ~{info['params']:,}")
        log_print(f"  Architecture: {info['layers']}")
        log_print(f"  Key concepts: {', '.join(info['concepts'])}")
    
    # TODO: Create visualization showing the evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    days = list(range(1, 8))
    params = [arch["params"] for arch in architectures.values()]
    
    ax1.plot(days, params, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Parameter Count')
    ax1.set_title('CNN Complexity Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Concepts accumulated over time
    concept_counts = [len(arch["concepts"]) for arch in architectures.values()]
    ax2.bar(days, concept_counts, alpha=0.7, color='green')
    ax2.set_xlabel('Day') 
    ax2.set_ylabel('Number of Key Concepts')
    ax2.set_title('Knowledge Accumulation')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot('architecture_evolution.png')

def test_on_galaxy_data() -> Dict[str, Any]:
    """
    Test our complete CNN on galaxy data and compare with previous approaches.
    """
    log_print("=== Testing Complete CNN on Galaxy Data ===", level="SUBHEADER")
    
    # Create test dataset
    galaxies = {}
    targets = {}
    
    # TODO: Create comprehensive test cases
    test_cases = [
        ("clean_spiral", {"add_noise": False, "spiral_arms": True}),
        ("noisy_spiral", {"add_noise": True, "spiral_arms": True, "noise_level": 0.1}),
        ("clean_elliptical", {"add_noise": False, "spiral_arms": False}),
        ("very_noisy", {"add_noise": True, "spiral_arms": True, "noise_level": 0.3}),
    ]
    
    for name, params in test_cases:
        galaxy = create_synthetic_galaxy(size=53, **params)
        galaxies[name] = galaxy
        # Create realistic targets
        targets[name] = np.array([
            np.random.uniform(-0.2, 0.2),  # g1
            np.random.uniform(-0.2, 0.2),  # g2  
            np.random.uniform(0.8, 1.5),   # sigma
            np.random.uniform(1.5, 3.0)    # flux
        ])
    
    # TODO: Test different architectures
    results = {}
    
    # Test our modern CNN
    log_print("Testing Day 7 Modern CNN...")
    modern_cnn = GalaxyShearCNN()
    
    modern_results = {}
    for name, galaxy in galaxies.items():
        pred = modern_cnn.forward(galaxy, training=False)
        loss = modern_cnn.compute_loss(pred, targets[name])
        modern_results[name] = {"prediction": pred, "loss": loss, "target": targets[name]}
        log_print(f"  {name}: Loss = {loss:.6f}")
    
    results["Modern CNN"] = modern_results
    
    # TODO: Compare with simpler baselines
    # You could implement a simple Day 1 MLP for comparison
    
    return results

def analyze_attention_patterns() -> None:
    """
    Analyze what the attention mechanisms learn when applied to galaxies.
    """
    log_print("=== Analyzing Attention Patterns ===", level="SUBHEADER")
    
    # TODO: Create attention visualization
    galaxy = create_synthetic_galaxy(size=53, spiral_arms=True)
    
    # TODO: If you implement attention in your CNN, extract and visualize the attention maps
    log_print("Creating synthetic attention analysis...")
    
    # Simulate what attention might look like
    attention_map = np.random.rand(53, 53)
    # Make it focus on galaxy center and spiral arms
    center = 26
    y, x = np.ogrid[:53, :53]
    # Higher attention near center
    attention_map += 2.0 * np.exp(-((x - center)**2 + (y - center)**2) / (2 * 10**2))
    # Normalize
    attention_map = attention_map / np.max(attention_map)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(galaxy, cmap='gray')
    ax1.set_title('Original Galaxy')
    ax1.axis('off')
    
    ax2.imshow(attention_map, cmap='Reds')
    ax2.set_title('Spatial Attention Map')
    ax2.axis('off')
    
    # Overlay attention on galaxy
    ax3.imshow(galaxy, cmap='gray', alpha=0.7)
    ax3.imshow(attention_map, cmap='Reds', alpha=0.5)
    ax3.set_title('Attention Overlay')
    ax3.axis('off')
    
    plt.tight_layout()
    save_plot('attention_analysis.png')
    
    log_print("Attention analysis shows focus on:")
    log_print("- Galaxy center (high signal region)")
    log_print("- Potential spiral arm locations")
    log_print("- Edge regions important for shear measurement")

def main() -> None:
    """Main execution function for Day 7 experiments."""
    log_experiment_start(7, "Complete CNN Architecture and Real Applications")
    
    # Demonstrate evolution from simple to complex
    demonstrate_architecture_evolution()
    
    # Test our complete architecture  
    test_results = test_on_galaxy_data()
    
    # Analyze what the network learns
    analyze_attention_patterns()
    
    log_print("=== Day 7 Summary ===", level="SUBHEADER")
    log_print("You've now built a complete, modern CNN from scratch!")
    log_print("Key achievements:")
    log_print("- Integrated all concepts from Days 1-6")
    log_print("- Built production-ready architecture") 
    log_print("- Connected theory to real applications")
    log_print("- Ready to understand and improve ShearNet!")
    
    log_experiment_end(7)

if __name__ == "__main__":
    main()