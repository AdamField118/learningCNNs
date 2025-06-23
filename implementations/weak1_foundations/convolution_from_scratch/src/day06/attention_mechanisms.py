"""
Attention mechanisms for spatial data and galaxy analysis.

This module explores how attention helps networks focus on important
galaxy features and adaptively weight different spatial regions.
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
from day01.neural_network_foundations import activation_function_sigmoid
from day06.activation_functions import ModernActivations

class SpatialAttention:
    """
    Spatial attention mechanism for galaxy images.
    
    Key insight: Learn WHERE to look in the galaxy image.
    Different regions (spiral arms, center, background) have different importance.
    """
    
    def __init__(self, feature_channels):
        """
        Initialize spatial attention module.
        
        Args:
            feature_channels: Number of channels in the feature map
        """
        self.feature_channels = feature_channels
        
        # Initialize attention parameters
        # Spatial attention learns a single attention map across all channels
        # Small 3x3 conv filter for computing spatial attention from pooled features
        self.spatial_conv_weights = np.random.normal(0, 0.1, (3, 3, 2, 1))  # [H, W, in_channels=2, out_channels=1]
        
    def forward(self, feature_map):
        """
        Apply spatial attention to feature map.
        
        Args:
            feature_map: Input features [height, width, channels]
            
        Returns:
            Attended features and attention weights
        """
        height, width, channels = feature_map.shape
        
        # Compute spatial attention weights
        # Step 1: Aggregate across channels (max and mean pooling)
        max_pooled = np.max(feature_map, axis=-1, keepdims=True)    # Max across channel dimension
        mean_pooled = np.mean(feature_map, axis=-1, keepdims=True)  # Mean across channel dimension
        
        # Step 2: Concatenate max and mean
        pooled_features = np.concatenate([max_pooled, mean_pooled], axis=-1)  # Stack max and mean [H, W, 2]
        
        # Step 3: Apply convolution to get attention map
        # This learns which spatial locations are important
        from day02.edge_detection_basics import manual_convolution_2d
        
        # Apply 3x3 conv with the learned weights
        # For simplicity, we'll use a learned 3x3 filter
        attention_kernel = np.sum(self.spatial_conv_weights, axis=-1)  # Sum over output channels
        attention_kernel = np.sum(attention_kernel, axis=-1)           # Sum over input channels
        
        # Apply convolution to the pooled features (using first channel as primary)
        attention_scores = manual_convolution_2d(pooled_features[:, :, 0], attention_kernel)
        
        # Step 4: Apply sigmoid to get attention weights (0 to 1)
        # Flatten for processing, then reshape back
        scores_flat = attention_scores.flatten()
        attention_weights_flat = activation_function_sigmoid(scores_flat)
        attention_weights = attention_weights_flat.reshape(attention_scores.shape)
        
        # Resize attention_weights to match feature_map if needed (due to convolution size reduction)
        if attention_weights.shape != (height, width):
            # Simple resize by padding or cropping
            att_h, att_w = attention_weights.shape
            if att_h < height or att_w < width:
                # Pad with edge values
                padded = np.ones((height, width)) * np.mean(attention_weights)
                start_h = (height - att_h) // 2
                start_w = (width - att_w) // 2
                padded[start_h:start_h+att_h, start_w:start_w+att_w] = attention_weights
                attention_weights = padded
            else:
                # Crop from center
                start_h = (att_h - height) // 2
                start_w = (att_w - width) // 2
                attention_weights = attention_weights[start_h:start_h+height, start_w:start_w+width]
        
        # Step 5: Apply attention to original features
        # Expand attention weights to match all channels
        attention_weights_expanded = np.expand_dims(attention_weights, axis=-1)  # [H, W, 1]
        attended_features = feature_map * attention_weights_expanded  # Element-wise multiply
        
        return attended_features, attention_weights

class ChannelAttention:
    """
    Channel attention mechanism (Squeeze-and-Excitation style).
    
    Key insight: Learn WHICH feature channels are important.
    Some channels might detect spiral arms, others detect galaxy centers, etc.
    """
    
    def __init__(self, num_channels, reduction_ratio=4):
        """
        Initialize channel attention module.
        
        Args:
            num_channels: Number of input channels
            reduction_ratio: How much to reduce in the bottleneck
        """
        self.num_channels = num_channels
        self.reduced_channels = max(num_channels // reduction_ratio, 1)
        
        # Initialize squeeze-and-excitation weights
        # First fully connected layer (squeeze)
        self.fc1_weights = np.random.normal(0, 0.1, (self.num_channels, self.reduced_channels))
        self.fc1_bias = np.zeros(self.reduced_channels)
        
        # Second fully connected layer (excitation)
        self.fc2_weights = np.random.normal(0, 0.1, (self.reduced_channels, self.num_channels))
        self.fc2_bias = np.zeros(self.num_channels)
        
    def forward(self, feature_map):
        """
        Apply channel attention to feature map.
        
        Args:
            feature_map: Input features [height, width, channels]
            
        Returns:
            Channel-attended features and attention weights
        """
        height, width, channels = feature_map.shape
        
        # Squeeze: Global Average Pooling
        # Compress spatial dimensions to get channel-wise statistics
        channel_stats = np.mean(feature_map, axis=(0, 1))  # Mean over height and width dimensions
        
        # Excitation: Learn channel importance
        # Step 1: Reduce dimensionality with ReLU
        fc1_output = np.dot(channel_stats, self.fc1_weights) + self.fc1_bias
        reduced = np.maximum(0, fc1_output)  # ReLU activation
        
        # Step 2: Expand back and apply sigmoid
        fc2_output = np.dot(reduced, self.fc2_weights) + self.fc2_bias
        channel_weights = activation_function_sigmoid(fc2_output)  # Apply sigmoid
        
        # Step 3: Apply channel attention
        # Reshape channel weights to broadcast properly
        channel_weights_reshaped = channel_weights.reshape(1, 1, -1)  # [1, 1, channels]
        attended_features = feature_map * channel_weights_reshaped  # Multiply each channel by its weight
        
        return attended_features, channel_weights

class GalaxyAttentionNetwork:
    """
    Complete attention network for galaxy analysis.
    
    Combines spatial and channel attention to focus on important
    galaxy features for tasks like shear measurement.
    """
    
    def __init__(self):
        """Initialize galaxy attention network."""
        # Initialize basic feature extraction
        self.edge_kernels = self._create_edge_kernels()
        num_feature_channels = len(self.edge_kernels)
        
        # Initialize attention modules
        self.spatial_attention = SpatialAttention(num_feature_channels)   # Create spatial attention
        self.channel_attention = ChannelAttention(num_feature_channels)   # Create channel attention
        
    def _create_edge_kernels(self):
        """Create edge detection kernels for feature extraction."""
        # Create multiple edge detection kernels
        # These will create our initial feature channels
        kernels = {
            'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
            'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
            'diagonal1': np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32),
            'diagonal2': np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float32),
        }
        return kernels
    
    def extract_features(self, galaxy_image):
        """Extract multi-channel features from galaxy."""
        from day02.edge_detection_basics import manual_convolution_2d
        
        # Apply all edge kernels to create feature channels
        feature_channels = []
        
        for name, kernel in self.edge_kernels.items():
            # Apply convolution and store result
            feature = manual_convolution_2d(galaxy_image, kernel)
            feature_channels.append(feature)
        
        # Stack into multi-channel feature map
        feature_map = np.stack(feature_channels, axis=-1)  # [H, W, C]
        
        return feature_map
    
    def forward(self, galaxy_image):
        """
        Forward pass with attention mechanisms.
        
        Args:
            galaxy_image: Input galaxy image
            
        Returns:
            Dictionary with features and attention weights
        """
        # Extract initial features
        features = self.extract_features(galaxy_image)
        
        # Apply channel attention first
        channel_attended, channel_weights = self.channel_attention.forward(features)
        
        # Apply spatial attention
        final_features, spatial_weights = self.spatial_attention.forward(channel_attended)
        
        return {
            'original_features': features,
            'channel_attended': channel_attended,
            'final_features': final_features,
            'channel_attention': channel_weights,
            'spatial_attention': spatial_weights
        }

def demonstrate_attention_concept():
    """
    Demonstrate the core concept of attention with a simple example.
    """
    print("=== Understanding Attention: Core Concept ===")
    
    # Create a simple example showing attention weights
    # Simulate a 1D sequence with different importance values
    sequence_length = 10
    sequence_values = np.random.randn(sequence_length)
    
    # Create attention scores (some positions more important)
    # Higher scores = more important positions
    attention_scores = np.array([0.1, 0.2, 0.8, 2.0, 1.5, 0.3, 0.1, 0.9, 0.4, 0.2])
    
    # Apply softmax to get attention weights
    exp_scores = np.exp(attention_scores - np.max(attention_scores))  # Subtract max for numerical stability
    attention_weights = exp_scores / np.sum(exp_scores)  # Softmax over attention_scores
    
    # Apply attention to get weighted output
    attended_output = np.sum(sequence_values * attention_weights)  # Weighted sum using attention weights
    uniform_output = np.mean(sequence_values)   # Simple average for comparison
    
    # Visualize the difference
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    bars1 = plt.bar(range(sequence_length), sequence_values, alpha=0.7, label='Original Values')
    # Color bars based on their values
    for i, bar in enumerate(bars1):
        if sequence_values[i] > 0:
            bar.set_color('blue')
        else:
            bar.set_color('red')
    plt.title('Original Sequence Values')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 4, 2)
    bars2 = plt.bar(range(sequence_length), attention_weights, alpha=0.7, color='orange', label='Attention Weights')
    plt.title('Attention Weights\n(Where to Focus)')
    plt.xlabel('Position')
    plt.ylabel('Attention Weight')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 4, 3)
    weighted_values = sequence_values * attention_weights
    bars3 = plt.bar(range(sequence_length), weighted_values, alpha=0.7, color='green', label='Weighted Values')
    plt.title('Attention-Weighted Values')
    plt.xlabel('Position')
    plt.ylabel('Weighted Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 4, 4)
    comparison = [uniform_output, attended_output]
    labels = ['Uniform\nAverage', 'Attention\nWeighted']
    colors = ['lightblue', 'lightcoral']
    bars4 = plt.bar(labels, comparison, alpha=0.7, color=colors)
    plt.title('Output Comparison')
    plt.ylabel('Output Value')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars4, comparison):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Uniform average: {uniform_output:.3f}")
    print(f"Attention-weighted: {attended_output:.3f}")
    print(f"Difference: {abs(attended_output - uniform_output):.3f}")
    
    print("\nKey Insight:")
    print("- Attention allows the network to focus on important positions!")
    print("- Instead of treating all inputs equally, it learns what matters")
    print("- The attention weights sum to 1.0, ensuring proper normalization")
    print(f"- Attention weights sum: {np.sum(attention_weights):.6f}")
    
    # Show which positions got highest attention
    top_positions = np.argsort(attention_weights)[-3:][::-1]
    print(f"\nTop 3 attended positions: {top_positions}")
    for pos in top_positions:
        print(f"  Position {pos}: weight={attention_weights[pos]:.3f}, value={sequence_values[pos]:.3f}")

def test_attention_on_galaxies():
    """
    Test attention mechanisms on galaxy images.
    """
    print("=== Testing Attention on Galaxy Features ===")
    
    # Create different galaxy types
    galaxies = {
        'spiral': create_synthetic_galaxy(size=48, spiral_arms=True, add_noise=False),
        'elliptical': create_synthetic_galaxy(size=48, spiral_arms=False, add_noise=False),
        'noisy': create_synthetic_galaxy(size=48, spiral_arms=True, add_noise=True)
    }
    
    # Initialize attention network
    attention_net = GalaxyAttentionNetwork()
    
    results = {}
    
    for galaxy_type, galaxy in galaxies.items():
        print(f"\nAnalyzing {galaxy_type} galaxy...")
        
        # Process with attention
        result = attention_net.forward(galaxy)
        results[galaxy_type] = result
        
        # Analyze attention patterns
        spatial_attention = result['spatial_attention']
        channel_attention = result['channel_attention']
        
        print(f"  Spatial attention range: [{np.min(spatial_attention):.3f}, {np.max(spatial_attention):.3f}]")
        print(f"  Spatial attention focus: {np.std(spatial_attention):.3f} (higher = more focused)")
        
        # Find the most attended spatial region
        max_attention_idx = np.unravel_index(np.argmax(spatial_attention), spatial_attention.shape)
        center = (spatial_attention.shape[0]//2, spatial_attention.shape[1]//2)
        distance_from_center = np.sqrt((max_attention_idx[0] - center[0])**2 + (max_attention_idx[1] - center[1])**2)
        
        print(f"  Peak attention at: {max_attention_idx}, distance from center: {distance_from_center:.1f} pixels")
        
        # Analyze channel preferences
        channel_names = ['SobelX', 'SobelY', 'Diagonal1', 'Diagonal2']
        print(f"  Channel preferences:")
        for i, (name, weight) in enumerate(zip(channel_names, channel_attention)):
            print(f"    {name}: {weight:.3f}")
        
        dominant_channel = np.argmax(channel_attention)
        print(f"  Dominant channel: {channel_names[dominant_channel]} ({channel_attention[dominant_channel]:.3f})")
        
        # Compute attention efficiency (how much it differs from uniform)
        uniform_spatial = np.ones_like(spatial_attention) / spatial_attention.size
        spatial_entropy = -np.sum(spatial_attention * np.log(spatial_attention + 1e-8))
        uniform_entropy = -np.sum(uniform_spatial * np.log(uniform_spatial + 1e-8))
        attention_efficiency = (uniform_entropy - spatial_entropy) / uniform_entropy * 100
        
        print(f"  Attention efficiency: {attention_efficiency:.1f}% (0%=uniform, 100%=perfectly focused)")
    
    # Visualize attention maps
    visualize_attention_maps(galaxies, results)
    
    return results

def visualize_attention_maps(galaxies, attention_results):
    """
    Visualize where the attention mechanism focuses.
    """
    print("\n=== Visualizing Galaxy Attention Maps ===")
    
    # Create comprehensive attention visualization
    fig, axes = plt.subplots(len(galaxies), 5, figsize=(20, 4*len(galaxies)))
    if len(galaxies) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (galaxy_type, galaxy) in enumerate(galaxies.items()):
        result = attention_results[galaxy_type]
        
        # Plot original galaxy
        axes[i, 0].imshow(galaxy, cmap='gray')
        axes[i, 0].set_title(f'{galaxy_type.title()} Galaxy')
        axes[i, 0].axis('off')
        
        # Plot feature channels (sum across channels)
        feature_sum = np.sum(np.abs(result['original_features']), axis=-1)
        im1 = axes[i, 1].imshow(feature_sum, cmap='hot')
        axes[i, 1].set_title('Feature Response\n(All Channels)')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        
        # Plot spatial attention
        spatial_attention = result['spatial_attention']
        im2 = axes[i, 2].imshow(spatial_attention, cmap='Reds', alpha=0.8)
        axes[i, 2].imshow(galaxy, cmap='gray', alpha=0.3)  # Overlay on galaxy
        axes[i, 2].set_title('Spatial Attention\n(Where to Look)')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
        
        # Plot attended features
        attended_sum = np.sum(np.abs(result['final_features']), axis=-1)
        im3 = axes[i, 3].imshow(attended_sum, cmap='hot')
        axes[i, 3].set_title('Final Attended Features')
        axes[i, 3].axis('off')
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)
        
        # Plot channel attention as bar chart
        channel_weights = result['channel_attention']
        channel_names = ['SobelX', 'SobelY', 'Diag1', 'Diag2']
        bars = axes[i, 4].bar(channel_names, channel_weights, 
                             color=['red', 'green', 'blue', 'orange'], alpha=0.7)
        axes[i, 4].set_title('Channel Attention\n(Which Features)')
        axes[i, 4].set_ylabel('Attention Weight')
        axes[i, 4].tick_params(axis='x', rotation=45)
        axes[i, 4].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, weight in zip(bars, channel_weights):
            axes[i, 4].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze what the network learned to focus on
    print("\nDetailed Attention Analysis:")
    print("=" * 60)
    
    for galaxy_type, result in attention_results.items():
        spatial_att = result['spatial_attention']
        channel_att = result['channel_attention']
        
        # Find where attention focuses most
        max_attention_pos = np.unravel_index(np.argmax(spatial_att), spatial_att.shape)
        center = (spatial_att.shape[0]//2, spatial_att.shape[1]//2)
        
        # Compute attention statistics
        attention_mass_center = np.array([
            np.sum(np.arange(spatial_att.shape[0])[:, None] * spatial_att) / np.sum(spatial_att),
            np.sum(np.arange(spatial_att.shape[1])[None, :] * spatial_att) / np.sum(spatial_att)
        ])
        
        # Analyze spatial distribution
        top_5_percent_threshold = np.percentile(spatial_att, 95)
        focused_region_size = np.sum(spatial_att > top_5_percent_threshold)
        total_pixels = spatial_att.size
        focus_percentage = focused_region_size / total_pixels * 100
        
        print(f"\n{galaxy_type.title()} Galaxy Analysis:")
        print(f"  Attention peak at: {max_attention_pos} (center: {center})")
        print(f"  Attention center of mass: ({attention_mass_center[0]:.1f}, {attention_mass_center[1]:.1f})")
        print(f"  Top 5% attention covers: {focus_percentage:.1f}% of pixels")
        print(f"  Channel preferences:")
        
        channel_names = ['SobelX (vertical edges)', 'SobelY (horizontal edges)', 
                        'Diagonal1 (NE-SW edges)', 'Diagonal2 (NW-SE edges)']
        
        for i, (name, weight) in enumerate(zip(channel_names, channel_att)):
            print(f"    {weight:.3f} - {name}")
        
        # Determine what the network is focusing on
        most_important_channel = np.argmax(channel_att)
        if most_important_channel == 0:
            focus_type = "vertical structures (spiral arms, ellipse minor axis)"
        elif most_important_channel == 1:
            focus_type = "horizontal structures (spiral arms, ellipse major axis)"
        elif most_important_channel == 2:
            focus_type = "diagonal structures (spiral arm orientations)"
        else:
            focus_type = "opposite diagonal structures (complex galaxy features)"
        
        print(f"  Primary focus: {focus_type}")
        
        # Spatial focus analysis
        if np.linalg.norm(attention_mass_center - np.array(center)) < 3:
            spatial_focus = "central regions (galaxy core/bulge)"
        else:
            spatial_focus = "off-center regions (spiral arms/disk features)"
        
        print(f"  Spatial focus: {spatial_focus}")

def attention_for_shear_measurement():
    """
    Demonstrate how attention helps with galaxy shear measurement.
    """
    print("=== Attention for Galaxy Shear Measurement ===")
    
    # Create galaxies with different characteristics
    galaxies = {
        'round': create_synthetic_galaxy(size=48, spiral_arms=True),
        'elliptical': create_synthetic_galaxy(size=48, spiral_arms=False)  # Naturally more elliptical
    }
    
    attention_net = GalaxyAttentionNetwork()
    
    print("Benefits of attention for weak lensing shear measurement:")
    print("=" * 55)
    print("1. Focus on galaxy edges (most sensitive to shape distortion)")
    print("2. Ignore background noise and irrelevant PSF artifacts")
    print("3. Adaptively weight different galaxy regions by S/N ratio")
    print("4. Improve precision for subtle ellipticity measurements")
    print("5. Learn optimal feature combinations for g1/g2 estimation")
    
    shear_results = {}
    
    for name, galaxy in galaxies.items():
        result = attention_net.forward(galaxy)
        spatial_focus = np.std(result['spatial_attention'])
        
        # Simulate shear measurement quality
        edge_features = result['final_features']
        total_signal = np.sum(np.abs(edge_features))
        noise_level = np.std(galaxy[galaxy < np.mean(galaxy)])
        signal_to_noise = total_signal / (noise_level + 1e-8)
        
        shear_results[name] = {
            'spatial_focus': spatial_focus,
            'signal_to_noise': signal_to_noise,
            'result': result
        }
        
        print(f"\n{name.title()} Galaxy:")
        print(f"  Spatial focus strength: {spatial_focus:.3f}")
        print(f"  Effective S/N for shear: {signal_to_noise:.1f}")
    
    # Visualize shear measurement benefits
    plt.figure(figsize=(16, 8))
    
    for i, (name, data) in enumerate(shear_results.items()):
        result = data['result']
        galaxy = galaxies[name]
        
        # Plot original galaxy
        plt.subplot(2, 4, i*4 + 1)
        plt.imshow(galaxy, cmap='gray')
        plt.title(f'{name.title()} Galaxy')
        plt.axis('off')
        
        # Plot spatial attention overlaid
        plt.subplot(2, 4, i*4 + 2)
        plt.imshow(galaxy, cmap='gray', alpha=0.6)
        attention_overlay = plt.imshow(result['spatial_attention'], cmap='Reds', alpha=0.7)
        plt.title('Attention Map\n(Shear-sensitive regions)')
        plt.axis('off')
        plt.colorbar(attention_overlay, fraction=0.046)
        
        # Plot edge strength before attention
        plt.subplot(2, 4, i*4 + 3)
        edge_strength = np.sum(np.abs(result['original_features']), axis=-1)
        plt.imshow(edge_strength, cmap='hot')
        plt.title('Raw Edge Features')
        plt.axis('off')
        
        # Plot attended edge features (what actually gets used for shear)
        plt.subplot(2, 4, i*4 + 4)
        attended_edges = np.sum(np.abs(result['final_features']), axis=-1)
        plt.imshow(attended_edges, cmap='hot')
        plt.title('Attended Features\n(For Shear Measurement)')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nConnection to Real Shear Measurement Pipelines:")
    print("-" * 50)
    print("Traditional approach:")
    print("  • Fixed weighting schemes (Gaussian, optimal)")
    print("  • Manual feature engineering")
    print("  • Separate noise estimation step")
    
    print("\nAttention-enhanced approach:")
    print("  • Learned adaptive weighting")
    print("  • End-to-end feature learning")
    print("  • Implicit noise handling")
    
    print("\nPotential Improvements for ShearNet:")
    print("  1. Add spatial attention before final regression layers")
    print("  2. Use channel attention to weight different filter responses")
    print("  3. Train attention jointly with g1/g2 prediction")
    print("  4. Could improve systematic error control")
    
    print("\nImplementation Considerations:")
    print("  • Attention adds computational overhead (~20-30%)")
    print("  • Need careful regularization to avoid overfitting")
    print("  • Validate that attention focuses on physically meaningful regions")
    print("  • Test robustness across different galaxy morphologies")

def modern_attention_insights():
    """
    Connect spatial attention to modern Transformer attention.
    """
    print("=== Modern Attention Insights ===")
    print("Connecting classical CV attention to modern architectures")
    
    print("\nSpatial Attention (implemented in this module):")
    print("  Purpose: WHERE to look in an image")
    print("  Method: Learns spatial importance maps")
    print("  Best for: Localized features (galaxy spiral arms, edges)")
    print("  Complexity: O(HxW) attention weights")
    print("  Memory: Low - single attention map")
    
    print("\nChannel Attention (Squeeze-and-Excitation style):")
    print("  Purpose: WHICH features are important")
    print("  Method: Learns feature channel weights via global pooling")
    print("  Best for: Feature selection and weighting")
    print("  Complexity: O(C) attention weights")
    print("  Memory: Very low - one weight per channel")
    
    print("\nSelf-Attention (Transformer-style):")
    print("  Purpose: HOW different parts relate to each other")
    print("  Method: Learns pairwise relationships between all positions")
    print("  Best for: Long-range dependencies, global context")
    print("  Complexity: O(N²) where N = HxWxC")
    print("  Memory: High - quadratic in sequence length")
    
    # Compare computational complexity
    print("\nComputational Complexity Comparison:")
    print("  (Assuming 48x48 galaxy images with 4 feature channels)")
    
    H, W, C = 48, 48, 4
    spatial_ops = H * W
    channel_ops = C
    self_attn_ops = (H * W) ** 2
    
    print(f"  Spatial Attention:     {spatial_ops:,} operations")
    print(f"  Channel Attention:     {channel_ops:,} operations")
    print(f"  Self-Attention:        {self_attn_ops:,} operations")
    print(f"  Self-Attention is {self_attn_ops // spatial_ops}x more expensive!")
    
    print("\nApplication to Different Astronomical Tasks:")
    print("  Galaxy Classification:")
    print("    ✓ Spatial: Focus on spiral arms vs smooth regions")
    print("    ✓ Channel: Weight texture vs edge features")
    print("    ⚠ Self: Overkill, adds unnecessary complexity")
    
    print("\n  Weak Lensing Shear Measurement:")
    print("    ✓ Spatial: Focus on high S/N galaxy regions")
    print("    ✓ Channel: Weight different edge orientations")
    print("    ⚠ Self: Too expensive for precision requirements")
    
    print("\n  Multi-Galaxy Scene Analysis:")
    print("    ✓ Spatial: Segment individual galaxies")
    print("    ✓ Channel: Different features for different galaxy types")
    print("    ✓ Self: Model galaxy-galaxy interactions")
    
    print("\nEvolution to Vision Transformers (ViTs):")
    print("  • ViTs divide image into patches (e.g., 16x16 pixels)")
    print("  • Each patch becomes a 'token' in the sequence")
    print("  • Self-attention relates every patch to every other patch")
    print("  • Very powerful but computationally expensive")
    
    print("\n  For 48x48 galaxy images with 8x8 patches:")
    patches = (H // 8) * (W // 8)
    vit_ops = patches ** 2
    print(f"    Patches: {patches} ({H//8}x{W//8})")
    print(f"    ViT operations: {vit_ops:,}")
    print(f"    Much more manageable than pixel-level self-attention!")
    
    print("\nRecommendations for Galaxy Analysis:")
    print("  Start simple: Spatial + Channel attention")
    print("    • Lower computational cost")
    print("    • Easier to interpret and debug")
    print("    • Often sufficient for astronomy tasks")
    
    print("\n  Consider self-attention for:")
    print("    • Very large images (>256x256)")
    print("    • Complex multi-object scenes")
    print("    • When you have abundant computational resources")
    print("    • Tasks requiring global context understanding")
    
    print("\nFurther Reading:")
    print("  • 'Attention Is All You Need' (Vaswani et al.) - Original Transformer")
    print("  • 'An Image is Worth 16x16 Words' (Dosovitskiy et al.) - Vision Transformer")
    print("  • 'Squeeze-and-Excitation Networks' (Hu et al.) - Channel attention")
    print("  • 'CBAM: Convolutional Block Attention Module' - Combined spatial/channel")

if __name__ == "__main__":
    print("Day 6: Attention Mechanisms for Galaxy Analysis!")
    print("Learning to focus on what matters in astronomical images")
    
    # Core concept
    print("\n" + "="*60)
    demonstrate_attention_concept()
    
    # Apply to galaxies
    print("\n" + "="*60)
    test_attention_on_galaxies()
    
    # Research applications
    print("\n" + "="*60)
    attention_for_shear_measurement()
    
    # Modern connections
    print("\n" + "="*60)
    modern_attention_insights()