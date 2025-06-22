"""
Feature map interpretation and CNN vision analysis.

This module explores what different layers of a CNN detect when processing
galaxy images, revealing the hierarchical feature learning process.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Centralized imports
from utils.test_data import create_synthetic_galaxy
from kernels.edge_detection_kernels import sobel_x_kernel, sobel_y_kernel
from day02.edge_detection_basics import manual_convolution_2d
from day04.pooling_operations import max_pooling_2d

class SimpleCNN:
    """
    A simple CNN pipeline for feature map analysis.
    
    This mimics the structure of networks like ShearNet to understand
    what features are learned at each layer.
    """
    
    def __init__(self):
        """Initialize a 3-layer CNN for galaxy analysis."""
        self.layers = []
        self.feature_maps = {}
        
        # Layer 1: Edge detection (low-level features)
        self.layer1_kernels = {
            'horizontal_edges': sobel_x_kernel(),
            'vertical_edges': sobel_y_kernel(),
            'diagonal_edges': np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], dtype=np.float32)
        }
        
        # Layer 2: Texture detection (mid-level features)
        self.layer2_kernels = {
            'texture_detector': np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32),
            'smooth_detector': np.ones((3, 3)) / 9,
            'center_surround': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        }
        
        # Layer 3: Complex patterns (high-level features)
        self.layer3_kernels = {
            'large_structure': np.array([
                [1, 1, 1, 1, 1],
                [1, -1, -1, -1, 1], 
                [1, -1, 8, -1, 1],
                [1, -1, -1, -1, 1],
                [1, 1, 1, 1, 1]
            ], dtype=np.float32) / 16,
            'radial_pattern': self._create_radial_kernel(5)
        }
    
    def _create_radial_kernel(self, size):
        """Create a radial detection kernel."""
        kernel = np.zeros((size, size))
        center = size // 2
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                kernel[i, j] = np.exp(-dist**2 / (2 * (size/4)**2))
        
        # Make it a center-surround detector
        kernel = kernel - np.mean(kernel)
        return kernel
    
    def forward_pass(self, galaxy_image, use_pooling=True):
        """
        Perform forward pass through the CNN, capturing all feature maps.
        
        Args:
            galaxy_image: Input galaxy image
            use_pooling: Whether to apply pooling (to test precision impact)
        
        Returns:
            Dictionary of feature maps from each layer
        """
        print(f"=== CNN Forward Pass (Pooling: {use_pooling}) ===")
        
        current_input = galaxy_image
        self.feature_maps = {'input': galaxy_image}
        
        # Layer 1: Edge Detection
        print("Layer 1: Edge Detection")
        layer1_outputs = {}
        for name, kernel in self.layer1_kernels.items():
            feature_map = manual_convolution_2d(current_input, kernel)
            layer1_outputs[name] = feature_map
            print(f"  {name}: {current_input.shape} -> {feature_map.shape}")
        
        self.feature_maps['layer1'] = layer1_outputs
        
        # Combine layer 1 outputs (simple max across channels)
        layer1_combined = np.maximum.reduce(list(layer1_outputs.values()))
        
        # Apply pooling if requested
        if use_pooling:
            layer1_pooled = max_pooling_2d(layer1_combined, pool_size=2)
            current_input = layer1_pooled
            self.feature_maps['layer1_pooled'] = layer1_pooled
            print(f"  After pooling: {layer1_combined.shape} -> {layer1_pooled.shape}")
        else:
            current_input = layer1_combined
        
        # Layer 2: Texture Detection
        print("Layer 2: Texture Detection")
        layer2_outputs = {}
        for name, kernel in self.layer2_kernels.items():
            feature_map = manual_convolution_2d(current_input, kernel)
            layer2_outputs[name] = feature_map
            print(f"  {name}: {current_input.shape} -> {feature_map.shape}")
        
        self.feature_maps['layer2'] = layer2_outputs
        
        # Combine layer 2 outputs
        layer2_combined = np.maximum.reduce(list(layer2_outputs.values()))
        
        # Apply pooling if requested
        if use_pooling:
            layer2_pooled = max_pooling_2d(layer2_combined, pool_size=2)
            current_input = layer2_pooled
            self.feature_maps['layer2_pooled'] = layer2_pooled
            print(f"  After pooling: {layer2_combined.shape} -> {layer2_pooled.shape}")
        else:
            current_input = layer2_combined
            
        # Layer 3: Complex Patterns  
        print("Layer 3: Complex Pattern Detection")
        layer3_outputs = {}
        for name, kernel in self.layer3_kernels.items():
            # Handle different kernel sizes
            try:
                feature_map = manual_convolution_2d(current_input, kernel)
                layer3_outputs[name] = feature_map
                print(f"  {name}: {current_input.shape} -> {feature_map.shape}")
            except Exception as e:
                print(f"  {name}: Skipped (size mismatch)")
        
        self.feature_maps['layer3'] = layer3_outputs
        
        return self.feature_maps
    
    def visualize_layer_progression(self, layer_name=None):
        """Visualize how features evolve through the network."""
        if layer_name is None:
            # Show progression through all layers
            self._visualize_full_progression()
        else:
            # Show specific layer in detail
            self._visualize_specific_layer(layer_name)
    
    def _visualize_full_progression(self):
        """Show the progression from input to final features."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Input
        axes[0, 0].imshow(self.feature_maps['input'], cmap='hot')
        axes[0, 0].set_title('Input Galaxy')
        axes[0, 0].axis('off')
        
        # Layer 1 (show combined)
        if 'layer1' in self.feature_maps:
            layer1_combined = np.maximum.reduce(list(self.feature_maps['layer1'].values()))
            axes[0, 1].imshow(layer1_combined, cmap='gray')
            axes[0, 1].set_title(f'Layer 1: Edges\n{layer1_combined.shape}')
            axes[0, 1].axis('off')
        
        # Layer 1 pooled
        if 'layer1_pooled' in self.feature_maps:
            axes[1, 1].imshow(self.feature_maps['layer1_pooled'], cmap='gray')
            axes[1, 1].set_title(f'Layer 1 Pooled\n{self.feature_maps["layer1_pooled"].shape}')
            axes[1, 1].axis('off')
        
        # Layer 2
        if 'layer2' in self.feature_maps:
            layer2_combined = np.maximum.reduce(list(self.feature_maps['layer2'].values()))
            axes[0, 2].imshow(layer2_combined, cmap='gray')
            axes[0, 2].set_title(f'Layer 2: Textures\n{layer2_combined.shape}')
            axes[0, 2].axis('off')
        
        # Layer 2 pooled
        if 'layer2_pooled' in self.feature_maps:
            axes[1, 2].imshow(self.feature_maps['layer2_pooled'], cmap='gray')
            axes[1, 2].set_title(f'Layer 2 Pooled\n{self.feature_maps["layer2_pooled"].shape}')
            axes[1, 2].axis('off')
        
        # Layer 3
        if 'layer3' in self.feature_maps and self.feature_maps['layer3']:
            layer3_combined = np.maximum.reduce(list(self.feature_maps['layer3'].values()))
            axes[0, 3].imshow(layer3_combined, cmap='gray')
            axes[0, 3].set_title(f'Layer 3: Patterns\n{layer3_combined.shape}')
            axes[0, 3].axis('off')
        
        # Hide unused plots
        for i in range(2):
            for j in range(4):
                if axes[i, j].get_title() == '':
                    axes[i, j].set_visible(False)
        
        plt.suptitle('CNN Feature Map Progression: Galaxy → Abstract Features', fontsize=16)
        plt.tight_layout()
        plt.show()

def test_cnn_feature_progression():
    """Test how features evolve through a simple CNN."""
    print("=== Testing CNN Feature Map Progression ===")
    
    # Create test galaxy
    galaxy = create_synthetic_galaxy(size=50, spiral_arms=True, add_noise=False)
    
    # Initialize CNN
    cnn = SimpleCNN()
    
    # Test with pooling
    print("\n--- With Pooling (Standard CNN) ---")
    feature_maps_pooled = cnn.forward_pass(galaxy, use_pooling=True)
    cnn.visualize_layer_progression()
    
    # Test without pooling
    print("\n--- Without Pooling (Precision-Preserving) ---")
    feature_maps_no_pool = cnn.forward_pass(galaxy, use_pooling=False)
    cnn.visualize_layer_progression()
    
    # Compare final representations
    print("\n=== Precision vs Robustness Analysis ===")
    
    if 'layer1_pooled' in feature_maps_pooled:
        pooled_final = feature_maps_pooled['layer1_pooled']
        no_pool_final = feature_maps_no_pool['layer1']
        no_pool_combined = np.maximum.reduce(list(no_pool_final.values()))
        
        print(f"With pooling final shape: {pooled_final.shape}")
        print(f"Without pooling final shape: {no_pool_combined.shape}")
        
        spatial_loss = (1 - pooled_final.size / no_pool_combined.size) * 100
        print(f"Spatial information loss with pooling: {spatial_loss:.1f}%")

def test_dilated_vs_pooled_receptive_fields():
    """Compare receptive field strategies: pooling vs dilation."""
    
    galaxy = create_synthetic_galaxy(size=50, spiral_arms=True)
    
    # Strategy 1: Pooling for large receptive field (what ShearNet does)
    edge_features = manual_convolution_2d(galaxy, sobel_x_kernel())
    pooled_result = max_pooling_2d(edge_features, pool_size=4)  # 48x48 -> 12x12
    
    # Strategy 2: Dilation for large receptive field (precision-preserving)
    from day03.convolution_mechanics import manual_convolution_2d_extended
    large_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    dilated_result = manual_convolution_2d_extended(galaxy, large_kernel, dilation=4)
    
    print(f"Pooled strategy: {galaxy.shape} -> {pooled_result.shape} (spatial loss: {((galaxy.size - pooled_result.size)/galaxy.size*100):.1f}%)")
    print(f"Dilated strategy: {galaxy.shape} -> {dilated_result.shape} (spatial loss: {((galaxy.size - dilated_result.size)/galaxy.size*100):.1f}%)")
    
    # Visualize the trade-off
    from utils.visualization import plot_feature_responses
    plot_feature_responses(galaxy, {
        'Pooled (Large Receptive Field)': pooled_result,
        'Dilated (Large Receptive Field)': dilated_result
    }, title="Receptive Field Strategies: Pooling vs Dilation")

if __name__ == "__main__":
    test_cnn_feature_progression()

    test_dilated_vs_pooled_receptive_fields()