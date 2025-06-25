import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Tuple, Optional, Union

# Add parent directory for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import enhanced output system
from utils.output_system import log_print, log_experiment_start, log_experiment_end, save_plot
from day02.edge_detection_basics import manual_convolution_2d
from day03.convolution_mechanics import manual_convolution_2d_extended
from utils.test_data import create_synthetic_galaxy
from kernels.edge_detection_kernels import sobel_x_kernel

def max_pooling_2d(
    feature_map: np.ndarray, 
    pool_size: int = 2, 
    stride: Optional[int] = None
) -> np.ndarray:
    """
    Implement max pooling using convolution windowing logic.
    
    Args:
        feature_map: 2D numpy array (output from convolution)
        pool_size: Size of pooling window (default 2x2)
        stride: Stride for pooling (default: same as pool_size)
    
    Returns:
        Pooled feature map (smaller spatial dimensions)
    """
    if stride is None:
        stride = pool_size
    
    # Get dimensions for output calculation (same as convolution)
    input_h, input_w = feature_map.shape
    output_h = (input_h - pool_size) // stride + 1
    output_w = (input_w - pool_size) // stride + 1
    output = np.zeros((output_h, output_w))
    
    # Use the same windowing logic as convolution, but apply max instead
    for i in range(output_h):
        for j in range(output_w):
            start_i = i * stride
            start_j = j * stride
            
            # Extract the window (this is what convolution iterates over)
            window = feature_map[start_i:start_i + pool_size, 
                               start_j:start_j + pool_size]
            
            # Apply max operation instead of convolution
            output[i, j] = np.max(window)
    
    return output

def average_pooling_2d(
    feature_map: np.ndarray, 
    pool_size: int = 2, 
    stride: Optional[int] = None
) -> np.ndarray:
    """
    Implement average pooling using convolution - this one's even cleaner!
    
    Average pooling is literally just convolution with a kernel of all 1s,
    followed by division by kernel size.
    
    Args:
        feature_map: 2D numpy array (output from convolution)
        pool_size: Size of pooling window (default 2x2)
        stride: Stride for pooling (default: same as pool_size)
    
    Returns:
        Average pooled feature map
    """
    if stride is None:
        stride = pool_size
    
    # Create kernel of all ones
    ones_kernel = np.ones((pool_size, pool_size))
    
    # Use existing convolution function!
    conv_result = manual_convolution_2d_extended(
        feature_map, 
        ones_kernel, 
        stride=stride, 
        padding=0
    )
    
    # Divide by kernel size to get average
    return conv_result / (pool_size * pool_size)

def test_pooling_on_galaxy_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply pooling to galaxy edge detection to see precision vs robustness trade-off.
    
    This demonstrates how pooling affects astronomical feature detection.
    
    Returns:
        Tuple of (galaxy, edge_features, max_pooled, avg_pooled)
    """
    log_print("=== Testing Pooling Effects on Galaxy Features ===", level="SUBHEADER")
    
    # Use centralized galaxy generator
    galaxy = create_synthetic_galaxy(size=50, spiral_arms=True)
    
    # Apply edge detection (Day 2/3 work)
    sobel_x = sobel_x_kernel()
    
    # Get edge features
    edge_features = manual_convolution_2d(galaxy, sobel_x)
    log_print(f"Edge features shape: {edge_features.shape}")
    
    # Apply both types of pooling
    max_pooled = max_pooling_2d(edge_features, pool_size=2)
    avg_pooled = average_pooling_2d(edge_features, pool_size=2)
    
    log_print(f"After 2x2 pooling: {edge_features.shape} -> {max_pooled.shape}")
    log_print(f"Spatial information lost: {((edge_features.size - max_pooled.size) / edge_features.size * 100):.1f}%")
    
    # Analyze the differences
    max_response = np.max(np.abs(max_pooled))
    avg_response = np.max(np.abs(avg_pooled))
    
    log_print(f"Max pooling strongest response: {max_response:.2f}")
    log_print(f"Average pooling strongest response: {avg_response:.2f}")
    log_print(f"Response preservation ratio: {max_response/avg_response:.2f}x")
    
    # Visualize the results
    from utils.visualization import plot_feature_responses
    plot_feature_responses(galaxy, {
        'Edge Features': edge_features,
        'Max Pooled (2x2)': max_pooled,
        'Average Pooled (2x2)': avg_pooled
    }, title="Pooling Effects on Galaxy Feature Detection")
    
    return galaxy, edge_features, max_pooled, avg_pooled

def test_pooling_size_effects() -> Dict[str, np.ndarray]:
    """
    Test how different pool sizes affect galaxy feature detection.
    
    Returns:
        Dictionary mapping pool descriptions to results
    """
    log_print("=== Testing Different Pool Sizes ===", level="SUBHEADER")
    
    galaxy = create_synthetic_galaxy(size=50, spiral_arms=True)
    sobel_x = sobel_x_kernel()
    edge_features = manual_convolution_2d(galaxy, sobel_x)
    
    pool_results = {'Original': edge_features}
    
    # Test different pool sizes
    for pool_size in [2, 3, 4]:
        max_result = max_pooling_2d(edge_features, pool_size=pool_size)
        
        # Calculate information loss
        loss_percent = ((edge_features.size - max_result.size) / edge_features.size * 100)
        
        log_print(f"Pool size {pool_size}x{pool_size}:")
        log_print(f"  Shape: {edge_features.shape} -> {max_result.shape}")
        log_print(f"  Info loss: {loss_percent:.1f}%")
        log_print(f"  Max response: {np.max(np.abs(max_result)):.2f}")
        
        pool_results[f'Max Pool {pool_size}x{pool_size}'] = max_result
    
    # Visualize all results
    from utils.visualization import plot_feature_responses
    plot_feature_responses(galaxy, pool_results, 
                         title="Pool Size Effects on Galaxy Detection")
    
    return pool_results

def test_translation_invariance() -> Tuple[float, float]:
    """
    Test how pooling makes features robust to small shifts.
    
    Returns:
        Tuple of (edge_similarity, pooled_similarity)
    """
    log_print("=== Testing Translation Invariance ===", level="SUBHEADER")
    
    # Create galaxy and shift it by 1 pixel
    galaxy_original = create_synthetic_galaxy(size=50, spiral_arms=True)
    galaxy_shifted = np.roll(galaxy_original, shift=1, axis=0)  # Shift down by 1 pixel
    
    sobel_x = sobel_x_kernel()
    
    # Get edge features for both
    edges_original = manual_convolution_2d(galaxy_original, sobel_x)
    edges_shifted = manual_convolution_2d(galaxy_shifted, sobel_x)
    
    # Apply pooling to both
    pooled_original = max_pooling_2d(edges_original, pool_size=2)
    pooled_shifted = max_pooling_2d(edges_shifted, pool_size=2)
    
    # Measure similarity
    def feature_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
        """Calculate normalized correlation between features."""
        f1_flat = f1.flatten()
        f2_flat = f2.flatten()
        correlation = np.corrcoef(f1_flat, f2_flat)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    # Compare similarities
    edge_similarity = feature_similarity(edges_original, edges_shifted)
    pooled_similarity = feature_similarity(pooled_original, pooled_shifted)
    
    log_print(f"Edge features similarity (1-pixel shift): {edge_similarity:.3f}")
    log_print(f"Pooled features similarity (1-pixel shift): {pooled_similarity:.3f}")
    log_print(f"Robustness improvement: {pooled_similarity/edge_similarity:.2f}x")
    
    # Visualize the trade-off
    from utils.visualization import plot_feature_responses
    plot_feature_responses(galaxy_original, {
        'Original Edges': edges_original,
        'Shifted Edges': edges_shifted, 
        'Pooled Original': pooled_original,
        'Pooled Shifted': pooled_shifted
    }, title="Translation Invariance: Precision vs Robustness")
    
    return edge_similarity, pooled_similarity

def demonstrate_adaptive_pooling() -> None:
    """Show how adaptive pooling works - fixed output size regardless of input."""
    log_print("=== Adaptive Pooling Demonstration ===", level="SUBHEADER")
    
    # Test with different input sizes
    sizes = [20, 30, 48]
    target_output = (8, 8)  # Fixed output size
    
    for size in sizes:
        galaxy = create_synthetic_galaxy(size=size, spiral_arms=True)
        sobel_x = sobel_x_kernel()
        features = manual_convolution_2d(galaxy, sobel_x)
        
        # Calculate required pool size for target output
        pool_h = features.shape[0] // target_output[0]
        pool_w = features.shape[1] // target_output[1]
        
        adaptive_result = max_pooling_2d(features, pool_size=pool_h)
        
        log_print(f"Input {size}x{size} -> Features {features.shape} -> Pool {pool_h}x{pool_h} -> Output {adaptive_result.shape}")
    
    log_print("\nAdaptive pooling enables CNNs to handle variable input sizes!")

def compare_pooling_strategies() -> Dict[str, Dict[str, Union[np.ndarray, float]]]:
    """
    Compare different pooling strategies on the same features.
    
    Returns:
        Dictionary with comparison results
    """
    log_print("=== Pooling Strategy Comparison ===", level="SUBHEADER")
    
    # Create test feature map
    galaxy = create_synthetic_galaxy(size=32, spiral_arms=True)
    sobel_x = sobel_x_kernel()
    features = manual_convolution_2d(galaxy, sobel_x)
    
    # Apply different pooling strategies
    max_2x2 = max_pooling_2d(features, pool_size=2)
    avg_2x2 = average_pooling_2d(features, pool_size=2)
    max_3x3 = max_pooling_2d(features, pool_size=3)
    avg_3x3 = average_pooling_2d(features, pool_size=3)
    
    strategies = {
        'Max 2x2': {'result': max_2x2, 'max_response': np.max(np.abs(max_2x2))},
        'Avg 2x2': {'result': avg_2x2, 'max_response': np.max(np.abs(avg_2x2))},
        'Max 3x3': {'result': max_3x3, 'max_response': np.max(np.abs(max_3x3))},
        'Avg 3x3': {'result': avg_3x3, 'max_response': np.max(np.abs(avg_3x3))}
    }
    
    # Create comparison visualization
    plt.figure(figsize=(15, 10))
    
    # Original features
    plt.subplot(2, 3, 1)
    plt.imshow(features, cmap='gray')
    plt.title(f'Original Features\n{features.shape}')
    plt.colorbar()
    plt.axis('off')
    
    # Pooled results
    for i, (name, data) in enumerate(strategies.items(), 2):
        plt.subplot(2, 3, i)
        plt.imshow(data['result'], cmap='gray')
        plt.title(f'{name}\n{data["result"].shape}\nMax: {data["max_response"]:.2f}')
        plt.colorbar()
        plt.axis('off')
    
    plt.tight_layout()
    save_plot("pooling_strategy_comparison.png")
    
    # Log comparison
    log_print("Pooling Strategy Comparison:")
    for name, data in strategies.items():
        shape = data['result'].shape
        reduction = features.size / data['result'].size
        log_print(f"  {name}: {shape}, {reduction:.1f}x reduction, max response: {data['max_response']:.3f}")
    
    return strategies

def main() -> None:
    """Main execution function for Day 4 experiments."""
    log_experiment_start(4, "Pooling Operations and Spatial Trade-offs")
    
    # Test with a simple feature map first
    test_feature_map = np.array([[1, 3, 2, 4],
                                [5, 6, 1, 2], 
                                [3, 2, 4, 1],
                                [1, 1, 3, 5]])
    
    log_print("=== Basic Pooling Test ===", level="SUBHEADER")
    log_print("Original feature map:")
    log_print(str(test_feature_map))
    
    # Test implementations
    max_result = max_pooling_2d(test_feature_map, pool_size=2)
    avg_result = average_pooling_2d(test_feature_map, pool_size=2)
    
    log_print("\nMax pooled (2x2):")
    log_print(str(max_result))
    log_print("\nAverage pooled (2x2):")
    log_print(str(avg_result))

    log_print("\n" + "="*50)
    
    # Galaxy pooling tests
    test_pooling_on_galaxy_features()
    test_pooling_size_effects()
    test_translation_invariance()
    demonstrate_adaptive_pooling()
    compare_pooling_strategies()
    
    log_experiment_end(4)

if __name__ == "__main__":
    main()