# CNN From Scratch - Day05 Experiment Output

Generated on: 2025-06-24 21:22:01

Output directory: `c:\Users\adamf\OneDrive\Documents\GitHub\learningCNNs\CNN_from_scratch\day05\out`

---


## Day 5: Feature Map Interpretation and CNN Vision

Experiment started at: 21:22:01


### === Testing CNN Feature Map Progression ===


--- With Pooling (Standard CNN) ---


### === CNN Forward Pass (Pooling: True) ===

Layer 1: Edge Detection

  horizontal_edges: (50, 50) -> (48, 48)

  vertical_edges: (50, 50) -> (48, 48)

  diagonal_edges: (50, 50) -> (48, 48)

  After pooling: (48, 48) -> (24, 24)

Layer 2: Texture Detection

  texture_detector: (24, 24) -> (22, 22)

  smooth_detector: (24, 24) -> (22, 22)

  center_surround: (24, 24) -> (22, 22)

  After pooling: (22, 22) -> (11, 11)

Layer 3: Complex Pattern Detection

  large_structure: (11, 11) -> (7, 7)

  radial_pattern: (11, 11) -> (7, 7)

![day05_cnn_feature_progression.png](plots\day05_cnn_feature_progression.png)


--- Without Pooling (Precision-Preserving) ---


### === CNN Forward Pass (Pooling: False) ===

Layer 1: Edge Detection

  horizontal_edges: (50, 50) -> (48, 48)

  vertical_edges: (50, 50) -> (48, 48)

  diagonal_edges: (50, 50) -> (48, 48)

Layer 2: Texture Detection

  texture_detector: (48, 48) -> (46, 46)

  smooth_detector: (48, 48) -> (46, 46)

  center_surround: (48, 48) -> (46, 46)

Layer 3: Complex Pattern Detection

  large_structure: (46, 46) -> (42, 42)

  radial_pattern: (46, 46) -> (42, 42)

![day05_cnn_feature_progression.png](plots\day05_cnn_feature_progression.png)


### 
=== Precision vs Robustness Analysis ===

With pooling final shape: (24, 24)

Without pooling final shape: (48, 48)

Spatial information loss with pooling: 75.0%


### === Dilated vs Pooled Receptive Fields ===

Pooled strategy: (50, 50) -> (12, 12) (spatial loss: 94.2%)

Dilated strategy: (50, 50) -> (42, 42) (spatial loss: 29.4%)

![day05_feature_responses.png](plots\day05_feature_responses.png)

![day05_receptive_field_comparison.png](plots\day05_receptive_field_comparison.png)


### === Feature Hierarchy Analysis ===


Analyzing spiral galaxy...


### === CNN Forward Pass (Pooling: True) ===

Layer 1: Edge Detection

  horizontal_edges: (48, 48) -> (46, 46)

  vertical_edges: (48, 48) -> (46, 46)

  diagonal_edges: (48, 48) -> (46, 46)

  After pooling: (46, 46) -> (23, 23)

Layer 2: Texture Detection

  texture_detector: (23, 23) -> (21, 21)

  smooth_detector: (23, 23) -> (21, 21)

  center_surround: (23, 23) -> (21, 21)

  After pooling: (21, 21) -> (10, 10)

Layer 3: Complex Pattern Detection

  large_structure: (10, 10) -> (6, 6)

  radial_pattern: (10, 10) -> (6, 6)

  Layer 1 - Sparsity: 0.82, Max response: 4.000

  Layer 2 - Sparsity: 0.30, Max response: 8.500


Analyzing elliptical galaxy...


### === CNN Forward Pass (Pooling: True) ===

Layer 1: Edge Detection

  horizontal_edges: (48, 48) -> (46, 46)

  vertical_edges: (48, 48) -> (46, 46)

  diagonal_edges: (48, 48) -> (46, 46)

  After pooling: (46, 46) -> (23, 23)

Layer 2: Texture Detection

  texture_detector: (23, 23) -> (21, 21)

  smooth_detector: (23, 23) -> (21, 21)

  center_surround: (23, 23) -> (21, 21)

  After pooling: (21, 21) -> (10, 10)

Layer 3: Complex Pattern Detection

  large_structure: (10, 10) -> (6, 6)

  radial_pattern: (10, 10) -> (6, 6)

  Layer 1 - Sparsity: 0.90, Max response: 2.000

  Layer 2 - Sparsity: 0.52, Max response: 4.500


Analyzing noisy galaxy...


### === CNN Forward Pass (Pooling: True) ===

Layer 1: Edge Detection

  horizontal_edges: (48, 48) -> (46, 46)

  vertical_edges: (48, 48) -> (46, 46)

  diagonal_edges: (48, 48) -> (46, 46)

  After pooling: (46, 46) -> (23, 23)

Layer 2: Texture Detection

  texture_detector: (23, 23) -> (21, 21)

  smooth_detector: (23, 23) -> (21, 21)

  center_surround: (23, 23) -> (21, 21)

  After pooling: (21, 21) -> (10, 10)

Layer 3: Complex Pattern Detection

  large_structure: (10, 10) -> (6, 6)

  radial_pattern: (10, 10) -> (6, 6)

  Layer 1 - Sparsity: 0.02, Max response: 3.959

  Layer 2 - Sparsity: 0.00, Max response: 6.871


### === Feature Evolution Visualization ===


### === CNN Forward Pass (Pooling: True) ===

Layer 1: Edge Detection

  horizontal_edges: (40, 40) -> (38, 38)

  vertical_edges: (40, 40) -> (38, 38)

  diagonal_edges: (40, 40) -> (38, 38)

  After pooling: (38, 38) -> (19, 19)

Layer 2: Texture Detection

  texture_detector: (19, 19) -> (17, 17)

  smooth_detector: (19, 19) -> (17, 17)

  center_surround: (19, 19) -> (17, 17)

  After pooling: (17, 17) -> (8, 8)

Layer 3: Complex Pattern Detection

  large_structure: (8, 8) -> (4, 4)

  radial_pattern: (8, 8) -> (4, 4)

![day05_feature_evolution_detailed.png](plots\day05_feature_evolution_detailed.png)

Day 5 experiments completed at: 21:22:07

============================================================

