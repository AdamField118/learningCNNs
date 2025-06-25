# CNN From Scratch - Day04 Experiment Output

Generated on: 2025-06-24 21:21:29

Output directory: `c:\Users\adamf\OneDrive\Documents\GitHub\learningCNNs\CNN_from_scratch\day04\out`

---


## Day 4: Pooling Operations and Spatial Trade-offs

Experiment started at: 21:21:29


### === Basic Pooling Test ===

Original feature map:

[[1 3 2 4]
 [5 6 1 2]
 [3 2 4 1]
 [1 1 3 5]]


Max pooled (2x2):

[[6. 4.]
 [3. 5.]]


Average pooled (2x2):

[[3.75 2.25]
 [1.75 3.25]]


==================================================


### === Testing Pooling Effects on Galaxy Features ===

Edge features shape: (48, 48)

After 2x2 pooling: (48, 48) -> (24, 24)

Spatial information lost: 75.0%

Max pooling strongest response: 4.00

Average pooling strongest response: 3.75

Response preservation ratio: 1.07x

![day04_feature_responses.png](plots\day04_feature_responses.png)


### === Testing Different Pool Sizes ===

Pool size 2x2:

  Shape: (48, 48) -> (24, 24)

  Info loss: 75.0%

  Max response: 4.00

Pool size 3x3:

  Shape: (48, 48) -> (16, 16)

  Info loss: 88.9%

  Max response: 4.00

Pool size 4x4:

  Shape: (48, 48) -> (12, 12)

  Info loss: 93.8%

  Max response: 4.00

![day04_feature_responses.png](plots\day04_feature_responses.png)


### === Testing Translation Invariance ===

Edge features similarity (1-pixel shift): 0.872

Pooled features similarity (1-pixel shift): 0.903

Robustness improvement: 1.04x

![day04_feature_responses.png](plots\day04_feature_responses.png)


### === Adaptive Pooling Demonstration ===

Input 20x20 -> Features (18, 18) -> Pool 2x2 -> Output (9, 9)

Input 30x30 -> Features (28, 28) -> Pool 3x3 -> Output (9, 9)

Input 48x48 -> Features (46, 46) -> Pool 5x5 -> Output (9, 9)


Adaptive pooling enables CNNs to handle variable input sizes!


### === Pooling Strategy Comparison ===

![day04_pooling_strategy_comparison.png](plots\day04_pooling_strategy_comparison.png)

Pooling Strategy Comparison:

  Max 2x2: (15, 15), 4.0x reduction, max response: 4.000

  Avg 2x2: (15, 15), 4.0x reduction, max response: 4.000

  Max 3x3: (10, 10), 9.0x reduction, max response: 4.000

  Avg 3x3: (10, 10), 9.0x reduction, max response: 2.667

Day 4 experiments completed at: 21:21:37

============================================================

