# CNN From Scratch - Day03 Experiment Output

Generated on: 2025-06-24 21:19:10

Output directory: `c:\Users\adamf\OneDrive\Documents\GitHub\learningCNNs\CNN_from_scratch\day03\out`

---


## Day 3: Deep Dive into Kernels and Convolution

Experiment started at: 21:19:10


### === Stride Effects Demonstration ===

Stride 1: Input (20, 20) -> Output (18, 18)

Stride 2: Input (20, 20) -> Output (9, 9)

Stride 3: Input (20, 20) -> Output (6, 6)

![day03_convolution_mechanics_stride_effects.png](plots\day03_convolution_mechanics_stride_effects.png)


### === Padding Strategies Demonstration ===

Padding 0: Input (10, 10) -> Output (8, 8)

Padding 1: Input (10, 10) -> Output (10, 10)

Padding 2: Input (10, 10) -> Output (12, 12)

![day03_convolution_mechanics_padding_effects.png](plots\day03_convolution_mechanics_padding_effects.png)


### === Dilation Effects Demonstration ===

Dilation 1: Kernel (3, 3) -> Effective (3, 3)

Receptive field: (3, 3)

Dilation 2: Kernel (3, 3) -> Effective (5, 5)

Receptive field: (5, 5)

Dilation 3: Kernel (3, 3) -> Effective (7, 7)

Receptive field: (7, 7)

![day03_convolution_mechanics_dilation_effects.png](plots\day03_convolution_mechanics_dilation_effects.png)


### === Output Dimension Analysis ===


### === Convolution Output Dimensions ===

```
Input Size | Kernel | Stride | Padding | Output Size
```

```
-------------------------------------------------------
```

```
       28 |      3 |      1 |       0 |          26
```

```
       28 |      3 |      1 |       1 |          28
```

```
       32 |      5 |      2 |       2 |          16
```

```
      100 |      7 |      3 |       3 |          34
```


### === Computational Cost Analysis ===


### === Computational Cost Comparison ===

```
Configuration | Operations | Output Shape | Relative Cost
```

```
------------------------------------------------------------
```

```
Base (3x3, stride=1) |    451,584 | (224, 224)   |    1.00x
```

```
Larger kernel (5x5)  |  1,254,400 | (224, 224)   |    2.78x
```

```
Stride=2             |    112,896 | (112, 112)   |    0.25x
```

```
No padding           |    443,556 | (222, 222)   |    0.98x
```


### === Kernel Comparison Test ===

![day03_feature_responses.png](plots\day03_feature_responses.png)

All convolution mechanics tests completed!

Day 3 experiments completed at: 21:19:16

============================================================



---
Generated on: 2025-06-24 21:19:20

---


## Day 3: kernel designs in galaxy images

Experiment started at: 21:19:20

![day03_feature_responses.png](plots\day03_feature_responses.png)

Day 3 experiments completed at: 21:19:23

============================================================

