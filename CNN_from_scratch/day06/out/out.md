# CNN From Scratch - Day06 Experiment Output

Generated on: 2025-06-24 21:42:59

Output directory: `c:\Users\adamf\OneDrive\Documents\GitHub\learningCNNs\CNN_from_scratch\day06\out`

---


## Day 6: Modern Activation Functions!

Experiment started at: 21:42:59

Exploring beyond ReLU for better galaxy analysis

=== Activation Function Properties Comparison ===

![day06_compare activation properties.png](plots\day06_compare activation properties.png)


Key Properties Analysis:

ReLU: Simple, fast, but can die (gradient=0 for x<0)

  - Zero gradient for negative inputs -> dying neurons

  - Not zero-centered -> inefficient learning

Swish: Smooth, self-gating, non-monotonic

  - Can decrease then increase (non-monotonic)

  - Smooth everywhere -> better gradients

  - Self-gating: uses own value to control activation

GELU: Smooth, probabilistic interpretation

  - Based on Gaussian CDF -> principled approach

  - Used in BERT, GPT -> proven in transformers

  - Smooth gradients help optimization

Mish: Smooth, non-monotonic, self-regularizing

  - Often empirically outperforms others

  - Strong negative information preservation

  - Self-regularizing properties

ELU: Smooth, negative saturation

  - Smooth everywhere unlike ReLU

  - Negative saturation prevents extreme negative values

  - Helps with vanishing gradients

Leaky ReLU: Simple fix for dying ReLU

  - Minimal computational overhead

  - Prevents dying neurons

  - Good baseline modern activation

Original edge features: min=-4.000, max=4.000

Range: 8.000



ReLU        : range=4.000, negative_info=0.0%, strong_resp=1.1%, zero_resp=91.9%, mean=0.121

Swish       : range=4.202, negative_info=9.6%, strong_resp=1.1%, zero_resp=82.2%, mean=0.083

GELU        : range=4.159, negative_info=9.6%, strong_resp=1.1%, zero_resp=82.2%, mean=0.102

Mish        : range=4.301, negative_info=9.6%, strong_resp=1.1%, zero_resp=82.2%, mean=0.091

ELU         : range=4.982, negative_info=9.6%, strong_resp=1.1%, zero_resp=82.2%, mean=0.062

Leaky ReLU  : range=4.040, negative_info=9.6%, strong_resp=1.1%, zero_resp=82.2%, mean=0.120

![day06_test activations on galaxy features.png](plots\day06_test activations on galaxy features.png)


Detailed Analysis for Galaxy Applications:

ReLU        : Feature diversity: 0.4% (9/2304 unique values)

Swish       : Feature diversity: 0.7% (17/2304 unique values)

GELU        : Feature diversity: 0.7% (16/2304 unique values)

Mish        : Feature diversity: 0.7% (17/2304 unique values)

ELU         : Feature diversity: 0.7% (17/2304 unique values)

Leaky ReLU  : Feature diversity: 0.7% (17/2304 unique values)


Implications for Galaxy Analysis:

- Higher negative info preservation -> Better capture of galaxy structure variations

- Lower zero responses -> More neurons stay active (avoid dying neurons)

- Smooth activations -> Better gradient flow for precise shear measurements

- Non-monotonic functions -> Can capture complex galaxy morphologies

=== Gradient Flow Analysis ===

Analyzing ReLU...

Analyzing Swish...

Analyzing GELU...

Analyzing Mish...

Analyzing ELU...

Analyzing Leaky ReLU...

![day06_exploding gradient rate.png](plots\day06_exploding gradient rate.png)


Gradient Preservation Results:

======================================================================

Activation   Mean Final   Stability    Vanishing%   Exploding%

======================================================================

ReLU         0.0000       0.0000       100.0        0.0         

Swish        0.0000       7.7699       100.0        0.0         

GELU         0.0000       0.0000       100.0        0.0         

Mish         0.0000       5.9959       100.0        0.0         

ELU          0.0079       2.6035       81.0         0.0         

Leaky ReLU   0.0000       6.9699       100.0        0.0         


Interpretation:

- Mean Final: Higher values indicate better gradient preservation

- Stability: Lower coefficient of variation indicates more consistent training

- Vanishing%: Lower percentage indicates fewer vanishing gradient problems

- Exploding%: Lower percentage indicates fewer exploding gradient problems

=== Activation Function Recommendations ===

 For Galaxy Shear Measurement (ShearNet-style tasks):

1. GELU or Mish - Smooth gradients help precision tasks

   -> Better convergence for sub-pixel accuracy requirements

2. Swish - Self-gating property good for subtle feature detection

   -> Can adaptively emphasize important features

3. Avoid standard ReLU - Dying ReLU hurts subtle feature detection

   -> Loss of gradient information is critical for precision tasks


 For Galaxy Classification (Morphology, redshift estimation):

1. Swish or GELU - Good balance of performance and stability

2. Mish - Often gives best empirical results

3. Leaky ReLU - If computational budget is very tight

   -> Minimal overhead compared to ReLU


 For Very Deep Networks (>50 layers, ResNet-style):

1. GELU or Mish - Best gradient flow properties

2. Combine with:

   - Residual connections (essential)

   - Batch normalization

   - Proper weight initialization

3. Avoid ELU in very deep networks - can cause instability


 Computational Considerations:

  ReLU        : 1.0x (baseline)

  Leaky ReLU  : 1.1x

  Swish       : 2.3x

  GELU        : 2.1x

  Mish        : 3.2x

  ELU         : 2.8x


 Specific Recommendations by Task:

  Weak Lensing Shear       : GELU > Mish

                             Why: Smooth gradients crucial for precision



  Galaxy-Galaxy Lensing    : Swish > GELU

                             Why: Good feature extraction, stable training



  Redshift Estimation      : Mish > Swish

                             Why: Complex non-linear relationships



  Star-Galaxy Separation   : Leaky ReLU > Swish

                             Why: Fast inference needed, clear boundaries



  Supernova Classification : GELU > Mish

                             Why: Temporal patterns, need smooth optimization



  Galaxy Simulation        : ELU > Swish

                             Why: Preserve negative information, physical constraints



Hyperparameter Tuning Tips:

- Swish: Try $eta$ $\in$ [0.5, 1.5] for the scaling parameter

- ELU: $lpha$ $\in$ [0.5, 2.0] depending on desired negative saturation

- Leaky ReLU: $lpha$ $\in$ [0.01, 0.1] for negative slope

- Always validate on held-out galaxy data with realistic noise


 Common Pitfalls:

- Don't change activation mid-training - causes instability

- Test with realistic galaxy noise levels - some activations more robust

- Consider activation choice early - affects all other hyperparameters

- Monitor gradient norms during training - early indicator of problems

Day 6 experiments completed at: 21:43:04

============================================================



---
Generated on: 2025-06-24 21:47:29

---


## Day 6: Attention Mechanisms for Galaxy Analysis!

Experiment started at: 21:47:29

Learning to focus on what matters in astronomical images


============================================================

![day06_demonstrate attention concept.png](plots\day06_demonstrate attention concept.png)

Uniform average: -0.018

Attention-weighted: -0.250

Difference: 0.233


Key Insight:

- Attention allows the network to focus on important positions!

- Instead of treating all inputs equally, it learns what matters

- The attention weights sum to 1.0, ensuring proper normalization

- Attention weights sum: 1.000000


Top 3 attended positions: [3 4 7]

  Position 3: weight=0.307, value=-0.351

  Position 4: weight=0.186, value=-0.781

  Position 7: weight=0.102, value=0.081


============================================================

=== Testing Attention on Galaxy Features ===


Analyzing spiral galaxy...

  Spatial attention range: [0.426, 0.702]

  Spatial attention focus: 0.038 (higher = more focused)

  Peak attention at: (np.int64(19), np.int64(7)), distance from center: 16.5 pixels

  Channel preferences:

    SobelX: 0.500

    SobelY: 0.500

    Diagonal1: 0.500

    Diagonal2: 0.500

  Dominant channel: SobelX (0.500)

  Attention efficiency: -9307.0% (0%=uniform, 100%=perfectly focused)


Analyzing elliptical galaxy...

  Spatial attention range: [0.460, 0.615]

  Spatial attention focus: 0.024 (higher = more focused)

  Peak attention at: (np.int64(8), np.int64(25)), distance from center: 15.1 pixels

  Channel preferences:

    SobelX: 0.500

    SobelY: 0.500

    Diagonal1: 0.500

    Diagonal2: 0.500

  Dominant channel: SobelX (0.500)

  Attention efficiency: -9401.0% (0%=uniform, 100%=perfectly focused)


Analyzing noisy galaxy...

  Spatial attention range: [0.441, 0.697]

  Spatial attention focus: 0.033 (higher = more focused)

  Peak attention at: (np.int64(19), np.int64(7)), distance from center: 16.5 pixels

  Channel preferences:

    SobelX: 0.500

    SobelY: 0.500

    Diagonal1: 0.500

    Diagonal2: 0.500

  Dominant channel: SobelX (0.500)

  Attention efficiency: -9252.9% (0%=uniform, 100%=perfectly focused)


=== Visualizing Galaxy Attention Maps ===

![day06_visualizing galaxy attention maps.png](plots\day06_visualizing galaxy attention maps.png)


Detailed Attention Analysis:

============================================================


Spiral Galaxy Analysis:

  Attention peak at: (np.int64(19), np.int64(7)) (center: (23, 23))

  Attention center of mass: (22.5, 22.5)

  Top 5% attention covers: 5.0% of pixels

  Channel preferences:

    0.500 - SobelX (vertical edges)

    0.500 - SobelY (horizontal edges)

    0.500 - Diagonal1 (NE-SW edges)

    0.500 - Diagonal2 (NW-SE edges)

  Primary focus: vertical structures (spiral arms, ellipse minor axis)

  Spatial focus: central regions (galaxy core/bulge)


Elliptical Galaxy Analysis:

  Attention peak at: (np.int64(8), np.int64(25)) (center: (23, 23))

  Attention center of mass: (22.5, 22.5)

  Top 5% attention covers: 5.0% of pixels

  Channel preferences:

    0.500 - SobelX (vertical edges)

    0.500 - SobelY (horizontal edges)

    0.500 - Diagonal1 (NE-SW edges)

    0.500 - Diagonal2 (NW-SE edges)

  Primary focus: vertical structures (spiral arms, ellipse minor axis)

  Spatial focus: central regions (galaxy core/bulge)


Noisy Galaxy Analysis:

  Attention peak at: (np.int64(19), np.int64(7)) (center: (23, 23))

  Attention center of mass: (22.5, 22.5)

  Top 5% attention covers: 5.0% of pixels

  Channel preferences:

    0.500 - SobelX (vertical edges)

    0.500 - SobelY (horizontal edges)

    0.500 - Diagonal1 (NE-SW edges)

    0.500 - Diagonal2 (NW-SE edges)

  Primary focus: vertical structures (spiral arms, ellipse minor axis)

  Spatial focus: central regions (galaxy core/bulge)


============================================================

=== Attention for Galaxy Shear Measurement ===

Benefits of attention for weak lensing shear measurement:

=======================================================

1. Focus on galaxy edges (most sensitive to shape distortion)

2. Ignore background noise and irrelevant PSF artifacts

3. Adaptively weight different galaxy regions by S/N ratio

4. Improve precision for subtle ellipticity measurements

5. Learn optimal feature combinations for g1/g2 estimation


Round Galaxy:

  Spatial focus strength: 0.050

  Effective S/N for shear: 45067206598.1


Elliptical Galaxy:

  Spatial focus strength: 0.032

  Effective S/N for shear: 21846552713.4

![day06_attention for shear measurement.png](plots\day06_attention for shear measurement.png)


Connection to Real Shear Measurement Pipelines:

--------------------------------------------------

Traditional approach:

- Fixed weighting schemes (Gaussian, optimal)

- Manual feature engineering

- Separate noise estimation step


Attention-enhanced approach:

- Learned adaptive weighting

- End-to-end feature learning

- Implicit noise handling


Potential Improvements for ShearNet:

  1. Add spatial attention before final regression layers

  2. Use channel attention to weight different filter responses

  3. Train attention jointly with g1/g2 prediction

  4. Could improve systematic error control


Implementation Considerations:

- Attention adds computational overhead (~20-30%)

- Need careful regularization to avoid overfitting

- Validate that attention focuses on physically meaningful regions

- Test robustness across different galaxy morphologies


============================================================

=== Modern Attention Insights ===

Connecting classical CV attention to modern architectures


Spatial Attention (implemented in this module):

  Purpose: WHERE to look in an image

  Method: Learns spatial importance maps

  Best for: Localized features (galaxy spiral arms, edges)

  Complexity: O(HxW) attention weights

  Memory: Low - single attention map


Channel Attention (Squeeze-and-Excitation style):

  Purpose: WHICH features are important

  Method: Learns feature channel weights via global pooling

  Best for: Feature selection and weighting

  Complexity: O(C) attention weights

  Memory: Very low - one weight per channel


Self-Attention (Transformer-style):

  Purpose: HOW different parts relate to each other

  Method: Learns pairwise relationships between all positions

  Best for: Long-range dependencies, global context

  Complexity: O(N²) where N = HxWxC

  Memory: High - quadratic in sequence length


Computational Complexity Comparison:

  (Assuming 48x48 galaxy images with 4 feature channels)

  Spatial Attention:     2,304 operations

  Channel Attention:     4 operations

  Self-Attention:        5,308,416 operations

  Self-Attention is 2304x more expensive!


Application to Different Astronomical Tasks:

  Galaxy Classification:

    Spatial: Focus on spiral arms vs smooth regions

    Channel: Weight texture vs edge features

    Self: Overkill, adds unnecessary complexity


  Weak Lensing Shear Measurement:

    Spatial: Focus on high S/N galaxy regions

    Channel: Weight different edge orientations

    Self: Too expensive for precision requirements


  Multi-Galaxy Scene Analysis:

    Spatial: Segment individual galaxies

    Channel: Different features for different galaxy types

    Self: Model galaxy-galaxy interactions


Evolution to Vision Transformers (ViTs):

 - ViTs divide image into patches (e.g., 16x16 pixels)

 - Each patch becomes a 'token' in the sequence

 - Self-attention relates every patch to every other patch

 - Very powerful but computationally expensive


  For 48x48 galaxy images with 8x8 patches:

    Patches: 36 (6x6)

    ViT operations: 1,296

    Much more manageable than pixel-level self-attention!


Recommendations for Galaxy Analysis:

  Start simple: Spatial + Channel attention

   - Lower computational cost

   - Easier to interpret and debug

   - Often sufficient for astronomy tasks


  Consider self-attention for:

   - Very large images (>256x256)

   - Complex multi-object scenes

   - When you have abundant computational resources

   - Tasks requiring global context understanding


Further Reading:

 - 'Attention Is All You Need' (Vaswani et al.) - Original Transformer

 - 'An Image is Worth 16x16 Words' (Dosovitskiy et al.) - Vision Transformer

 - 'Squeeze-and-Excitation Networks' (Hu et al.) - Channel attention

 - 'CBAM: Convolutional Block Attention Module' - Combined spatial/channel

Day 6 experiments completed at: 21:47:35

============================================================



---
Generated on: 2025-06-24 21:53:32

---


## Day 6: Batch Normalization

Experiment started at: 21:53:32

=== Internal Covariate Shift Problem ===

Early training - Layer 1: mean=-0.04, std=2.89

Later training - Layer 1: mean=1.88, std=2.89

Early training - Layer 2 (post-ReLU): mean=0.37, std=0.56

Later training - Layer 2 (post-ReLU): mean=2.35, std=2.24

![day06_demonstrate internal covariate shift.png](plots\day06_demonstrate internal covariate shift.png)


Key Problem: As training progresses, input distributions to each layer change,

making it harder for subsequent layers to learn effectively.

BatchNorm Solution: Keep input distributions normalized and stable.

=== Training Stability Comparison ===

![day06_compare training stability.png](plots\day06_compare training stability.png)

Without Batch Norm:

  - Unstable training with high variance

  - Requires careful learning rate tuning

  - Prone to gradient explosion/vanishing

  - Final loss: 0.905


With Batch Norm:

  - Stable, smooth convergence

  - Allows higher learning rates

  - Robust to hyperparameter choices

  - Final loss: 0.832

=== Batch Normalization for Galaxy Networks ===

Created batch of 8 galaxies

Galaxy brightness ranges:

  Bright Spiral: min=0.000, max=0.815, mean=0.163

  Bright Spiral: min=0.000, max=1.114, mean=0.222

  Bright Spiral: min=0.000, max=1.939, mean=0.387

  Faint Elliptical: min=0.000, max=0.256, mean=0.061

  Faint Elliptical: min=0.000, max=0.471, mean=0.110

  Faint Elliptical: min=0.000, max=0.326, mean=0.080

  Distant Noisy: min=0.000, max=0.465, mean=0.104

  Distant Noisy: min=0.000, max=0.234, mean=0.052


Before Batch Norm (after edge detection):

  Mean: -0.001

  Std: 0.739

  Min: -7.757

  Max: 7.757

  Range: 15.515


After Batch Norm (after edge detection):

  Mean: -0.000

  Std: 2.654

  Min: -11.585

  Max: 11.664

  Range: 23.249

![day06_batch norm for galaxy networks.png](plots\day06_batch norm for galaxy networks.png)


Benefits for galaxy networks:

- Stable training regardless of galaxy brightness variations

- Consistent feature extraction across different galaxy types

- Faster convergence for weak lensing shear measurement

- Less sensitive to initialization and hyperparameters

- Enables processing of mixed galaxy populations effectively


Demonstrating inference mode:

Running mean: [0.0019847  0.00211311 0.00250357 0.00165739 0.00126323]

Running var: [0.90014114 0.90018255 0.9001935  0.9000923  0.90005131]

Test galaxy normalized using running statistics: mean=0.216, std=0.298

Day 6 experiments completed at: 21:53:36

============================================================



---
Generated on: 2025-06-24 21:53:40

---


## Day 6: Residual Connections

Experiment started at: 21:53:40

=== Demonstrating Vanishing Gradient Problem ===

![day06_demonstrate vanishing gradients.png](plots\day06_demonstrate vanishing gradients.png)

Without residual connections:

Initial gradient: 1.0

Final gradient: 0.011529


With residual connections:

Final gradient: 1.000000

Improvement factor: 86.74x

=== Residual vs Plain Network Comparison ===

Plain Network:

  Plain Layer 1: (48, 48), max=4.000

  Plain Layer 2: (46, 46), max=8.000

  Plain Layer 3: (44, 44), max=7.812

  Plain Layer 4: (42, 42), max=3.031

  Plain Layer 5: (40, 40), max=0.586


Residual Network:

  Residual Network:

  Residual Block 1: (46, 46), max=1.000

  Residual Block 2: (42, 42), max=1.000

  Residual Block 3: (38, 38), max=1.000

![day06_feature_responses.png](plots\day06_feature_responses.png)


Plain network final response: 0.586

Residual network final response: 1.000

Residual network maintains stronger signal through deeper layers!

=== Residual Connections for Galaxy Analysis ===

![day06_feature_responses.png](plots\day06_feature_responses.png)

Key insight: Galaxy shear measurement needs BOTH:

1. Low-level edge information (precise shape)

2. High-level structural information (galaxy type)

Residual connections preserve both!


Signal preservation analysis:

Layer 1 (edges): 4.000

Layer 2 without residual: 4.000 (100.0% preserved)

Layer 2 with residual: 7.314 (182.9% preserved)

=== Why ResNet Changed Everything ===

Before ResNet (2015):

- Networks deeper than ~20 layers performed WORSE

- Not due to overfitting - even training error increased!

- Vanishing gradients made deep networks untrainable


ResNet's insight:

- Let layers learn residual function F(x) = H(x) - x

- Identity mapping is easier to learn than complex mapping

- Skip connections provide gradient superhighway


Impact:

- Enabled 50, 101, even 1000+ layer networks

- Dramatic improvements in image recognition

- Foundation for modern architectures (Transformers use similar ideas)


Mathematical insight:

Traditional layer: H(x) = F(weight*x + bias)

Residual layer: H(x) = F(weight*x + bias) + x

If optimal mapping is close to identity, F can learn small corrections

=== Visualizing Residual Information Flow ===

![day06_feature_responses.png](plots\day06_feature_responses.png)

Notice how residual connections maintain signal strength!

Day 6 experiments completed at: 21:53:47

============================================================

