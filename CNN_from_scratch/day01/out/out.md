# CNN From Scratch - Day01 Experiment Output

Generated on: 2025-06-24 20:22:09

Output directory: `c:\Users\adamf\OneDrive\Documents\GitHub\learningCNNs\CNN_from_scratch\day01\out`

---


## Day 1: Neural Network Foundations

Experiment started at: 20:22:09


### === Testing Basic Operations ===

Inputs: [0.5 0.3]

Weights: [[0.4 0.7]]

Bias: [0.1]

Linear output (Wx + b): [0.51]

After sigmoid: [0.62480647]

After ReLU: [0.51]


Testing with negative input: [-0.5]

Sigmoid(-0.5): [0.37754067]

ReLU(-0.5): [0.]


### === Comparing Activation Functions ===

```
Sigmoid output stats: shape=(100,), min=0.007, max=0.993, mean=0.500, std=0.390
```

```
ReLU output stats: shape=(100,), min=0.000, max=5.000, mean=1.263, std=1.630
```

![day01_activation_comparison.png](plots\day01_activation_comparison.png)


### === Testing Perceptron Decision Making ===

Testing Perceptron Decision Making:

Weights: [[1 1]]

Bias: [-0.5]


Point		Sigmoid		ReLU

----------------------------------------

[0 0]		0.378		0.000

[0 1]		0.622		0.500

[1 0]		0.622		0.500

[1 1]		0.818		1.500


### === Parameter Explosion Demonstration ===

Parameter count for different approaches:

28x28 image -> 100 hidden neurons:

Parameters needed: 78,400


200x200 galaxy image -> 100 hidden neurons:

Parameters needed: 4,000,000


Why this is a problem:

- Each neuron needs to learn about EVERY pixel

- No sharing of knowledge between similar patterns

- Massive memory requirements

- Easy to overfit


### === Learning Through Backpropagation ===

Target output: 1.0

Input: [1. 1.]

Initial weights: [[ 0.24835708 -0.06913215]]

Initial bias: [0.32384427]



Epoch  0: Output=0.6232, Error=0.3768, Loss=0.0710

Epoch  2: Output=0.6354, Error=0.3646, Loss=0.0665

Epoch  4: Output=0.6469, Error=0.3531, Loss=0.0623

Epoch  6: Output=0.6578, Error=0.3422, Loss=0.0586

Epoch  8: Output=0.6680, Error=0.3320, Loss=0.0551


Final weights: [[0.3282899  0.01080067]]

Final bias: [0.40377709]

Final output: 0.6729 (target: 1.0)

Day 1 experiments completed at: 20:22:11

============================================================

