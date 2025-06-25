# CNN From Scratch - Day02 Experiment Output

Generated on: 2025-06-24 20:33:04

Output directory: `c:\Users\adamf\OneDrive\Documents\GitHub\learningCNNs\CNN_from_scratch\day02\out`

---


## Day 2: Convolution and Edge Detection

Experiment started at: 20:33:04


### === Testing 1D Convolution ===

Signal: [1 2 3 4 5]

Kernel: [0.5 1.  0.5]

Result: [0.5 2.  4.  6.  8.  7.  2.5]

NumPy result: [0.5 2.  4.  6.  8.  7.  2.5]

Match: True


### === Testing 2D Convolution ===

Original image:

[[0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]]


Vertical edge detection result:

[[ 0.  3.  0. -3.  0.]
 [ 0.  3.  0. -3.  0.]
 [ 0.  3.  0. -3.  0.]
 [ 0.  3.  0. -3.  0.]
 [ 0.  3.  0. -3.  0.]]


Horizontal edge detection result:

[[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]


### === Edge Detection Demonstration ===

![day02_convolution_demo.png](plots\day02_convolution_demo.png)

![day02_edge_detection_demo.png](plots\day02_edge_detection_demo.png)

Edge detection completed!


### === Galaxy Edge Detection ===

![day02_galaxy_edge_detection.png](plots\day02_galaxy_edge_detection.png)

Galaxy shape: (50, 50)

Edge magnitude range: [0.000, 4.472]

Day 2 experiments completed at: 20:33:06

============================================================

