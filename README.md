# What is this?
This is an implementation of [Bundle Adjustment](https://en.wikipedia.org/wiki/Bundle_adjustment) using the Levenberg-Marquardt algorithm.

# What's Bundle Adjustment?
Imagine you have 2 pictures of the same scene, each with different viewpoints. Imagine we specify some landmarks in the scene (e.g. a tree). You want to figure out: 
1. The positions and poses of the cameras
2. The absolute positions of the landmarks

Bundle adjustment is a technique in which is able to solve for these two, by minimizing a loss function.

# Tricks
## Damping
To help solutions converge, we rely on the Levenberg-Marquadt algorithm with damping. Damping allows the algorithm to take small steps when near to a solution, and larger steps otherwise. 

## Sparsity
We also take advantage of sparsity to improve convergence - essentially, the matrix J^TJ using in Levenberg-Marquardt is highly sparse. We partition the matrix into regions A,B,C, such that A, B are block diagonal matrices, and C is a dense matrix. We implement versions of matrix inversion and linear matrix equation solving which take advantage of the block structure of A and B to reduce computation. 

## Gradient checking
The code takes advantage of hand-derived jacobians for various operations. To verify these hand-derived jacobians, we used PyTorch.

# How to run
1. First install numpy: `pip install numpy` 
2. Next, you may run bundle adjustment for a problem using: `python3 bundle_adjustment.py --problem <problemfile>`

The solutions will be output into the same working directory as the bundle adjustment script.

To evaluate the final loss for a solution, you may run:
`python3 eval_reconstruction.py --solution <solutionfile>`

# Caveats
## Numerical Precision
We do blocked matrix operations for efficiency. However, we do not implement any techniques to improve numerical precision of solutions. Since our matrices are large, this means that our computation is off from the ground truth by quite a bit, which may hurt convergence in some cases. 

