# Weight initialization

If we start with weights that are too small, all activations tend to zero for deeper network layers

If we start with weights that are too large, we have an exploding gradients problem.

## Kaiming Initialization

$$W = R(D_{in}, D_{out}) * \sqrt{2 / D_{in}}.$$