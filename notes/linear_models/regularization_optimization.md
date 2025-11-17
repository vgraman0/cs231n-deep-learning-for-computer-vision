# Regularization 

For classification, we defined a loss function $$\mathcal L(W) = \sum_{i=1}^N \ell(f(x_i, W), y_i)$$
where we took the cross-entropy loss function.

This is called the **data loss**, measuring how well the predictions match the training data. However, we want our data to generalize to other examples.

$$L(W) = \sum_{i=1}^N \ell(f(x_i, W), y_i) + \lambda R(W)$$
![alt text](<img\classification loss pipeline.png>)
We can add a penalty that will come at a cost of performing worse on training data but have better generalization. This helps us avoid fitting noise in the data.

## Regularization strategies
- $L^2$ regularization: $R(W) = \|W\|_2^2$
- $L^1$ regularization: $R(W) = \|W\|_1$
- Elastic net: $R(W) = \beta\|W\|_2^2 + \|W\|_1$

There are more complex strategies as well:
- Dropout
- Batch normalization
- Stochastic depth, frational pooling

 