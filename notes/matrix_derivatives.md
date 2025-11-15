# Loss function derivatives
During the optimization stage, we take the gradient of our loss function with respect to a matrix. The notation around this is often very simplified in practice since we are dealing with tensors. 

Instead, we write out the derivatives explicitly. This makes the result much more clear (in my opinion).

## Single datapoint
Let's start with a single datapoint $x \in R^D$ (a row vector).
Our goal is to take the derivative of $F$ with respect to $W \in \mathbb{R}^{D \times C}$, where $F(W) = L(xW)$. 

Define a scoring function $s(W) = xW$. Then
$$s_i = \sum_{d=1}^D x_d W_{di}$$

Taking the derivative of $s_i$ with respect to $W_{jk}$:
$$\frac{\partial s_i}{\partial W_{jk}} = \sum_{d=1}^D x_d \frac{\partial W_{di}}{\partial W_{jk}} = x_j \delta_{ik}.$$

Notice that we only have non-zero entries when taking the derivative with respect to $W_{ji}$:
$$\frac{\partial s_i}{\partial W_{ji}} = x_j.$$

Using the chain rule,
$$
\begin{align*}
\frac{\partial F}{\partial W_{ij}} &= \sum_{c=1}^C \frac{\partial L}{\partial s_c}\frac{\partial s_c}{\partial W_{ij}} \\ &= \sum_{c=1}^C \frac{\partial L}{\partial s_c}x_i\delta_{cj} \\ &= \frac{\partial L}{\partial s_j}x_i \\ &= (x^\top (\nabla_s F))_{ij}
\end{align*}$$

## Multiple datapoints

The result extends naturally to multiple datapoints, $X \in \mathbb{R}^{N \times D}$, $W \in \mathbb{R}^{D \times C}$. Our new scoring function returns a matrix:
$$s_{ij} = \sum_{d=1}^D X_{id}W_{dj}.$$

Taking the derivative of $s_{ij}$ with respect to $W_{k\ell}$:
$$\frac{\partial s_{ij}}{\partial W_{k\ell}} = \sum_{d=1}^D X_{id}\delta_{dk}\delta_{j\ell} = X_{ik}\delta_{j\ell}.$$

The element-wise derivative of our loss function is
$$
\begin{align*}
\frac{\partial F}{\partial W_{ij}} &= \sum_{n=1}^N\sum_{c=1}^C \frac{\partial L}{\partial s_{nc}}\frac{\partial s_{nc}}{\partial W_{ij}} \\ &= \sum_{n=1}^N\sum_{c=1}^C \frac{\partial L}{\partial s_{nc}}X_{ni}\delta_{cj}\\  &= \sum_{n=1}^N \frac{\partial L}{\partial s_{nj}}X_{ni} \\ &= (X^\top\nabla_{s}L)_{ij}.
\end{align*}$$