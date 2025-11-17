# Linear Classifier

This is the most important building block for all of deep learning.

## Parametric approach

We have an image $x \in \R^{32 \times 32 \times 3}$ and a function $$f(x, W) = Wx + b\in \R^{10}$$
where each output entry is the score in each class.

We want to find the best $W \in \R$ that minimizes the errors for $f$.

$b$ is a bias vector. This is an input independent value that helps with class separation.

## Classification Optimization Problem

Pick a loss function $\mathcal L$ and come up with an efficient way to find the parameters that minimize the loss function.

Given a dataset of $\{(x_i, y_i)\}_{i=1}^N$, our loss function is defined as the average error:
$$\mathcal L(W) = \frac{1}{N} \sum_i \mathcal \ell_i(f(x_i, W), y_i).$$

## Softmax classifier
Interpret the raw classifier scores as probabilities. We start with our linear output $\mathbf O = f(x, W)$ and then apply the softmax function to enter probability space:
$$\hat{\mathbf{Y_i}} = \phi_i(\mathbf O) = \frac{\exp(\mathbf O_i)}{\sum_j \exp(\mathbf O_j)}.$$

Now, define the labels $\mathbf Y$ as a one-hot encoding vector. 

Now, we take
$$ 
\begin{align*}
\mathcal L(W) &= - \log P(Y | X; W) \\
&= -\sum_{i=1}^N \log P(y_i \vert x_i ; W)\\  
&= - \sum_{i=1}^N \sum_{k=1}^K Y_{ik} \log \hat{\mathbf Y}_{ik} \\
&:= \sum_{i=1}^N \ell (Y_i, \hat{Y}_i)
\end{align*}
$$

In the case of softmax, 
$$\ell(\mathbf y, \hat{\mathbf y}) = \log \sum_{k=1}^{K} \exp(o_k) - \sum_{k=1}^{K} y_ko_k.$$
This has a very nice derivative:
$$\partial_{o_j} \ell(y, \hat y) = \phi_j(o) - y_j.$$
This is exactly the different between the inferred probability and the actualy probability.

### Cross-Entropy Loss
The loss function is known as the **cross-entropy loss**:$$\ell (\mathbf y, \hat {\mathbf y}) = - \sum_{k=1}^K y_j \log \hat y_j.$$

Cross-entropy comes from the information theoretic quantity: $H(P, Q) = \sum_{j} -P(j) \log Q(j)$. This corresponds to the expected surprisal of an observer with subjective probabilities Q seeing data that was generated according to P. 

So we can interpret the loss in two ways:
- Maximizing the likelihood of the observed data
- Minimizing the surprisal required to communicate labels

