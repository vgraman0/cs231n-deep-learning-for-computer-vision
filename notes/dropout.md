## Dropout

This is a regularization layer. We add randomization in the training process that we take away in test time. This improves generalization.

Randomly set neurons to zero with probability $p$.

At test time, all the neurons are active, but **we scale the activations for each neuron** to keep the expected output at test time the same as the expected output at training time.