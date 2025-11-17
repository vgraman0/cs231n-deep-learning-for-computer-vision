## K-fold Cross Validation

![image.png](img/image%205.png)

Useful for small datasets, but not practical for deep learning scales.

## How does this perform?

Quite poorly on pixel data. These are high dimensional objects (many pixels) and distances over high dimensional spaces are not intuitive.

In practice:

- Preprocess the data: normalize the features to have zero mean and unit variance.
- Consider PCA/NCA/Random Projections to reduce the dimensionality.