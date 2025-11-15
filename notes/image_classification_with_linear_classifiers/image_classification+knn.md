# 1.1 Image classification + KNN

Notes: https://cs231n.github.io/classification/

Video: https://youtu.be/pdqofxJeBN8?si=mIdurz4BgXJqsbbw

**Image classification:** Given a set of labels and an image, label the image.

![image.png](img/image.png)

**Pipeline:** 

1. Input: N images w/ K possible classes *training set*
2. Learning: Use the training set to learn what each class looks like *train a classifier, learn a model*
3. Evaluation: Check the quality of the classifier by asking it to predict labels for a new set of images.

## K-Nearest Neighbors

Define a distance function between images $d(I_1, I_2)$.  Given an image I where we want to predict the label, compute the distance $d(I, I_t)$ for all $I_t$ in the training data, and return the label of the image with the minimal distance.

- Training will take constant time: we just load the data into memory.
- Prediction takes $O(N)$ operations.

This is bad - it doesn’t scale when the size of our training set grows.

![image.png](img/image%201.png)

We can generalize this to **K-nearest neighbors**. Take the K closest points and take the majority label.

![image.png](img/image%202.png)

However, we are going to have regions where no decision can be made. This is a good way to find regions where we need more data.

### Choosing a distance function

![image.png](img/image%203.png)

- Notice that L1 distance is not rotation invariant, while L2 distance is!
- L1 is a better choice is we want to be more sensitive to feature values.

![image.png](img/image%204.png)

[http://vision.stanford.edu/teaching/cs231n-demos/knn/](http://vision.stanford.edu/teaching/cs231n-demos/knn/)

## K-fold Cross Validation

![image.png](img/image%205.png)

Useful for small datasets, but not practical for deep learning scales.

## How does this perform?

Quite poorly on pixel data. These are high dimensional objects (many pixels) and distances over high dimensional spaces are not intuitive.

In practice:

- Preprocess the data: normalize the features to have zero mean and unit variance.
- Consider PCA/NCA/Random Projections to reduce the dimensionality.