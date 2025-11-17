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
