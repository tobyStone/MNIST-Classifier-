# MNIST-Classifier-
Udacity Nanodegree in Deep Learning Project 1, with instructions to follow these steps:

Step 1

Load the dataset from torchvision.datasets
Use transforms or other PyTorch methods to convert the data to tensors, normalize, and flatten the data.
Create a DataLoader for your dataset

Step 2

Visualize the dataset using the provided function and either:
Your training data loader and inverting any normalization and flattening
A second DataLoader without any normalization or flattening
Explore the size and shape of the data to get a sense of what your inputs look like naturally and after transformation. Provide a brief justification of any necessary preprocessing steps or why no preprocessing is needed.

Step 3

Using PyTorch, build a neural network to predict the class of each given input image
Create an optimizer to update your network’s weights
Use the training DataLoader to train your neural network

Step 4

Evaluate your neural network’s accuracy on the test set.
Tune your model hyperparameters and network architecture to improve your test set accuracy, achieving at least 90% accuracy on the test set.

Step 5

Use torch.save to save your trained model.


An example of the digits the classifoer will try to class into 0-9 labels, using the programmed model:

![](https://github.com/tobyStone/MNIST-Classifier-/blob/main/MNISTexample.png)
