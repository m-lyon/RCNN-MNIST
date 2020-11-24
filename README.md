# RCNN MNIST Model
This project contains an implementation of a Recurrent Convolutional Neural Network used on MNIST digits data.

## Overview



## Motivation
The motivating problem for this architecture is as follows:

To provide a model that can take image based data, hand drawn digits for example, and iteratively encode information about the data in a 'state' tensor. The 'state' tensor then represents underlying structure present in the digits. This 'state' can then be paired with some output parameter vector, such as a Nx90 degrees rotation vector, and decoded into an augmented version of the inputs, such as a rotated digit.

For example, given 4 images of the digit '9' and a 180 degree rotation vector, the output would be a rotated 9 digit (which would look like a 6).

This general architecture allows the flexibility of learning and storing features about a given dataset in an iterative manner. This is useful for datasets of high dimensionality, such as 4D images, where loading training data may be prohibitive, as well as for data that has a varying number of inputs in one of the dimensions, such as a stream of related images.

The architecture for this network is inspired from [insert refs here].

## Example Usage
```python
import numpy as np
import matplotlib.pyplot as plt
from RCNN import RCNNModel, Tutorial

# Get a specified amount of digits from a subset of the handdrawn digit dataset
digits = Tutorial.get_digit_set(digit=4, num=5)

# Display one of the digits using matplotlib
plt.imshow(digits[0,0,:,:,0], cmap='gist_gray')

## Create rotation vector
# [[1., 0., 0., 0.,]] -> 0 degrees
# [[0., 1., 0., 0.,]] -> 90 degrees
# [[0., 0., 1., 0.,]] -> 180 degrees
# [[0., 0., 0., 1.,]] -> 270 degrees
rotation_vector = np.array([[0., 1., 0., 0.]])

# Load the model with pretrained weights
model = RCNNModel(load_weights=True)

# Encode the state tensor with your digits
model.encode(digits)

# Produce a rotated digit with the rotation vector
rotated_digit = model.decode(rotation_vector)

# Display the image
plt.imshow(rotated_digit[0,:,:,0], cmap='gist_gray')
```

## Implementation Details


