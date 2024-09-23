# Implementation-of-CNN

## AIM

To Develop a convolutional deep neural network for digit classification.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.(http://alexlenail.me/NN-SVG/index.html)

### image

![Screenshot 2024-09-23 111618](https://github.com/user-attachments/assets/9b8d1c38-b4ef-4b7a-a88e-ebc2b52a90b5)


## DESIGN STEPS

### Step 1:
Load the MNIST dataset, which contains images of handwritten digits and corresponding labels.

### Step 2:
Reshape and normalize the images by scaling the pixel values between 0 and 1.

### Step 3:
Define a CNN model using layers such as convolution, pooling, and fully connected layers. Add a softmax output layer for classification.

### Step 4:
Compile the model using an appropriate optimizer (e.g., Adam), loss function (e.g., categorical crossentropy), and evaluation metric (accuracy).

### Step 5:
Train the model on the training dataset for a fixed number of epochs (e.g., 10 epochs) while monitoring the validation accuracy using early stopping to halt training once a desired accuracy is achieved.

### Step 6:
Stop training if the model reaches 98% accuracy on the training set to prevent overfitting and save computation time.


## PROGRAM

### Name:Hemasonica P
### Register Number:212222230048

```
import os
import base64
import numpy as np
import tensorflow as tf

# Get current working directory
current_dir = os.getcwd()

# Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, "/content/mnist.npz.zip")

# Load data (discard test set)
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

print(f"training_images is of type {type(training_images)}.\ntraining_labels is of type {type(training_labels)}\n")

# Inspect shape of the data
data_shape = training_images.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")


def reshape_and_normalize(images):
    """Reshapes the array of images and normalizes pixel values.

    Args:
        images (numpy.ndarray): The images encoded as numpy arrays

    Returns:
        numpy.ndarray: The reshaped and normalized images.
    """

    ### START CODE HERE ###

    # Reshape the images to add an extra dimension (at the right-most side of the array)
    images = np.expand_dims(images, axis=-1)

    # Normalize pixel values
    images = images/255.0

    ### END CODE HERE ###

    return images

# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Apply your function
training_images = reshape_and_normalize(training_images)

print('Name: Hemasonica P            RegisterNumber: 212222230048        \n')
print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")

def convolutional_model():
    """Returns the compiled (but untrained) convolutional model.

    Returns:
        tf.keras.Model: The model which should implement convolutions.
    """

    ## START CODE HERE ###

    # Define the model
    model = tf.keras.models.Sequential([

    # Add convolutions and max pooling
    tf.keras.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Add the same layers as before
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

    ### END CODE HERE ###

    # Compile the model
    model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)
    return model

model.summary()
### Model compiling and Training
from tensorflow.keras.callbacks import EarlyStopping
model = convolutional_model()
training_history = model.fit(training_images, training_labels, epochs=10, callbacks=[EarlyStopping()])
```
## OUTPUT

### Reshape and Normalize output
![Screenshot 2024-09-16 115104](https://github.com/user-attachments/assets/b90893f1-448c-4f99-99b9-a762abb69239)



### Training the model output

![Screenshot 2024-09-16 115119](https://github.com/user-attachments/assets/03f254da-5345-4a95-8556-86a24d77c123)




## RESULT
Thus the program to create a Convolution Neural Network to classify images is successfully implemented.
