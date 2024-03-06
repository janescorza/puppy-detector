import math
import os
import numpy as np

from utils.image_preprocessing import process_images_in_folder

def shuffle_dataset(X, Y):
    """
    Shuffle the dataset incrementally.
    
    Arguments:
    X -- a numpy array of shape (number of images, length*height*depth) containing the normalized image vectors
    Y -- a numpy array of shape (number of images, 1) containing the labels (1 for dog, 0 for cat)
    
    Returns:
    X_shuffled -- shuffled X array
    Y_shuffled -- shuffled Y array
    """
    m = X.shape[1] # number of training examples
    permutation = list(np.random.permutation(m))
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation].reshape((1, m))
    print("X_shuffled", X_shuffled.shape)
    print("Y_shuffled", Y_shuffled.shape)
    return X_shuffled, Y_shuffled


def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []

    inc = mini_batch_size

    # Step 1 - Partition (shuffled_X, shuffled_Y).
    # Cases with a complete mini batch size only i.e each of 64 examples.
    num_complete_minibatches = math.floor(m / mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[:,k*inc:(k+1)*inc]
        mini_batch_Y = Y[:,k*inc:(k+1)*inc]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
    if m % mini_batch_size != 0:

        mini_batch_X = X[:,(k+1)*inc:((k+1)*inc)+(m % mini_batch_size)]
        mini_batch_Y = Y[:,(k+1)*inc:((k+1)*inc)+(m % mini_batch_size)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def prepare_dataset(dog_folder_path, cat_folder_path):
    """
    Prepare the dataset by processing all dog and cat images, creating labels,
    and combining them into a single array of normalized vectors and labels.
    
    Arguments:
    dog_folder_path -- path to the folder containing the dog images
    cat_folder_path -- path to the folder containing the cat images
    
    Returns:
    X -- a numpy array of shape (number of images, length*height*depth) containing the normalized image vectors
    Y -- a numpy array of shape (number of images, 1) containing the labels (1 for dog, 0 for cat)
    """
    print("Processing dog images...")
    dog_images = process_images_in_folder(dog_folder_path)
    print("dog images shape: ", dog_images.shape)
    num_dogs = dog_images.shape[1]
    print("dog images number: ", num_dogs)
    dog_labels = np.ones((1, num_dogs))
    print("dog labels shape:", dog_labels.shape)

    print("Processing cat images...")
    cat_images = process_images_in_folder(cat_folder_path)
    print("cat images shape: ", cat_images.shape)
    num_cats = cat_images.shape[1]
    print("cat images number: ", num_cats)
    cat_labels = np.zeros((1, num_cats))
    print("cat labels shape:", cat_labels.shape)

    # Combine images and labels
    # The reshape assumes that both dog_images and cat_images have the same shape
    X = np.concatenate((dog_images, cat_images), axis=1)
    print("Combined X shape:", X.shape)
    Y = np.concatenate((dog_labels, cat_labels), axis=1)
    print("Combined Y shape:", Y.shape)
    
    X_shuffled, Y_shuffled = shuffle_dataset(X,Y)

    mini_batches = random_mini_batches(X_shuffled, Y_shuffled)
    print("Mini batches: ", len(mini_batches))
    print("Each mini batch shape: ", mini_batches[0][0].shape)

    print("Images: ", X_shuffled.shape, "Labels: ", Y_shuffled.shape)
    return X_shuffled, Y_shuffled