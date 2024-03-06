import os
import numpy as np

from utils.image_preprocessing import preprocess_images_in_folder

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
    dog_images = preprocess_images_in_folder(dog_folder_path)
    print("dog images shape: ", dog_images.shape)
    num_dogs = dog_images.shape[1]
    print("dog images number: ", num_dogs)
    dog_labels = np.ones((1, num_dogs))
    print("dog labels shape:", dog_labels.shape)

    print("Processing cat images...")
    cat_images = preprocess_images_in_folder(cat_folder_path)
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
    

    print("Images: ", X.shape, "Labels: ", Y.shape)
    return X, Y
    # print("Shuffle dataset...")
    # X_shuffled, Y_shuffled = shuffle_dataset(X,Y)
    # print("Dataset shuffled")

    print("Images: ", X_shuffled.shape, "Labels: ", Y_shuffled.shape)
    return X_shuffled, Y_shuffled

    # Previous implementation return with minibatches in the prep
    # number_of_examples = X_shuffled.shape[1]
    # print(f"Number of examples: {number_of_examples}")
    # print("Prepare minibatches...")
    # mini_batches = random_mini_batches(X_shuffled, Y_shuffled)
    # print("Mini batches: ", len(mini_batches))
    # print("Each mini batch shape: ", mini_batches[0][0].shape)
    # return mini_batches, number_of_examples
