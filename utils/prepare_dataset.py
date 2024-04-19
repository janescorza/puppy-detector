import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from utils.image_preprocessing import preprocess_images_in_folder

def save_data(data, filename):
    np.save(filename, data)

def load_data(filename):
    return np.load(filename + '.npy')

def load_parameters(filename):
    """
    Load neural network parameters from a file.

    Args:
    filename (str): The path to the file from which to load the parameters.

    Returns:
    dict: The parameters of the neural network.
    """
    return np.load(filename, allow_pickle=True).item()

def shuffle_dataset(X, Y):
    """
    Shuffle the dataset using sklearn's shuffle to ensure X and Y remain in sync.

    Arguments:
    X -- a numpy array of shape (length*height*depth, number of images) containing the normalized image vectors
    Y -- a numpy array of shape (1, number of images) containing the labels (1 for dog, 0 for cat)

    Returns:
    X_shuffled -- shuffled X array
    Y_shuffled -- shuffled Y array
    """
    X, Y = shuffle(X.T, Y.T, random_state=42)  # Shuffle along the first axis
    X_shuffled = X.T
    Y_shuffled = Y.T

    print("X shape after shuffle:", X_shuffled.shape)
    print("Y shape after shuffle:", Y_shuffled.shape)

    return X_shuffled, Y_shuffled

def load_small_dataset():
    train_dataset = h5py.File('data/training_set/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # train set labels

    test_dataset = h5py.File('data/test_set/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # test set labels
    
    classes = np.array(test_dataset["list_classes"][:])

    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def prepare_small_dataset():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_small_dataset()
    show_sample = input("Would you like to see an example of a picture in the dataset? (y/n)")
    if show_sample.lower() == 'y':
        index = 50
        plt.imshow(train_x_orig[index])
        plt.show()
        print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
        
    m_train = train_x_orig.shape[0]
    print ("Number of training examples: " + str(m_train))
    print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    # Reshape to flatten dimensions
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    print ("train_x's shape: " + str(train_x.shape))
    return train_x, train_y, test_x, test_y, classes


def prepare_dataset(dog_folder_path, cat_folder_path, output_folder, force_preprocess=False):
    """
    Prepare the dataset by processing all dog and cat images, creating labels,
    and combining them into a single array of normalized vectors and labels.
    
    Arguments:
    dog_folder_path -- path to the folder containing the dog images
    cat_folder_path -- path to the folder containing the cat images
    output_folder -- path to the folder that should contain the preprocessed data
    force_preprocess -- boolean flag to force the preprocessing of the data even if it is already saved
    
    Returns:
    X -- a numpy array of shape (length*height*depth, number of image) containing the normalized image vectors
    Y -- a numpy array of shape (1, number of images) containing the labels (1 for dog, 0 for cat)
    """

    # Evaluate if we have preprocessed data
    x_path = os.path.join(output_folder, 'X')
    y_path = os.path.join(output_folder, 'Y')

    # Check if data is already processed and saved
    if not force_preprocess and os.path.exists(x_path + '.npy') and os.path.exists(y_path + '.npy'):
        X = load_data(x_path)
        Y = load_data(y_path)
    else:
        # Preprocess images for dogs (label 1) and cats (label 0)
        print("Processing dog images...")
        dog_images = preprocess_images_in_folder(dog_folder_path)
        # dog_images, dog_mean, dog_std  = preprocess_images_in_folder(dog_folder_path)
        print("dog images shape: ", dog_images.shape)
        dog_labels = np.ones(dog_images.shape[1])
        print("Processing cat images...")
        cat_images = preprocess_images_in_folder(cat_folder_path)
        # cat_images,  cat_mean, cat_std = preprocess_images_in_folder(cat_folder_path)
        print("cat images shape: ", cat_images.shape)
        cat_labels = np.zeros(cat_images.shape[1])

        X = np.hstack([dog_images, cat_images])
        print("Combined X shape:", X.shape)

        # Set labels for dogs and cats images in X
        Y = np.hstack([dog_labels, cat_labels]).reshape(1, -1)
        print("Combined Y shape:", Y.shape)

        print("Shuffle dataset...")
        X, Y = shuffle_dataset(X,Y)
        # X_shuffled, Y_shuffled = shuffle(X, Y, random_state=42)
        print("Dataset shuffled")

        # Save processed data
        save_data(X, x_path)
        save_data(Y, y_path)
    

    print("Images: ", X.shape, "Labels: ", Y.shape)
    return X, Y
