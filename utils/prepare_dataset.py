import os
import numpy as np
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

def prepare_dataset(base_path, dog_folder_relative_path, cat_folder_relative_path, output_folder_relative_path, force_preprocess=False):
    """
    Prepare the dataset by processing all dog and cat images, creating labels,
    and combining them into a single array of normalized vectors and labels.
    
    Arguments:
    base_path -- base path of the main script
    dog_folder_relative_path -- relative path to the folder containing the dog images
    cat_folder_relative_path -- relative path to the folder containing the cat images
    output_folder_relative_path -- relative path to the folder that should contain the preprocessed data
    force_preprocess -- boolean flag to force the preprocessing of the data even if it is already saved
    
    Returns:
    X -- a numpy array of shape (length*height*depth, number of image) containing the normalized image vectors
    Y -- a numpy array of shape (1, number of images) containing the labels (1 for dog, 0 for cat)
    """

    dog_folder_path = os.path.join(base_path, dog_folder_relative_path)  
    cat_folder_path = os.path.join(base_path, cat_folder_relative_path)  
    output_folder = os.path.join(base_path, output_folder_relative_path)  

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
