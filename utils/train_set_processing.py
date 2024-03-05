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
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    print("X_shuffled", X_shuffled.shape)
    Y_shuffled = Y[indices]
    print("Y_shuffled", Y_shuffled.shape)
    print("Dataset shuffled")
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
    # Process dog images
    dog_images = process_images_in_folder(dog_folder_path)
    print("dog images shape: ", dog_images.shape)
    num_dogs = len(dog_images)
    print("dog images number: ", num_dogs)
    dog_labels = np.ones((num_dogs, 1))
    print("dog labels shape:", dog_labels.shape)

    print("Processing cat images...")
    # Process cat images
    cat_images = process_images_in_folder(cat_folder_path)
    num_cats = len(cat_images)
    cat_labels = np.zeros((num_cats, 1))
    print("cat labels shape:", cat_labels.shape)

    # Combine images and labels
    # The reshape assumes that both dog_images and cat_images have the same shape
    X = np.concatenate((dog_images, cat_images), axis=0)
    print("X shape:", X.shape)
    Y = np.concatenate((dog_labels, cat_labels), axis=0)
    print("Y shape:", Y.shape)

    
    X_shuffled, Y_shuffled = shuffle_dataset(X,Y)
    return X_shuffled, Y_shuffled

    # Shuffle the images and labels
    np.random.seed(0) # Setting the seed for reproducibility
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation]
    print("X permutation shape:", X.shape)
    Y = Y[permutation]
    print("Y permutation shape:", Y.shape)
    return X, Y