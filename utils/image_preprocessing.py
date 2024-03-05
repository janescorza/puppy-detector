from PIL import Image
import numpy as np
import os

def load_image(image_path, size=(64, 64)):
    """
    Load an image from a file path and convert it to a numpy array.
    
    Arguments:
    image_path -- path to the image file
    
    Returns:
    image_array -- numpy array representation of the image
    """
    image = Image.open(image_path)
    image = image.resize(size) # TODO consider effect of resizing
    image_array = np.array(image)
    return image_array

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2]),1)
    return v

def normalize_image(image_vector):
    """
    Normalize an image vector.
    
    Arguments:
    image_vector -- a vector of shape (length*height*depth, 1)
    
    Returns:
    normalized_vector -- a normalized vector of shape (length*height*depth, 1)
    """
    normalized_vector = image_vector / 255
    return normalized_vector

def process_images_in_folder(folder_path):
    """
    Process all JPG images in a folder.
    
    Arguments:
    folder_path -- path to the folder containing the images
    
    Returns:
    normalized_images -- a NumPy array of shape (length*height*depth, number of images) containing normalized vectors
    """
    normalized_vectors = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image_array = load_image(image_path)
            image_vector = image2vector(image_array)
            normalized_vector = normalize_image(image_vector)
            normalized_vectors.append(normalized_vector)
    # Stack vectors horizontaly to get an array with shape (length*height*depth, number of images) 
    normalized_images = np.hstack(normalized_vectors)
    return normalized_images
