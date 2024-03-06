import numpy as np

from utils.neural_network import L_layer_model
from utils.prepare_dataset import prepare_dataset

def get_model_hyperparameters(X_shape):
    """
    Defines the hyperparameters for the neural network model.

    Returns:
        layers_dims: A tuple containing the dimensions of each layer in the network.
        learning_rate: The learning rate to use for training the network.
    """
    if X_shape and X_shape[1] > 0:
        n_x = X_shape[0]
    else:
        raise ValueError('The train set is empty. Please check your path to the train set and the files in the folder.')
    n_h_1 = 16
    n_h_2 = 12
    n_h_3 = 8
    n_y = 1  # Set a single output node for the classifier

    layers_dims = (n_x, n_h_1, n_h_2, n_h_3, n_y)
    learning_rate = 0.0075

    return layers_dims, learning_rate

def main():

    path_to_dog_train_set = "/Users/jan.escorza.fuertes.prv/Repos/puppy-detector/data/training_set/dogs"
    path_to_cat_train_set = "/Users/jan.escorza.fuertes.prv/Repos/puppy-detector/data/training_set/cats"

    print("Preparing dataset...")
    # train_x, train_y = prepare_dataset(path_to_dog_train_set, path_to_cat_train_set)
    train_x, train_y, = prepare_dataset(path_to_dog_train_set, path_to_cat_train_set)
    print("Dataset prepared")

    print("Prepare hyperparameters...")
    layers_dims, learning_rate = get_model_hyperparameters(train_x.shape)
    print("Hyperparameters prepared")

    print("Model first run with 1 epoch....")
    #TODO: Maybe avoid running the model with 1 singe epoch or if so print model 
    parameters = L_layer_model(train_x, train_y, layers_dims, num_epochs = 1, print_cost = False)
    #TODO: add decreasing gradient!

    print("Model second run with 3000 epochs....")
    # Train the model
    parameters = L_layer_model(train_x, train_y, layers_dims, num_epochs = 3000, print_cost = True)


    # Load and preprocess the image from the user

    # Feed the image through the network to get a prediction
    output = 1

    # Determine if the image is of a puppy
    if output > 0.5:
        print("The image is of a puppy.")
    else:
        print("The image is not of a puppy.")


if __name__ == "__main__":
    main()