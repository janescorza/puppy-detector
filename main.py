import numpy as np
import matplotlib.pyplot as plt

from utils.neural_network import L_layer_model
from utils.train_set_processing import prepare_dataset


def main():

    # Get input and treat it correctly


    path_to_dog_train_set = "/Users/jan.escorza.fuertes.prv/Repos/puppy-detector/data/training_set/dogs"
    path_to_cat_train_set = "/Users/jan.escorza.fuertes.prv/Repos/puppy-detector/data/training_set/cats"

    print("Preparing dataset...")
    train_x, train_y = prepare_dataset(path_to_dog_train_set, path_to_cat_train_set)
    print("Images: ", train_x.shape, "Labels: ", train_y.shape)
    print("Dataset prepared")
    learning_rate = 0.0075

    # n_x Assuming the images are of the same size within the train set
    if len(train_x) > 0:
        n_x = train_x[0].shape[0]
    else:
        raise ValueError('The train set is empty. Please check your path to the train set and the files in the folder.')
    print("n_x:", n_x)
    # Set a single output node for the classifier
    # TODO review what is the required size for n_h_1
    n_h_1 = 10
    # n_h_1 = n_x
    n_h_2 = 20
    n_h_3 = 10
    n_y = 1 
    #set the dimensions of input layer, hidden layers (add L values) and output layer
    layers_dims = (n_x, n_h_1, n_h_2, n_h_3, n_y) 
    
    print("Model first run with 1 iteration....")
    # Initialize and run the model
    parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 1, print_cost = False)

    print("Cost after first iteration: " + str(costs[0]))

    print("Model second run with 2500 iterations....")
    # Train the model
    parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

    def plot_costs(costs, learning_rate=0.0075):
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    plot_costs(costs, learning_rate)

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