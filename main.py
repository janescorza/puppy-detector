import os
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.image_preprocessing import preprocess_image
from utils.neural_network import L_layer_model, L_model_forward
from utils.prepare_dataset import load_parameters, prepare_dataset, prepare_small_dataset, save_data

np.random.seed(2) # keep consistency in random calls

def initialize_paths(base_path):
    """
    Constructs and returns a dictionary of key paths used throughout the script.

    Arguments:
    base_path -- String, the base directory of the script (typically where the script is running from).

    Returns:
    Dictionary with keys pointing to specific directory paths needed for training, development, and parameter storage.
    """
    paths = {
        "train_dogs": "data/training_set/dogs",
        "train_cats": "data/training_set/cats",
        "train_output": "data/training_set/",
        "dev_dogs": "data/dev_set/dogs",
        "dev_cats": "data/dev_set/cats",
        "dev_output": "data/dev_set/",
        "params_output": "model/params"
    }
    for key, path in paths.items():
        paths[key] = os.path.join(base_path, path)
    return paths

def prepare_model_hyperparameters(X_shape, small_model=False):
    """
    Prepare and tune the hyperparameters for the neural network model based on the input shape.

    Arguments:
    X_shape -- Tuple containing the shape of the input dataset.

    Returns:
    Tuple of layer dimensions, learning rate, mini-batch size, and number of epochs.

    """
    if not X_shape or X_shape[1] == 0:
        raise ValueError('The training set is empty. Please check your path to the training set and the files in the folder.')
    if small_model:
        n_x = X_shape[0]
        n_h_1 = 16
        n_h_2 = 8
        n_h_3 = 4
        n_y = 1  # Set a single output node for the classifier
        learning_rate = 0.0008
        mini_batch_size = 32
        num_epochs = 200
    else:
        n_x = X_shape[0]
        n_h_1 = 2048
        n_h_2 = 1024
        n_h_3 = 256
        n_y = 1
        learning_rate = 0.001
        mini_batch_size = 128
        num_epochs = 50
    
    layers_dims = (n_x, n_h_1, n_h_2, n_h_3, n_y)

    return layers_dims, learning_rate, mini_batch_size, num_epochs

def show_costs_plot(costs, learning_rate):
    """
    Plots the cost function over the training epochs.

    Arguments:
    costs -- List, containing the cost values for each epoch.
    learning_rate -- Float, the learning rate used for training.
    """
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 1)')
    plt.title(f"Learning rate = {learning_rate}")
    plt.show()

def train_model(paths):
    """
    Conducts the model training process using the dataset and saves the trained parameters.

    Arguments:
    paths -- Dictionary, containing paths to the training data and where to save the parameters.

    Returns:
    parameters -- Dictionary containing the trained model parameters.
    """
    start_load = time.time()
    print("Preparing training dataset...")
    train_x, train_y = prepare_dataset(paths["train_dogs"], paths["train_cats"], paths["train_output"])
    load_time = time.time() - start_load
    print(f"Training dataset prepared in {load_time:.2f} seconds")

    print("Prepare hyperparameters...")
    layers_dims, learning_rate, mini_batch_size, num_epochs = prepare_model_hyperparameters(train_x.shape)
    print("Hyperparameters prepared")

    print("Training the model for several epochs...")
    parameters, costs = L_layer_model(train_x, train_y, layers_dims, learning_rate, mini_batch_size, num_epochs = num_epochs, print_cost=True)
    
    save_data(parameters, paths["params_output"])
    
    training_time = time.time() - start_load
    print(f"Training completed and parameters saved in {training_time:.2f} seconds")
    
    show_costs_plot(costs, learning_rate)

    return parameters

def train_small_model(train_x, train_y):
    """
    Conducts the model training process using the dataset and saves the trained parameters.

    Arguments:
    train_x -- numpy array, training input data
    train_y -- numpy array, training labels

    Returns:
    parameters -- Dictionary containing the trained model parameters.
    """
    start_load = time.time()
    print("Prepare hyperparameters...")
    layers_dims, learning_rate, mini_batch_size, num_epochs = prepare_model_hyperparameters(train_x.shape, small_model=True)
    print("Hyperparameters prepared")

    print("Training the model for several epochs...")
    parameters, costs = L_layer_model(train_x, train_y, layers_dims, learning_rate, mini_batch_size, num_epochs = num_epochs, print_cost=True)
        
    training_time = time.time() - start_load
    print(f"Training completed and parameters saved in {training_time:.2f} seconds")
    
    show_costs_plot(costs, learning_rate)

    return parameters


def load_or_train_model(paths):
    """
    Loads existing model parameters or trains a new model if no parameters are found.

    Arguments:
    paths -- Dictionary, containing paths where model parameters might be stored or need to be saved.

    Returns:
    Dictionary of the model parameters.
    """
    params_file = paths["params_output"] + ".npy" 
    if os.path.exists(params_file):
        parameters = load_parameters(params_file)
        print("Parameters loaded successfully.")
    else:
        print("No parameters file found. Proceeding to train the model.")
        parameters = train_model(paths)
    return parameters

def evaluate_model(parameters, eval_x, eval_y):
    """
    Evaluate the accuracy of the model on a given development set.

    Arguments:
    parameters -- Dictionary, trained model parameters.
    eval_x -- Array, input data for the development set.
    dev_y -- Array, true labels for the development set.

    Returns:
    Float, classification accuracy of the model on the development set.
    """
    AL, _ = L_model_forward(eval_x, parameters)
    predictions = (AL > 0.5).astype(float)
    accuracy = np.mean(predictions == eval_y)

    return accuracy

def predict_image(image_path, parameters):
    """
    Generate a prediction for a given image using the trained neural network.

    Arguments:
    image_path -- String, path to the image to predict.
    parameters -- Dictionary, trained model parameters.

    Returns:
    Array, output of the neural network prediction.
    """
    normalized_image_vector = preprocess_image(image_path)
    single_image_vector = normalized_image_vector.reshape(-1, 1)
    print("Image shape:", single_image_vector.shape)
    AL, _ = L_model_forward(single_image_vector, parameters)

    return AL

def main():

    small_or_large = input("Would you like to run the model on a smal testing dataset or on the large puppy detection dataset? (s/l) ")
    small_model = small_or_large.lower() == 's'
    if small_model:
       
        train_x, train_y, test_x, test_y, classes = prepare_small_dataset()
        
        parameters = train_small_model(train_x, train_y)
        print("Evaluating on the train dataset...")
        accuracy = evaluate_model(parameters, train_x, train_y)
        print(f"Model accuracy on the train set: {accuracy:.2%}")
        print("Evaluating on the test dataset...")
        accuracy = evaluate_model(parameters, test_x, test_y)
        print(f"Model accuracy on the test set: {accuracy:.2%}")
    else:
        base_path = os.path.dirname(__file__)
        paths = initialize_paths(base_path)
        
        train_or_load = input("Would you like to retrain a new model or not (and instead use existing parameters)? (y/n) ")
        parameters = load_or_train_model(paths) if train_or_load.lower() == 'n' else train_model(paths)

        print("Preparing dev dataset...")
        dev_x, dev_y = prepare_dataset(paths["dev_dogs"], paths["dev_cats"], paths["dev_output"])
        print("Dev dataset prepared")

        print("Evaluating on the dev dataset...")
        accuracy = evaluate_model(parameters, dev_x, dev_y)
        print(f"Model accuracy on the dev set: {accuracy:.2%}")

        print("Evaluating on the train dataset...")
        train_x, train_y = prepare_dataset(paths["train_dogs"], paths["train_cats"], paths["train_output"])
        accuracy = evaluate_model(parameters, train_x, train_y)
        print(f"Model accuracy on the train set: {accuracy}")

        while True:
            user_image_path = input("Enter the path to the image you want to predict (or 'q' to quit): ")
            if user_image_path.lower() == 'q':
                break   # Exit the loop if user enters 'q'

            print("Your selected image path: ", user_image_path)
            prediction = predict_image(user_image_path, parameters)

            if prediction > 0.5:
                print(f"\nThe image is of a puppy! (with a certainty of {prediction})")
            else:
                print("\nThe image is not of a puppy, but of an impostor!")

    print("Thanks for using the puppy detector, I hope it was interesting and you could learn how a simple neural network could be used for image classification. Have a great day!")

if __name__ == "__main__":
    main()
