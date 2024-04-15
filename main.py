import os
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.image_preprocessing import preprocess_image
from utils.neural_network import L_layer_model, L_model_forward
from utils.prepare_dataset import load_parameters, prepare_dataset, save_data

def prepare_model_hyperparameters(X_shape):
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
    n_h_1 = 1024
    n_h_2 = 512
    n_h_3 = 256
    n_y = 1  # Set a single output node for the classifier

    layers_dims = (n_x, n_h_1, n_h_2, n_h_3, n_y)
    learning_rate = 0.001
    mini_batch_size = 128
    num_epochs = 5

    return layers_dims, learning_rate, mini_batch_size, num_epochs

def evaluate_model(parameters, dev_x, dev_y):
    AL, _ = L_model_forward(dev_x, parameters)
    predictions = (AL > 0.5).astype(float)
    accuracy = np.mean(predictions == dev_y)
    return accuracy

# def predict_image(image_path, parameters, train_mean, train_std):
def predict_image(image_path, parameters):
    # Load and preprocess the image
    normalized_image_vector = preprocess_image(image_path)
    single_image_vector = normalized_image_vector.reshape(-1, 1)
    print("Image shape:", single_image_vector.shape)
    # Feed the image through the network to get a prediction
    AL, _ = L_model_forward(single_image_vector, parameters)
    return AL

def main():

    relative_path_to_dog_train_set = "data/training_set/dogs"
    relative_path_to_cat_train_set = "data/training_set/cats"
    relative_path_to_train_output_folder = "data/training_set/"

    relative_path_to_dog_dev_set = "data/dev_set/dogs"
    relative_path_to_cat_dev_set = "data/dev_set/cats"
    relative_path_to_dev_output_folder = "data/dev_set/"

    relalative_path_to_params_output_folder = "model/params"

    base_path = os.path.dirname(__file__)

    path_to_params_output_folder =  os.path.join(base_path, relalative_path_to_params_output_folder)  


    # train and load could be used as input words once this is no longer in dev
    train_or_load = input("Would you like to train (y) a new model or instead use existing parameters (n)? ")

    if train_or_load.lower() == 'n' and os.path.exists(path_to_params_output_folder):
        parameters = load_parameters(path_to_params_output_folder)
        print("Parameters loaded successfully.")
    else:
        if not os.path.exists(path_to_params_output_folder):
            print("No parameters file found. Proceeding to train the model.")

        start_load = time.time()
        print("Preparing training dataset...")
        train_x, train_y = prepare_dataset(base_path, relative_path_to_dog_train_set, relative_path_to_cat_train_set, relative_path_to_train_output_folder)
        # train_x, train_y, train_mean, train_std = prepare_dataset(base_path, relative_path_to_dog_train_set, relative_path_to_cat_train_set, relative_path_to_train_output_folder)
        print("Training dataset prepared")

        print("Prepare hyperparameters...")
        layers_dims, learning_rate, mini_batch_size, num_epochs = prepare_model_hyperparameters(train_x.shape)
        print("Hyperparameters prepared")

        print("Train the model with several epochs....")
        # Train the model
        parameters, costs = L_layer_model(train_x, train_y, layers_dims, learning_rate=learning_rate, mini_batch_size=mini_batch_size, num_epochs = num_epochs, print_cost = True)

        # plot the training cost
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 1)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

        save_data(parameters, path_to_params_output_folder)
        
        load_time = time.time() - start_load  # Time taken to load and preprocess data
        print(f"Training complete and parameters saved in {load_time:.2f} seconds")

    print("Preparing dev dataset...")
    dev_x, dev_y = prepare_dataset(base_path, relative_path_to_dog_dev_set, relative_path_to_cat_dev_set, relative_path_to_dev_output_folder)
    print("Training dataset prepared")

    # Evaluate the model on dev
    print("Evaluating on the dev dataset...")
    accuracy = evaluate_model(parameters, dev_x, dev_y)
    print(f"Model accuracy on the dev set: {accuracy}")


    # Load and preprocess the image from the user
    user_image_path = input("Enter the path to the image you want to predict: ")
    prediction = predict_image(user_image_path, parameters)
    # prediction = predict_image(user_image_path, parameters, train_mean, train_std)
    # Output the prediction
    if prediction > 0.5:
        print("The image is of a puppy!")
    else:
        print("The image is not of a puppy, of an impostor!")

    # Loop to keep asking for predictions until user quits
    while True:
        # Load and preprocess the image from the user
        user_image_path = input("Enter the path to the image you want to predict (or 'q' to quit): ")
        if user_image_path.lower() == 'q':
            break  # Exit the loop if user enters 'q'
        print("User image path: ", user_image_path)
        prediction = predict_image(user_image_path, parameters)
        # Output the prediction
        if prediction > 0.5:
            print("\nThe image is of a puppy!")
        else:
            print("\nThe image is not of a puppy, but of an impostor!")

    print("Thanks for using the puppy detector!")



if __name__ == "__main__":
    main()