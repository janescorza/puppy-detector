import os
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.image_preprocessing import preprocess_image
from utils.neural_network import L_layer_model, L_model_forward
from utils.prepare_dataset import load_parameters, prepare_dataset, save_data

def prepare_model_hyperparameters(X_shape):
    """
    Prepare and tune the hyperparameters for the neural network model based on the input shape.

    Arguments:
    X_shape -- Tuple containing the shape of the input dataset.

    Returns:
    Tuple of layer dimensions, learning rate, mini-batch size, and number of epochs.

    """
    if not X_shape or X_shape[1] == 0:
        raise ValueError('The training set is empty. Please check your path to the training set and the files in the folder.')
    n_x = X_shape[0]
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
    """
    Evaluate the accuracy of the model on a given development set.

    Arguments:
    parameters -- Dictionary, trained model parameters.
    dev_x -- Array, input data for the development set.
    dev_y -- Array, true labels for the development set.

    Returns:
    Float, classification accuracy of the model on the development set.
    """
    AL, _ = L_model_forward(dev_x, parameters)
    predictions = (AL > 0.5).astype(float)
    accuracy = np.mean(predictions == dev_y)

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

    base_path = os.path.dirname(__file__)

    relative_path_to_dog_train_set = "data/training_set/dogs"
    relative_path_to_cat_train_set = "data/training_set/cats"
    relative_path_to_train_output_folder = "data/training_set/"

    relative_path_to_dog_dev_set = "data/dev_set/dogs"
    relative_path_to_cat_dev_set = "data/dev_set/cats"
    relative_path_to_dev_output_folder = "data/dev_set/"

    relalative_path_to_params_output = "model/params.npy"
    path_to_params_output =  os.path.join(base_path, relalative_path_to_params_output)  


    # train and load could be used as input words once this is no longer in dev
    train_or_load = input("Would you like to retrain (y) a new model or not (n) and instead use existing parameters? ")

    if train_or_load.lower() == 'n' and os.path.exists(path_to_params_output):
        parameters = load_parameters(path_to_params_output)
        print("Parameters loaded successfully.")
    else:
        if not os.path.exists(path_to_params_output):
            print("No parameters file found. Proceeding to train the model.")

        start_load = time.time()
        print("Preparing training dataset...")
        train_x, train_y = prepare_dataset(base_path, relative_path_to_dog_train_set, relative_path_to_cat_train_set, relative_path_to_train_output_folder)
        load_time = time.time() - start_load 
        print(f"Training dataset prepared in {load_time:.2f} seconds")

        start_train = time.time()
        print("Prepare hyperparameters...")
        layers_dims, learning_rate, mini_batch_size, num_epochs = prepare_model_hyperparameters(train_x.shape)
        print("Hyperparameters prepared")

        print("Train the model for several epochs....")
        parameters, costs = L_layer_model(train_x, train_y, layers_dims, learning_rate=learning_rate, mini_batch_size=mini_batch_size, num_epochs = num_epochs, print_cost = True)

        # plot the training cost
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 1)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

        save_data(parameters, path_to_params_output)

        training_time = time.time() - start_train
        print(f"Training completed and parameters saved in {training_time:.2f} seconds")

    print("Preparing dev dataset...")
    dev_x, dev_y = prepare_dataset(base_path, relative_path_to_dog_dev_set, relative_path_to_cat_dev_set, relative_path_to_dev_output_folder)
    print("Training dataset prepared")

    print("Evaluating on the dev dataset...")
    accuracy = evaluate_model(parameters, dev_x, dev_y)
    print(f"Model accuracy on the dev set: {accuracy}")

    print("Evaluating on the train dataset...")
    train_x, train_y = prepare_dataset(base_path, relative_path_to_dog_train_set, relative_path_to_cat_train_set, relative_path_to_train_output_folder)
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

    print("Thanks for using the puppy detector, I hope it was interesting and you could learn how a simple neural network can be used for image classification. Have a great day!")



if __name__ == "__main__":
    main()