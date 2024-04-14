import os
import numpy as np
from utils.image_preprocessing import preprocess_image
from utils.neural_network import L_layer_model, L_model_forward
from utils.prepare_dataset import prepare_dataset
import matplotlib.pyplot as plt

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
    n_h_1 = 24
    n_h_2 = 12
    n_h_3 = 8
    n_y = 1  # Set a single output node for the classifier

    layers_dims = (n_x, n_h_1, n_h_2, n_h_3, n_y)
    learning_rate = 0.0007

    return layers_dims, learning_rate

def evaluate_model(parameters, dev_x, dev_y):
    AL, _ = L_model_forward(dev_x, parameters)
    predictions = (AL > 0.5).astype(float)
    accuracy = np.mean(predictions == dev_y)
    return accuracy

def predict_image(image_path, parameters):
    # Load and preprocess the image
    normalized_image_vector = preprocess_image(image_path)
    single_image_vector = normalized_image_vector.reshape(-1, 1)
    print("Image shape:", single_image_vector.shape)
    # Feed the image through the network to get a prediction
    AL, _ = L_model_forward(single_image_vector, parameters)
    return AL

def main():

    path_to_dog_train_set = "data/training_set/dogs"
    path_to_cat_train_set = "data/training_set/cats"
    path_to_train_output_folder = "data/training_set/"

    path_to_dog_dev_set = "data/dev_set/dogs"
    path_to_cat_dev_set = "data/dev_set/cats"
    path_to_dev_output_folder = "data/dev_set/"

    base_path = os.path.dirname(__file__)


    print("Preparing training dataset...")
    train_x, train_y, = prepare_dataset(base_path, path_to_dog_train_set, path_to_cat_train_set, path_to_train_output_folder)
    print("Training dataset prepared")

    print("Prepare hyperparameters...")
    layers_dims, learning_rate = prepare_model_hyperparameters(train_x.shape)
    print("Hyperparameters prepared")

    print("Train the model with several epochs....")
    # Train the model
    parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_epochs = 10, print_cost = True)

    # plot the training cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 1)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    print("Preparing dev dataset...")
    dev_x, dev_y, = prepare_dataset(path_to_dog_dev_set, path_to_cat_dev_set, path_to_dev_output_folder)
    print("Training dataset prepared")

    # Evaluate the model on dev
    print("Evalutating on the dev dataset...")
    accuracy = evaluate_model(parameters, dev_x, dev_y)
    print(f"Model accuracy on the dev set: {accuracy}")


    # Load and preprocess the image from the user
    user_image_path = input("Enter the path to the image you want to predict: ")
    prediction = predict_image(user_image_path, parameters)
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