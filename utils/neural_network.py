
import numpy as np
import math
import time
from sklearn.utils import shuffle


def initialize_parameters_he(layers_dims):
    """
    Initialize parameters with He initialization for the weights and zeros for the biases.

    Arguments:
    layer_dims -- List containing the size of each layer.
    
    Returns:
    parameters -- Dictionary containing the parameters "W1", "b1", ..., "WL", "bL":
                  "W1" -- weight matrix of shape (layers_dims[1], layers_dims[0])
                  "b1" -- bias vector of shape (layers_dims[1], 1)
                  ...
                  "WL" -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                  "bL" -- bias vector of shape (layers_dims[L], 1)
    """
    
    parameters = {}
    L = len(layers_dims) - 1    # Number of layers in the network
    print("🚀 ~ layers_dims:", layers_dims)
    for l in range(1, L + 1):
        parameters['W'+str(l)]= np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
        parameters['b'+str(l)]= np.zeros((layers_dims[l],1))
    return parameters

def initialize_adam(parameters) :
    """
    Initializes v and s for the Adam optimizer.
    
    Arguments:
    parameters -- Dictionary containing parameters "W1", "b1", ..., "WL", "bL".

    Returns:
    v -- Dictionary that will contain the exponentially weighted average of the gradient. Initialized with zeros.
    s -- Dictionary that will contain the exponentially weighted average of the squared gradient. Initialized with zeros.
    """
    
    L = len(parameters) // 2    # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        
    return v, s

def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- True "label" vector, shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    shuffled_X, shuffled_Y = shuffle(X.T, Y.T, random_state=42)
    shuffled_X = shuffled_X.T
    shuffled_Y = shuffled_Y.T

    # Step 2: Partition (shuffled_X, shuffled_Y).
    num_complete_minibatches = math.floor(m / mini_batch_size)  # Number of mini batches of size mini_batch_size
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:,(k+1)*mini_batch_size:((k+1)*mini_batch_size)+(m % mini_batch_size)]
        mini_batch_Y = shuffled_Y[:,(k+1)*mini_batch_size:((k+1)*mini_batch_size)+(m % mini_batch_size)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using the Adam optimization algorithm.
    
    Arguments:
    parameters -- Dictionary containing your parameters "W1", "b1", ..., "WL", "bL".
    grads -- Dictionary containing your gradients for each parameter "dW1", "db1", ..., "dWL", "dbL".
    v -- Adam variable, moving average of the first gradient, dictionary.
    s -- Adam variable, moving average of the squared gradient, dictionary.
    t -- Adam variable, counts the number of steps taken.
    learning_rate -- The learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates.
    beta2 -- Exponential decay hyperparameter for the second moment estimates.
    epsilon -- Hyperparameter preventing division by zero in Adam updates.

    Returns:
    parameters -- Dictionary containing your updated parameters.
    v -- Adam variable, moving average of the first gradient, dictionary.
    s -- Adam variable, moving average of the squared gradient, dictionary.
    """
    
    L = len(parameters) // 2    # number of layers in the neural networks
    v_corrected = {}    # Initializing first moment estimate, python dictionary
    s_corrected = {}    # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(1, L + 1):
        # Moving average of the gradients
        v["dW" + str(l)] = beta1*v["dW" + str(l)] + (1-beta1)*(grads["dW" + str(l)])
        v["db" + str(l)] = beta1*v["db" + str(l)] + (1-beta1)*(grads["db" + str(l)])

        # Compute bias-corrected first moment estimate
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1-np.power(beta1, t))
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1-np.power(beta1, t))
        # Moving average of the squared gradients
        s["dW" + str(l)] = beta2*s["dW" + str(l)] + (1-beta2)*(np.power(grads["dW" + str(l)],2))
        s["db" + str(l)] = beta2*s["db" + str(l)] + (1-beta2)*(np.power(grads["db" + str(l)],2))
        # Compute bias-corrected second raw moment estimate
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1-np.power(beta2, t))
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1-np.power(beta2, t))

        # Update parameters
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate*(v_corrected["dW" + str(l)]/(np.sqrt(s_corrected["dW" + str(l)])+epsilon))
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate*(v_corrected["db" + str(l)]/(np.sqrt(s_corrected["db" + str(l)])+epsilon))

    return parameters, v, s, v_corrected, s_corrected

def linear_forward(A, W, b):
    """
    Linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W,A)+b
    cache = (A, W, b)
    
    return Z, cache

def sigmoid(z):
    """
    Sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.
    
    Return:
    s -- sigmoid(z)
    cache -- a tuple containing "Z"
    """
    z = np.clip(z, -10, 10) # To prevent overflows in the exponential calculation within the sigmoid function
    s = 1/(1+np.exp(-z))
    cache = z

    return s, cache


def relu(z):
    """
    ReLU of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    r -- ReLU(z)
    cache -- a tuple containing "Z"
    """
    r = np.maximum(0, z)
    cache = z

    return r, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z) 
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []
    A = X   # activations in the first layer (input data)
    L = len(parameters) // 2    # number of layers in the neural network

    
    # [LINEAR -> RELU]*(L-1) and adding "cache" to the "caches" list.
    # The for loop starts at 1 because layer 0 is the input layer
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    # LINEAR -> SIGMOID for the last layer.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
          
    return AL, caches

def compute_cost(AL, Y, epsilon):
    """
    Cost function for a neural network with binary classification.

    Arguments:
    AL -- probability vector corresponding to label predictions, shape (1, number of examples)
    Y -- true "label" vector, shape (1, number of examples)
    epsilon -- hyperparameter preventing -infinity or division by zero in cost calculation

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from AL with respect to Y .
    cost = -(1./m) * (np.dot(Y, np.log(AL + epsilon).T) + np.dot(1-Y, np.log(1-AL + epsilon).T))

    
    cost = np.squeeze(cost) # To make sure the cost is the expected integer(e.g. this turns [[17]] into 17).

    return cost

def linear_backward(dZ, cache):
    """
    Linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ,A_prev.T)
    db = 1/m * np.sum(dZ,axis=1, keepdims=True) # db is summed by the rows of dZ
    dA_prev = np.dot(W.T,dZ)
    
    return dA_prev, dW, db

def relu_backward(dA, Z):
    """
    Gradient of the ReLU function

    Arguments:
    dA -- post-activation gradient for current layer l
    Z -- output of the forward propagation of the same layer l

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    dZ = np.array(dA, copy=True)    # initialize dZ to be a copy of dA
    dZ[Z <= 0] = 0  # Applying the derivative of the ReLu activation function.

    return dZ

def sigmoid_backward(dA, Z):
    """
    Gradient of the sigmoid function

    Arguments:
    dA -- post-activation gradient for current layer l
    Z -- output of the forward propagation of the same layer l

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    s, cache = sigmoid(Z)
    dZ = dA * s * (1 - s)

    return dZ


def linear_activation_backward(dA, cache, activation):
    """
    Backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA ,activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ =sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, epsilon):
    """
    Backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID model
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it caches[l], for l in range(L-1))
                the cache of linear_activation_forward() with "sigmoid" (it caches[L-1])
    epsilon -- hyperparameter preventing division by zero

    
    Returns:
    grads -- A dictionary with the gradients
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL + epsilon) - np.divide(1 - Y, 1 - AL + epsilon))   # derivative of the cost with respect to AL
        
    # Lth layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
        
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):

        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp
        
    return grads

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0007, mini_batch_size = 64,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 5000, print_cost = True):
    """
    L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if dog, 0 if non-dog), of shape (1, number of examples)
    number_of_examples -- total number of examples across all minibatches
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    mini_batch_size -- the size of a mini batch
    num_iterations -- number of iterations of the optimization loop
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- if True, it prints the cost every certain number of epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    L = len(layers_dims)    # number of layers in the neural networks
    costs = []
    t = 0   # initializing the counter required for Adam update
    m = X.shape[1]  # number of examples accross all minibatches
    
    parameters = initialize_parameters_he(layers_dims)

    v,s = initialize_adam(parameters)

    for i in range(num_epochs):

        minibatches = random_mini_batches(X, Y, mini_batch_size)
        cost_total = 0

        start_epoch = time.time()

        for j, minibatch in enumerate(minibatches):
            # selection of a minibatch
            (minibatch_X, minibatch_Y) = minibatch 

            start_forward = time.time()
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = L_model_forward(minibatch_X, parameters)
            forward_time = time.time() - start_forward
            
            # Compute cost and add to the cost total
            cost_total += compute_cost(AL, minibatch_Y, epsilon)
                    
            start_backward = time.time()
            # Backward propagation
            grads = L_model_backward(AL, minibatch_Y, caches, epsilon)
            backward_time = time.time() - start_backward
    
            # Update parameters.
            start_update = time.time()
            t = t + 1 # Adam counter increase
            parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,  epsilon)
            update_time = time.time() - start_update

            # Uncomment for detailed timing information
            # print(f"Epoch {i}, Minibatch {j}: Forward {forward_time:.2f}s, Backward {backward_time:.2f}s, Update {update_time:.2f}s Min AL={np.min(AL)}, Max AL={np.max(AL)}")
        
        epoch_time = time.time() - start_epoch
        print(f"Epoch {i} completed in {epoch_time:.2f} seconds")

        cost_avg = cost_total / m
        # Print the cost every 1 epochs
        if print_cost and i % 1 == 0:
            print("🚀 Cost after epoch %i: %f" %(i, cost_avg))
        # Append cost every 1 epochs
        if print_cost and i % 1 == 0:
            costs.append(cost_avg)
    
    

    return parameters, costs

