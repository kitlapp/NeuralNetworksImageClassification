import pandas as pd
import tensorflow as tf
import numpy as np


def batch_and_prefetch_datasets(train_set, validation_set, test_set, batch_size):
    """
    Batches and prefetches the train, validation, and test sets to optimize performance.

    Args:
        train_set (tf.data.Dataset): The training dataset.
        validation_set (tf.data.Dataset): The validation dataset.
        test_set (tf.data.Dataset): The testing dataset.
        batch_size (int): The number of samples per batch.

    Returns:
        A tuple containing:
         - Batched and prefetched training dataset.
         - Batched and prefetched validation dataset.
         - Batched and prefetched testing dataset.
         - A batch of validation inputs.
         - A batch of validation targets.
    """
    # Batch and prefetch the datasets to improve performance:
    train_set = train_set.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    validation_set = validation_set.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_set = test_set.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Extract a batch of inputs and targets from the validation dataset:
    validation_set_inputs, validation_set_targets = next(iter(validation_set))

    return train_set, validation_set, test_set, validation_set_inputs, validation_set_targets


def create_image_model(input_shape, output_size, hidden_layer_sizes, activation_fun, activation_fun_output):
    """
    Create a neural network model for image processing with customizable hidden layers and
    activation functions.

    Parameters:
    - input_shape (tuple): Shape of the input images (e.g., (28, 28, 1)).
    - output_size (int): Number of output units, typically equal to the number of classes for
    classification problems (e.g., 10 for MNIST).
    - hidden_layer_sizes (list of int): List of integers where each integer represents the number
    of neurons in a hidden layer.
      Example: [50, 100, 50] defines a network with three hidden layers of sizes 50, 100, and 50
      respectively.
    - activation_fun (str): Activation function for the hidden layers. Options include 'relu',
    'sigmoid', 'tanh', etc.
    - activation_fun_output (str): Activation function for the output layer. Common choices are
    'softmax' for classification or 'sigmoid' for binary classification.

    Returns:
    - tf.keras.Model: A Keras Sequential model with the specified architecture.
    """

    # Initialize a Sequential model:
    model = tf.keras.Sequential()

    # Add the input layer with the specified input shape:
    model.add(tf.keras.layers.Input(shape=input_shape))

    # Flatten the input images to a 1D vector:
    model.add(tf.keras.layers.Flatten())

    # Add hidden layers based on the provided sizes and activation function:
    for size in hidden_layer_sizes:
        model.add(tf.keras.layers.Dense(size, activation=activation_fun))

    # Add the output layer with the specified activation function:
    model.add(tf.keras.layers.Dense(output_size, activation=activation_fun_output))

    return model


def train_eval_present_results(train_set, validation_inputs, validation_targets, test_set, model,
                               epochs, optimizer, learn_rate, mom):
    """
    Trains and evaluates a neural network model on the given datasets and returns the test accuracy
    and loss.

    Parameters:
    - train_set (tf.data.Dataset): The training dataset.
    - validation_inputs (tf.Tensor): The inputs of the validation dataset.
    - validation_targets (tf.Tensor): The targets of the validation dataset.
    - test_set (tf.data.Dataset): The test dataset.
    - model (tf.keras.Model): The neural network model to train and evaluate.
    - epochs (int): Number of epochs to train the model.
    - optimizer (str): The optimization technique to use ('adam' or 'sgd').
    - learn_rate (float): The learning rate for the optimizer.
    - mom (float): The momentum parameter for the SGD optimizer (ignored if using 'adam').

    Returns:
    - tuple: A tuple containing the test accuracy and test loss.
    """
    # Initialize the optimizer based on the specified type and parameters:
    if optimizer.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    elif optimizer.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learn_rate, momentum=mom)
    else:
        raise ValueError("Unsupported optimizer. Please use 'adam' or 'sgd'.")

    # Compile the model with the specified optimizer and loss function:
    model.compile(
        optimizer=optimizer,  # Hyperparameter
        loss='sparse_categorical_crossentropy',  # Hyperparameter but we won't change it
        metrics=['accuracy']
    )
    # Train the model with the training set and validate on the validation set:
    model.fit(train_set, epochs=epochs,
              validation_data=(validation_inputs, validation_targets), verbose=0)

    # Evaluate the model on the test set to assess performance:
    test_loss, test_accuracy = model.evaluate(test_set, verbose=2)

    return test_accuracy, test_loss


def gathered_information(n_range, input_shape, output_size, hidden_layer_sizes, activation_fun, activation_fun_output,
                         train_set, validation_inputs, validation_targets, test_set, epochs, optimizer, learn_rate,
                         mom):
    """
    Collects and summarizes the performance metrics of a neural network model by running
    multiple training and evaluation cycles. Returns a DataFrame containing the average and
    standard deviation of test accuracy and loss.

    Parameters:
    - n_range (int): The number of times to run the training and evaluation.
    - train_set (tf.data.Dataset): The training dataset.
    - validation_inputs (tf.Tensor): The inputs of the validation dataset.
    - validation_targets (tf.Tensor): The targets of the validation dataset.
    - test_set (tf.data.Dataset): The test dataset.
    - input_shape (tuple): Shape of the input images.
    - output_size (int): Number of output units (typically number of classes).
    - hidden_layer_sizes (list of int): List of integers where each integer represents the number
      of neurons in a hidden layer.
    - activation_fun (str): Activation function for the hidden layers.
    - activation_fun_output (str): Activation function for the output layer.
    - epochs (int): Number of epochs to train the model.
    - optimizer (str): The optimization technique to use ('adam' or 'sgd').
    - learn_rate (float): The learning rate for the optimizer.
    - mom (float): The momentum parameter for the SGD optimizer (ignored if using 'adam').

    Returns:
    - pd.DataFrame: A DataFrame containing the average and standard deviation of test accuracy
      and loss.
    """
    model_accuracy = []
    model_loss = []

    # Run the training and evaluation multiple times to ensure robustness
    for i in range(n_range):
        # Create a new model at the beginning of every cycle to ensure fresh initialization
        current_model = create_image_model(
            input_shape=input_shape,
            output_size=output_size,
            hidden_layer_sizes=hidden_layer_sizes,
            activation_fun=activation_fun,
            activation_fun_output=activation_fun_output
        )
        results = train_eval_present_results(
            train_set=train_set,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            test_set=test_set,
            model=current_model,
            epochs=epochs,
            optimizer=optimizer,
            learn_rate=learn_rate,
            mom=mom
        )

        accuracy, loss = results
        model_accuracy.append(accuracy)
        model_loss.append(loss)

    # Calculate average and standard deviation for accuracy and loss:
    average_test_acc = np.mean(model_accuracy)
    std_test_acc = np.std(model_accuracy)
    average_test_loss = np.mean(model_loss)
    std_test_loss = np.std(model_loss)

    # Create a dictionary to store the results:
    dict_final = {
        'Number of Runs': n_range,
        'Optimization Technique': optimizer,
        'Average Test Accuracy': average_test_acc,
        'Accuracy Standard Deviation': std_test_acc,
        'Average Test Loss': average_test_loss,
        'Loss Standard Deviation': std_test_loss
    }
    # Convert the dictionary to a DataFrame:
    dict_final = pd.DataFrame([dict_final]).transpose()
    dict_final.columns = ['Value']

    return dict_final
