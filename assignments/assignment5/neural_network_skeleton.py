# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest

# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os


class NeuralNetwork:
    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # --- PLEASE READ --
        # Use the parameters below to train your feed-forward neural network.

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25

        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 1e-3

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        np.random.seed(0)  # Setting random seed for reproducibility.

        self.weights, self.biases = None, None  # Initializing weights and biases

        self.total_layers = (
            None  # Initializing the number of layers in the neural network.
        )

        """
        I have implemented the neural network as two lists, one with the weight matrices between each layer,
        and the other with the bias vectors.
        """
        if hidden_layer:
            self.weights = [
                np.random.randn(self.hidden_units, input_dim),
                np.random.randn(1, self.hidden_units),
            ]
            self.biases = [np.random.randn(self.hidden_units, 1), np.random.randn(1, 1)]
            self.total_layers = 3
        else:
            self.weights = [np.random.randn(1, input_dim)]
            self.biases = [np.random.randn(1, 1)]
            self.total_layers = 2

        self.sigmoid = lambda x: 1.0 / (
            1.0 + np.exp(-x)
        )  # The sigmoid activation function: 1 / (1 + e^(-x))

        self.sigmoid_derivative = lambda x: self.sigmoid(x) * (
            1 - self.sigmoid(x)
        )  # The derivative of the sigmoid activation function to be used in the backpropagation algorithm.

    def load_data(
        self, file_path: str = os.path.join(os.getcwd(), "data_breast_cancer.p")
    ) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data["x_train"], data["y_train"]
            self.x_test, self.y_test = data["x_test"], data["y_test"]

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""
        for _ in range(self.epochs):
            for x, y in zip(self.x_train, self.y_train):

                weights_gradient = [
                    None for weight in self.weights
                ]  # Initializing weight gradients for each layer which are going to be used to update the weights in the network.

                biases_gradient = [
                    None for bias in self.biases
                ]  # Initializing bias gradients for each layer which are going to be used to update the biases in the network.

                activation = np.expand_dims(x, axis=1)
                activations = [
                    activation
                ]  # A list for storing all the activations when doing forward propagation

                values = (
                    []
                )  # A list for storing weight * x + bias values without applying the activation function.

                for weight, bias in zip(self.weights, self.biases):
                    value = np.dot(weight, activation) + bias
                    values.append(value)

                    activation = self.sigmoid(value)
                    activations.append(activation)

                """
                Calculating the error delta from output layer to be propagated backwards in the network. It is calculated
                by taking the derivative of the loss function, which in our case is MSE, and multiply with derivate of
                the sigmoid function applied on the value that entered the last layer of the network.
                """

                error_delta = (activations[-1] - y) * self.sigmoid_derivative(
                    values[-1]
                )

                weights_gradient[-1] = np.dot(
                    error_delta, activations[-2].T
                )  # Setting error delta multiplied with the second last layer activations as weight gradient for last layer.

                biases_gradient[-1] = error_delta  # Setting error delta as bias gradient for last layer.

                """
                This for-loop does the same as the code from line 128 - 136, but for each layer in the network.
                Thus, the error is propagated backwards in the network, and the gradients for each layer are set.
                """
                for layer in range(2, self.total_layers):
                    error_delta = np.dot(
                        self.weights[-layer + 1].T, error_delta
                    ) * self.sigmoid_derivative(values[-layer])

                    weights_gradient[-layer] = np.dot(
                        error_delta, activations[-layer - 1].T
                    )

                    biases_gradient[-layer] = error_delta

                self.weights = [
                    weight - self.lr * weight_gradient
                    for weight, weight_gradient in zip(self.weights, weights_gradient)
                ]  # Updating the weights of the network by w_i - learning_rate * nabla w_i (w_i is the weight matrix at layer i, and nabla w_i is weight gradient.)

                self.biases = [
                    bias - self.lr * bias_gradient
                    for bias, bias_gradient in zip(self.biases, biases_gradient)
                ]  # Updating the biases of the network by b_i - learning_rate * nabla b_i (b_i is the bias vector at layer i, and nabla b_i is weight gradient.)

    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """

        # Applying forward propagation by performing weight * x + bias through the whole network.
        activation = np.expand_dims(x, axis=1)
        for weight, bias in zip(self.weights, self.biases):
            activation = self.sigmoid(np.dot(weight, activation) + bias)
        return activation[0][0]  # Probability for the input x belonging to class 1.


class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, "The prediction needs to be in [0, 1] range."
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""
        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        self.assertTrue(
            accuracy > self.threshold,
            "This implementation is most likely wrong since "
            f"the accuracy ({accuracy}) is less than {self.threshold}.",
        )

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""
        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        self.assertTrue(
            accuracy > self.threshold,
            "This implementation is most likely wrong since "
            f"the accuracy ({accuracy}) is less than {self.threshold}.",
        )


if __name__ == "__main__":
    unittest.main()
