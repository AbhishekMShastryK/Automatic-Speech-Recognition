import numpy as np
import pickle


def compute_softmax(x):
    """Compute softmax transformation in a numerically stable way.

    Args:
        x: logits, of the shape [d, batch]

    Returns:
        Softmax probability of the input logits, of the shape [d, batch].
    """
    out = x - np.max(x, 0, keepdims=True)
    out_exp = np.exp(out)
    exp_sum = np.sum(out_exp, 0, keepdims=True)
    probs = out_exp / exp_sum
    return probs


def compute_ce_loss(out, y, loss_mask):
    """Compute cross-entropy loss, averaged over valid training samples.

    Args:
        out: dnn output, of shape [dout, batch].
        y: integer labels, [batch].
        loss_mask: loss mask of shape [batch], 1.0 for valid sample, 0.0 for padded sample.

    Returns:
        Cross entropy loss averaged over valid samples, and gradient wrt output.
    """



class FeedForwardNetwork:

    def __init__(self, din, dout, num_hidden_layers, hidden_layer_width):


    def forward(self, x):
        """Forward the feedforward neural network.

        Args:
            x: shape [din, batch].

        Returns:
            Output of shape [dout, batch], and a list of hidden layer activations,
            each of the shape [hidden_layer_width, batch].
        """


    def backward(self, x, hidden, loss_grad, loss_mask):
        """Backpropagation of feedforward neural network.

        Args:
            x: input, of shape [din, batch].
            hidden: list of hidden activations, each of the shape [hidden_layer_width, batch].
            loss_grad: gradient with respect to out, of shape [dout, batch].
            loss_mask: loss mask of shape [batch], 1.0 for valid sample, 0.0 for padded sample.

        Returns:
            Returns gradient, averaged over valid samples, with respect to weights and biases.
        """


    def update_model(self, w_updates, b_updates):
        """Update the weights and biases of the model.

        Args:
            w_updates: a list of updates to each weight matrix.
            b_updates: a list of updates to weight bias vector.
        """
        self.weights = [w + u for w, u in zip(self.weights, w_updates)]
        self.biases = [b + u for b, u in zip(self.biases, b_updates)]

    def predict(self, x):
        """Compute predictions on a minibath.

        Args:
            x: input, of shape [din, batch].
        
        Returns:
            The discrete model predictions and the probabilities of predicting each class.
        """
        out, _ = self.forward(x)
        probs = compute_softmax(out)
        return np.argmax(out, 0), probs

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'weights': self.weights, 'biases': self.biases}, f)

    def restore_model(self, filename):
        with open(filename, 'rb') as f:
            loaded_dict = pickle.load(f)
            self.weights = loaded_dict['weights']
            self.biases = loaded_dict['biases']
