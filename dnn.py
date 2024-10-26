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
    dout, batch_size = out.shape

    # Compute softmax probabilities
    probs = compute_softmax(out)

    # Convert labels to one-hot encoding
    one_hot_labels = np.zeros_like(out)
    one_hot_labels[y, np.arange(batch_size)] = 1.0

    # Calculate cross-entropy loss per sample
    log_probs = np.log(probs + 1e-12)  # Adding epsilon to avoid log(0)
    per_sample_loss = -np.sum(one_hot_labels * log_probs, axis=0)

    # Apply loss mask to cross-entropy loss
    masked_loss = per_sample_loss * loss_mask

    # Average the loss only over valid samples
    valid_sample_count = np.sum(loss_mask)  # Sum of valid samples
    if valid_sample_count > 0:
        average_loss = np.sum(masked_loss) / valid_sample_count
    else:
        average_loss = 0.0

    # Calculate gradient with respect to output
    grad_out = (probs - one_hot_labels)  # Gradient of softmax cross-entropy loss
    grad_out *= loss_mask[np.newaxis, :]  # Apply the mask to the gradient

    return average_loss, grad_out

class FeedForwardNetwork:

    def __init__(self, din, dout, num_hidden_layers, hidden_layer_width):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        # First layer
        self.weights.append(np.random.uniform(-0.05, 0.05, (hidden_layer_width, din)))
        self.biases.append(np.zeros(hidden_layer_width))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.weights.append(np.random.uniform(-0.05, 0.05, (hidden_layer_width, hidden_layer_width)))
            self.biases.append(np.zeros(hidden_layer_width))

        # Output layer
        self.weights.append(np.random.uniform(-0.05, 0.05, (dout, hidden_layer_width)))
        self.biases.append(np.zeros(dout))

    def forward(self, x):
        """Forward the feedforward neural network.

        Args:
            x: shape [din, batch].

        Returns:
            Output of shape [dout, batch], and a list of hidden layer activations,
            each of the shape [hidden_layer_width, batch].
        """
        activations = []
        a = x

        # Hidden layers with ReLU activation
        for i in range(self.num_hidden_layers):
            z = np.dot(self.weights[i], a) + self.biases[i][:, np.newaxis]
            a = np.maximum(0, z)  # ReLU activation
            activations.append(a)

        # Output layer without activation
        out = np.dot(self.weights[-1], a) + self.biases[-1][:, np.newaxis]
        return out, activations

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
        grad_w = []
        grad_b = []

        # Apply mask to loss gradient
        delta = loss_grad * loss_mask[np.newaxis, :]

        # Gradient for the output layer
        grad_w_out = np.dot(delta, hidden[-1].T)
        grad_b_out = np.sum(delta, axis=1)

        grad_w.insert(0, grad_w_out)
        grad_b.insert(0, grad_b_out)

        # Backpropagate through hidden layers
        for i in range(self.num_hidden_layers - 1, -1, -1):
            delta = np.dot(self.weights[i + 1].T, delta)
            delta[hidden[i] <= 0] = 0  # ReLU derivative

            if i == 0:
                grad_w_hidden = np.dot(delta, x.T)
            else:
                grad_w_hidden = np.dot(delta, hidden[i - 1].T)

            grad_b_hidden = np.sum(delta, axis=1)

            grad_w.insert(0, grad_w_hidden)
            grad_b.insert(0, grad_b_hidden)

        # Normalize gradients by the number of valid samples
        valid_sample_count = np.sum(loss_mask)
        if valid_sample_count > 0:
            grad_w = [g / valid_sample_count for g in grad_w]
            grad_b = [g / valid_sample_count for g in grad_b]

        return grad_w, grad_b

    def update_model(self, w_updates, b_updates):
        """Update the weights and biases of the model.

        Args:
            w_updates: a list of updates to each weight matrix.
            b_updates: a list of updates to weight bias vector.
        """
        self.weights = [w + u for w, u in zip(self.weights, w_updates)]
        self.biases = [b + u for b, u in zip(self.biases, b_updates)]

    def predict(self, x):
        """Compute predictions on a minibatch.

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
