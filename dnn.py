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
    batch = out.shape[1]
    probs = compute_softmax(out)
    neg_log_probs = - np.log(probs)

    y_one_hot = np.zeros_like(out)
    col_idx = np.arange(batch, dtype=np.int32)
    rol_idx = np.array(y, dtype=np.int32)
    y_one_hot[rol_idx, col_idx] = 1.0
    loss_vec = np.sum(neg_log_probs * y_one_hot, 0)
    loss = np.sum(loss_vec * loss_mask) / np.sum(loss_mask)
    return loss, probs - y_one_hot


class FeedForwardNetwork:

    def __init__(self, din, dout, num_hidden_layers, hidden_layer_width):
        self.din = din
        self.dout = dout
        self.num_hidden_layers = num_hidden_layers
        self.weights = []
        self.biases = []
        widths = [din] + [hidden_layer_width] * num_hidden_layers + [dout]
        for i in range(self.num_hidden_layers+1):
            w = np.random.uniform(-0.05, 0.05, size=[widths[i+1], widths[i]])
            self.weights.append(w)
            b = np.zeros(widths[i+1])
            self.biases.append(b)

    def forward(self, x):
        """Forward the feedforward neural network.

        Args:
            x: shape [din, batch].

        Returns:
            Output of shape [dout, batch], and a list of hidden layer activations,
            each of the shape [hidden_layer_width, batch].
        """
        res = x
        hidden = []
        for i in range(self.num_hidden_layers+1):
            res = np.matmul(self.weights[i], res) + self.biases[i][:, None]
            if i < self.num_hidden_layers:
                # Apply ReLU activation.
                res[res < 0] = 0
                hidden.append(res)
        return res, hidden

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
        num_samples = np.sum(loss_mask)

        loss_grad = loss_grad * loss_mask[None, :]
        # Last layer is linear, no activation.
        # shape [dout, hidden_layer_width]
        grad_w = [np.matmul(loss_grad, np.transpose(hidden[-1])) / num_samples]
        grad_b = [np.sum(loss_grad, 1) / num_samples]
        # shape [hidden_layer_width, n]
        grad_hidden = np.matmul(np.transpose(self.weights[-1]), loss_grad)

        for i in range(self.num_hidden_layers-1, -1, -1):
            activation = hidden[i]
            # Gradient of ReLU activation.
            # shape [hidden_layer_width, n]
            grad_hidden[activation <= 0] = 0
            grad_hidden *= loss_mask[None, :]
            if i==0:
                prev_activation = x
            else:
                prev_activation = hidden[i-1]
            grad_w.insert(0, np.matmul(grad_hidden, np.transpose(prev_activation)) / num_samples)
            grad_b.insert(0, np.sum(grad_hidden, 1) / num_samples)
            # shape [prev_layer_width, n]
            grad_hidden = np.matmul(np.transpose(self.weights[i]), grad_hidden)

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
