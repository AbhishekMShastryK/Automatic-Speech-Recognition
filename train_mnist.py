import numpy as np
import pickle
import mnist_input_generator
import dnn


with open("mnist.pkl",'rb') as f:
    mnist_dict = pickle.load(f)

# Training data. 59000 samples. Divide by 255 to converge unit8 values to the range [0, 1].
train_iter = mnist_input_generator.MnistInputGenerator(mnist_dict['training_images'][:59000]/255.0, mnist_dict['training_labels'][:59000], batch_size=200, shuffle=True)
while train_iter.epoch!=1:
    x, y = train_iter.next()
assert train_iter.total_num_steps == 295

# Validation data. 1000 samples.
valid_iter = mnist_input_generator.MnistInputGenerator(mnist_dict['training_images'][59000:]/255.0, mnist_dict['training_labels'][59000:], batch_size=200, shuffle=True)
while valid_iter.epoch!=1:
    x, y = valid_iter.next()
assert valid_iter.total_num_steps == 5

del mnist_dict


# Define model.
din = 784
dout = 10
num_hidden_layers = 2
hidden_layer_width= 500
network = dnn.FeedForwardNetwork(din, dout, num_hidden_layers, hidden_layer_width)


def evaluate(model, data_iter):
    data_iter.epoch = 0
    data_iter.step_in_epoch = 0
    total_num_samples = 0.0
    total_num_errors = 0.0

    # Need exactly one pass.
    while data_iter.epoch != 1:
        x, y = data_iter.next()
        # dnn takes input with the format [din, batch_size].
        x = np.transpose(x)
        batch_size = x.shape[1]
        total_num_samples += batch_size

        pred_labels, _ = model.predict(x)
        num_errors = np.sum(np.not_equal(pred_labels, y))
        total_num_errors += num_errors

    return total_num_errors / total_num_samples


def train():
    epoch = 0
    train_iter.epoch = 0
    train_iter.step_in_epoch = 0
    learning_rate = 0.001
    beta = 0.99
    momentum_w = [np.zeros_like(w) for w in network.weights]
    momentum_b = [np.zeros_like(b) for b in network.biases]

    # Check initial error_rate.
    valid_error_rate = evaluate(network, valid_iter)
    print(f'epoch {train_iter.epoch}, validation error_rate={valid_error_rate}')

    while train_iter.epoch != 50:

        x, y = train_iter.next()
        # dnn takes input with the format [din, batch_size].
        x = np.transpose(x)
        batch_size = x.shape[1]
        # There is no padded data.
        loss_mask = np.ones([batch_size], dtype=np.float32)

        # Forward pass and back-propagation.
        out, hidden = network.forward(x)
        loss, loss_grad = dnn.compute_ce_loss(out, y, loss_mask)
        grad_w, grad_b = network.backward(x, hidden, loss_grad, loss_mask)

        # Gradient descent with given learning rate.
        momentum_w = [beta * mw + gw for (mw, gw) in zip (momentum_w, grad_w)]
        momentum_b = [beta * mb + gb for (mb, gb) in zip (momentum_b, grad_b)]
        w_updates = [-learning_rate * mw for mw in momentum_w]
        b_updates = [-learning_rate * mb for mb in momentum_b]
        network.update_model(w_updates, b_updates)

        # Just finished one epoch. Check validation performance.
        if epoch != train_iter.epoch:
            network.save_model('mnist_model.pkl')
            valid_error_rate = evaluate(network, valid_iter)
            print(f'epoch {train_iter.epoch}, validation error_rate={valid_error_rate}')
            epoch = train_iter.epoch


if __name__=="__main__":
    train()
