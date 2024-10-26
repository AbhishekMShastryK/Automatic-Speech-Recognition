import numpy as np
import pickle
import mnist_input_generator
import dnn
from train_mnist import evaluate

with open("mnist.pkl",'rb') as f:
    mnist_dict = pickle.load(f)

# Test data. 10000 samples.
test_iter = mnist_input_generator.MnistInputGenerator(mnist_dict['test_images']/255.0, mnist_dict['test_labels'], batch_size=200, shuffle=True)
while test_iter.epoch!=1:
    x, y = test_iter.next()
assert test_iter.total_num_steps == 50

del mnist_dict

# Instantiate the model.
din = 784
dout = 10
num_hidden_layers = 2
hidden_layer_width= 500
network = dnn.FeedForwardNetwork(din, dout, num_hidden_layers, hidden_layer_width)
network.restore_model('mnist_model.pkl')

# Final test with the last model.
test_error_rate = evaluate(network, test_iter)
print(f'test error_rate: {test_error_rate}')


# Check model predictions on test.
import matplotlib.pyplot as plt

x, y = test_iter.next()
x = np.transpose(x)
batch_size = x.shape[1]
pred_labels, _ = network.predict(x)

for i in range(batch_size):
    img = x[:, i].reshape(28,28) # First image in the training set.
    ground_truth = y[i]
    predicted = pred_labels[i]
    plt.imshow(img, cmap='gray')
    plt.title(f'truth label={ground_truth}, predicted label={predicted}')
    plt.show() # Show the image