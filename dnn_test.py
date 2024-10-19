import numpy as np
import dnn
'''
# Use the following two lines to allow reloading the module
# after you make quick changes of your implementation.
from importlib import reload
reload(dnn)
'''

din = 3
dout = 4
num_hidden_layers = 2
hidden_layer_width= 5

network = dnn.FeedForwardNetwork(din, dout, num_hidden_layers, hidden_layer_width)
weights = network.weights
biases = network.biases
np.testing.assert_array_equal(biases[0], np.zeros(5))
np.testing.assert_array_equal(weights[0].shape, [5, 3])
np.testing.assert_array_less(np.fabs(weights[0]), 0.05)
np.testing.assert_array_equal(biases[1], np.zeros(5))
np.testing.assert_array_equal(weights[1].shape, [5, 5])
np.testing.assert_array_less(np.fabs(weights[1]), 0.05)
np.testing.assert_array_equal(biases[2], np.zeros(4))
np.testing.assert_array_equal(weights[2].shape, [4, 5])
np.testing.assert_array_less(np.fabs(weights[2]), 0.05)

# Load saved model.
network.restore_model('saved_model.pkl')

batch = 5
x = np.array([[-0.06033798, -0.8918106 , -0.70005365],
       [ 1.95022332, -0.64767359, -0.22318488],
       [ 0.52776347,  1.41395595, -0.23890085],
       [-0.65497405, -1.49502141, -1.63897778],
       [ 0.44595019, -0.84741226, -1.45960806]], dtype=np.float32)
x = np.transpose(x)
out, hidden = network.forward(x)
expected_out = np.array([[ 4.34743082e-06,  1.25414951e-05,  0.00000000e+00,
         6.06637245e-06,  4.83825908e-06],
       [ 3.27698390e-05,  1.80326909e-05,  0.00000000e+00,
         5.57702747e-05,  1.01536209e-05],
       [-2.25149958e-05, -1.06616297e-05,  0.00000000e+00,
        -4.12358368e-05, -3.79635519e-06],
       [ 8.33065005e-06,  5.47776651e-06,  0.00000000e+00,
         1.24731287e-05,  5.18307481e-06]], dtype=np.float32)
np.testing.assert_allclose(out, expected_out, rtol=1e-7, atol=1e-7)


y = np.array([1, 0, 3, 2, 3], dtype=np.int32)
loss, loss_grad = dnn.compute_ce_loss(out, y, loss_mask=np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
expected_loss = 1.3862973983655673
np.testing.assert_allclose(loss, expected_loss, atol=1e-5, rtol=1e-5)
expected_loss_grad = np.array([[ 0.24999965, -0.74999845,  0.25      ,  0.24999945,  0.25000019],
       [-0.74999324,  0.25000292,  0.25      ,  0.25001188,  0.25000151],
       [ 0.24999294,  0.24999575,  0.25      , -0.75001238,  0.24999803],
       [ 0.25000065,  0.24999978, -0.75      ,  0.25000105, -0.74999973]], dtype=np.float32)
np.testing.assert_allclose(loss_grad, expected_loss_grad, rtol=1e-7, atol=1e-7)


grad_w, grad_b = network.backward(x, hidden, loss_grad, loss_mask=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32))
expected_grad_w0 = np.array([[-2.45092894e-04, -1.66730083e-04, -2.87217738e-04],
       [ 9.65504039e-06,  6.66218213e-05,  1.39545252e-04],
       [ 2.91363832e-04,  2.94834397e-04,  4.59528208e-04],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-2.78619851e-05, -3.23049618e-05, -8.05187829e-05]], dtype=np.float32)
np.testing.assert_allclose(grad_w[0], expected_grad_w0, rtol=1e-7, atol=1e-7)
expected_grad_b0 = np.array([-1.76370249e-05, -7.20692119e-05,  8.80053607e-07,  0.00000000e+00, 3.15515168e-05], dtype=np.float32)
np.testing.assert_allclose(grad_b[0], expected_grad_b0, rtol=1e-7, atol=1e-7)


grad_w, grad_b = network.backward(x, hidden, loss_grad, loss_mask=np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32))
expected_grad_w0 = np.array([[ 3.82779351e-05,  3.92541146e-04,  2.89638516e-04],
       [-7.42157289e-05,  1.41027683e-04,  2.42910267e-04],
       [-4.93721720e-05, -7.29733190e-04, -5.72826083e-04],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 4.41691597e-05, -8.39319959e-05, -1.44566961e-04]], dtype=np.float32)
np.testing.assert_allclose(grad_w[0], expected_grad_w0, rtol=1e-7, atol=1e-7)
expected_grad_b0 = np.array([-4.39002733e-04, -1.66421569e-04,  8.18260283e-04,  0.00000000e+00, 9.90450541e-05], dtype=np.float32)
np.testing.assert_allclose(grad_b[0], expected_grad_b0, rtol=1e-7, atol=1e-7)
