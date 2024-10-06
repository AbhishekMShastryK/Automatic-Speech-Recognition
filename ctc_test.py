import ctc
import numpy as np
'''
# Use the following two lines to allow reloading the module
# after you make quick changes of your implementation.
from importlib import reload
reload(ctc)
'''

''' The logits are generated as follows.
B, Tin, C = 4, 5, 4
x = - np.random.rand(B, Tin, C)
x[:, :, 0] += 0.25
probs = np.exp(x) / np.sum(np.exp(x), 1, keepdims=True)
logits = np.log(probs)
'''

logits = np.array([[[-1.24300644, -1.53148002, -1.82497478, -1.84361387],
        [-2.03160175, -1.70886812, -1.64796946, -1.35971863],
        [-1.62002954, -1.40484963, -1.40301463, -1.76084657],
        [-1.98033258, -1.63302012, -2.09690261, -1.71650425],
        [-1.40877317, -1.82033641, -1.28140467, -1.45487309]],

       [[-1.96385535, -1.50716001, -1.9819731 , -1.83315796],
        [-1.87125787, -1.89124944, -1.56504305, -1.51213544],
        [-1.19393974, -1.39374673, -1.181992  , -1.53134799],
        [-1.60003132, -1.45696074, -1.60324448, -1.86780564],
        [-1.60525462, -1.9207874 , -1.92954589, -1.39051422]],

       [[-1.71517678, -1.52948705, -1.84688996, -1.82338406],
        [-1.47198252, -1.64516467, -1.28080548, -1.78481608],
        [-1.41079472, -1.35956706, -1.5350524 , -1.87975484],
        [-1.70789392, -1.80905966, -1.83021258, -1.41456818],
        [-1.7993792 , -1.77317859, -1.66793738, -1.29085457]],

       [[-1.55366847, -1.26996081, -1.89464267, -1.46784448],
        [-1.64916202, -2.01003219, -1.81682787, -1.43484833],
        [-1.67093085, -1.27157232, -1.09819259, -1.70241746],
        [-1.30096143, -1.81958572, -2.02875027, -1.53187437],
        [-1.9952732 , -1.94708287, -1.50456277, -2.0169981 ]]], dtype=np.float32)

assert logits.shape == (4, 5, 4)

input_lens = np.array([4, 3, 3, 5])
Tout = 3
labels = np.array([[1, 0, 0], [2, 2, 0], [1, 2, 3], [1, 2, 3]])
output_lens = np.array([1, 2, 3, 3])

best_costs, paths = ctc.compute_forced_alignment(-logits, input_lens, labels, output_lens)

expected_best_costs= np.array([5.98974431, 5.03522297, 4.69004737, 7.33527495], dtype=np.float32)
np.testing.assert_allclose(best_costs, expected_best_costs, atol=1e-5, rtol=1e-5)

expected_paths = np.array([[0, 1, 1, 1, 0],
       [2, 0, 2, 0, 0],
       [1, 2, 3, 0, 0],
       [1, 0, 2, 0, 3]], dtype=np.int32)
np.testing.assert_array_equal(paths, expected_paths)

