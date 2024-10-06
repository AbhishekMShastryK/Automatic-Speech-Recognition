
import numpy as np
import input_generator
import time
'''
# Use the following two lines to allow reloading the module
# after you make quick changes of your implementation.
from importlib import reload
reload(input_generator)
'''
# Start timing
start_time = time.time()

# Tests for tokenizer.
tokenizer = input_generator.CharacterTokenizer()
res = tokenizer.StringToIds("HELLO WORLD")
assert res == [8, 5, 12, 12, 15, 28, 23, 15, 18, 12, 4]

res = tokenizer.IdsToString(res)
assert res == "HELLO WORLD"

# Test for splicing and subsampling.
x = np.reshape(np.arange(10), [5, 2])
output = input_generator.splice_and_subsample(x, 2, 3)
expected_output = np.array([[0, 1, 0, 1, 0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 6, 7, 8, 9, 8, 9]])
np.testing.assert_array_equal(output, expected_output)

# Tests for input generator.
train = input_generator.InputGenerator('train_data.json', batch_size=3, shuffle=True, context_length=1, subsampling_rate=2)
while train.epoch!=1:
    train.next()

# Debugging: Output the actual number of utterances and total number of steps for train
# print(f"Total number of utterances in train set: {len(train.utterances)}")
# print(f"Total number of steps in train: {train.total_num_steps}")
assert train.total_num_steps == 9513

dev = input_generator.InputGenerator('dev_data.json', batch_size=1, shuffle=False, context_length=1, subsampling_rate=2)
while dev.epoch!=1:
    batch = dev.next()
# Debugging: Output the actual number of utterances and total number of steps for dev
# print(f"Total number of utterances in dev set: {len(dev.utterances)}")
# print(f"Total number of steps in dev: {dev.total_num_steps}")
assert dev.total_num_steps == 2703

# The last utterance in the dev set json file.
assert len(batch) == 1
utt_id, x, y = batch[0]
assert utt_id == '8842-304647-0013'
assert tokenizer.IdsToString(y) == 'THOU LIKE ARCTURUS STEADFAST IN THE SKIES WITH TARDY SENSE GUIDEST THY KINGDOM FAIR BEARING ALONE THE LOAD OF LIBERTY'
assert x.shape == (443, 249)

# Test if the features are spliced and subsampled correctly.
x_snapshot = x[::100, ::20]
x_snapshot_expected = np.array([[-1.4427805 , -0.81610394, -0.1908448 , -1.2240493 ,  1.0337198 ,
        -1.5900221 , -0.6417345 , -1.0831673 , -0.7118636 , -1.3884459 ,
        -1.191491  , -0.9431238 , -1.0709589 ],
       [-0.89767   ,  0.73163056,  0.8631302 ,  0.43763936, -0.39725196,
         0.06540346,  0.39955908,  0.28316557,  0.22668439, -0.34037644,
         0.16509789,  0.17730421,  0.49456334],
       [ 0.03970152,  0.80783534,  1.167894  ,  1.9627504 , -1.3195344 ,
         1.5099506 ,  0.97416425,  1.5734375 ,  0.36320043,  0.4721861 ,
         0.02632302,  1.5215383 ,  0.46117696],
       [-0.13378555,  1.5407803 ,  2.1650908 , -0.00832719, -1.0797622 ,
         1.6549267 ,  1.9711747 ,  0.4862852 , -0.28525084,  0.34953523,
         1.8243071 ,  1.1194069 , -0.19495478],
       [ 0.14379376,  0.899281  ,  1.0536076 ,  0.68551457, -0.04561186,
         1.9267566 ,  1.7878052 ,  1.599719  , -0.6094765 ,  0.42619205,
         0.95611477,  1.5003734 , -0.28901154]], dtype=np.float32)
np.testing.assert_allclose(x_snapshot, x_snapshot_expected, rtol=1e-5, atol=1e-5)

# End timing
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")