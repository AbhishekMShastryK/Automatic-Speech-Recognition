import json
import kaldiio
import numpy as np

# Task 1: Tokenizer
class CharacterTokenizer:
    def __init__(self):
        # Character-to-ID mapping (A=1, B=2,...Z=26, apostrophe=27, space=28)
        self.char_to_id = {chr(i): i - 64 for i in range(65, 91)}  # A-Z
        self.char_to_id[' '] = 28  # Space
        self.char_to_id["'"] = 27  # Apostrophe
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}  # Reverse mapping
    
    def StringToIds(self, text):
        # Converts a string into a list of integer IDs
        return [self.char_to_id[char] for char in text]
    
    def IdsToString(self, ids):
        # Converts a list of IDs back into a string
        return ''.join([self.id_to_char[i] for i in ids])

# Task 2: Splice and Subsample
def splice_and_subsample(input_feature_matrix, C, r):
    # Get the number of frames (T)
    T = input_feature_matrix.shape[0]
    
    # Step 1: Splicing
    spliced_features = []
    for t in range(T):
        # Handle left context
        if t - C < 0:  # If less than C frames to the left, pad by repeating the first frame
            left_padding = np.tile(input_feature_matrix[0], (C - t, 1))
            left_context = np.vstack((left_padding, input_feature_matrix[0:t]))
        else:
            left_context = input_feature_matrix[t - C:t]

        # Handle right context
        if t + C >= T:  # If less than C frames to the right, pad by repeating the last frame
            right_padding = np.tile(input_feature_matrix[T - 1], (t + C + 1 - T, 1))
            right_context = np.vstack((input_feature_matrix[t+1:T], right_padding))
        else:
            right_context = input_feature_matrix[t + 1:t + C + 1]

        # Concatenate left context, current frame, and right context
        spliced_frame = np.concatenate((left_context, input_feature_matrix[t:t+1], right_context), axis=0)
        spliced_features.append(spliced_frame.flatten())  # Flatten the spliced frame
    
    spliced_features = np.array(spliced_features)  # Convert to a numpy array
    
    # Step 2: Subsampling
    T_prime = (T + r - 1) // r
    subsampled_features = spliced_features[::r]  # Take every r-th frame
    
    return subsampled_features[:T_prime]

# Task 3: Input Generator
class InputGenerator:
    def __init__(self, json_file, batch_size, shuffle, context_length, subsampling_rate):
        # Load data from JSON file
        with open(json_file, 'r') as f:
            self.data = json.load(f)["utts"]  # Access the "utts" key in the JSON
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.context_length = context_length
        self.subsampling_rate = subsampling_rate
        self.epoch = 0
        self.total_num_steps = 0
        self.steps_in_epoch = 0
        self.utterances = list(self.data.keys())  # Get list of utterances
        
        # Shuffle the utterances if required
        if shuffle:
            np.random.shuffle(self.utterances)

    def next(self):

        batch = []
        for _ in range(self.batch_size):
            utt_id = self.utterances[self.steps_in_epoch]
            features_path = self.data[utt_id]["feat"]  # This is the full ark path with offset
            
            # Load the acoustic features using kaldiio
            features = kaldiio.load_mat(features_path)  # Pass the ark path directly to load_mat
            
            # Process the features with splicing and subsampling
            spliced_subsampled_features = splice_and_subsample(features, self.context_length, self.subsampling_rate)
            
            tokenizer = CharacterTokenizer()  # Instantiate tokenizer
            tokenized_transcript = tokenizer.StringToIds(self.data[utt_id]["text"])  # Tokenize the transcript
            
            batch.append((utt_id, spliced_subsampled_features, tokenized_transcript))  # Append the processed data
            self.steps_in_epoch += 1
            
            # Check if we have finished an epoch
            if self.steps_in_epoch == len(self.utterances):
                self.epoch += 1
                self.steps_in_epoch = 0
                if self.shuffle:
                    np.random.shuffle(self.utterances)

        # Increment the total number of steps
        self.total_num_steps += 1
        return batch

