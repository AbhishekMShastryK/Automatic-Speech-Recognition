import json
import kaldiio
import numpy as np
import random
import string


def splice_and_subsample(x, context=0, sub_rate=1):
    """Splice and subsample a sequence.

    Args:
        x: numpy array of shape [n, d]
        context: int, the length of context in both left and right
        sub_rate: int, subsampling rate

    Returns:
        Output array of shape [(n+subrate-1)//sub_rate, (2*context+1)*d].
    """
    y = []

    # left context
    for i in range(context, 0, -1):
        y.append(np.concatenate([np.repeat(x[np.newaxis, 0, :], i, axis=0), x[:-i]], axis=0))

    y.append(x)

    # right context
    for i in range(1, context + 1):
        y.append(np.concatenate([x[i:], np.repeat(x[np.newaxis, -1, :], i, axis=0)], axis=0))

    return np.concatenate(y, axis=1)[::sub_rate]


class CharacterTokenizer:

    def __init__(self):
        alphabet = string.ascii_uppercase
        self.char_to_id = dict()
        self.id_to_char = dict()
        # id 0 is reserved for blank.
        for i, c in enumerate(alphabet):
            self.char_to_id[c] = i + 1
            self.id_to_char[i+1] = c
        # Apostrophy.
        self.char_to_id["'"] = 27
        self.id_to_char[27] = "'"
        # Space.
        self.char_to_id[" "] = 28
        self.id_to_char[28] = " "

    def IdsToString(self, ids):
        chars = map(lambda id: self.id_to_char[id], ids)
        return "".join(chars).strip()

    def StringToIds(self, s):
        return list(map(lambda char: self.char_to_id[char], s))

class InputGenerator:

    def __init__(self, input_json, batch_size=1, shuffle=False, context_length=0, subsampling_rate=1):
        with open(input_json, "rb") as f:
            self.input_dict = json.load(f)['utts']

        self.utterance_ids = list(self.input_dict.keys())
        self.num_utterances = len(self.utterance_ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_steps_per_epoch = self.num_utterances // self.batch_size
        self.num_elements_to_pad = 0
        remainder = self.num_utterances % batch_size
        if remainder > 0:
            self.num_steps_per_epoch += 1
            self.num_elements_to_pad = self.batch_size - remainder

        self.epoch = 0
        self.step_in_epoch = 0
        self.epoch_order = self.generate_epoch_order()
        self.tokenizer = CharacterTokenizer()

        self.context_length = context_length
        self.subsampling_rate = subsampling_rate

    def generate_epoch_order(self):
        epoch_order = list(range(self.num_utterances))
        if self.shuffle:
            random.shuffle(epoch_order)
        if self.num_elements_to_pad > 0:
            epoch_order = epoch_order + epoch_order[:self.num_elements_to_pad]
        return epoch_order

    def next(self):
        output = []
        for i in range(self.step_in_epoch * self.batch_size, (self.step_in_epoch+1) * self.batch_size):
            utterance_id = self.utterance_ids[self.epoch_order[i]]
            d = self.input_dict[utterance_id]
            x = splice_and_subsample(kaldiio.load_mat(d['feat']), self.context_length, self.subsampling_rate)
            y = self.tokenizer.StringToIds(d['text'])
            output.append((utterance_id, x, y))

        self.step_in_epoch += 1
        if self.step_in_epoch == self.num_steps_per_epoch:
            self.epoch += 1
            self.step_in_epoch = 0
            self.epoch_order = self.generate_epoch_order()

        return output

    @property
    def total_num_steps(self):
        return self.num_steps_per_epoch * self.epoch + self.step_in_epoch