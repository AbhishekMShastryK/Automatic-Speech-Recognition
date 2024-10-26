import random

class MnistInputGenerator:

    def __init__(self, inputs, outputs, batch_size=1, shuffle=False):
        self.num_samples = inputs.shape[0]
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_steps_per_epoch = self.num_samples // self.batch_size
        self.num_elements_to_pad = 0
        remainder = self.num_samples % batch_size
        if remainder > 0:
            self.num_steps_per_epoch += 1
            self.num_elements_to_pad = self.batch_size - remainder

        self.epoch = 0
        self.step_in_epoch = 0
        self.epoch_order = self.generate_epoch_order()

    def generate_epoch_order(self):
        epoch_order = list(range(self.num_samples))
        if self.shuffle:
            random.shuffle(epoch_order)
        if self.num_elements_to_pad > 0:
            epoch_order = epoch_order + epoch_order[:self.num_elements_to_pad]
        return epoch_order

    def next(self):
        indices = self.epoch_order[self.step_in_epoch * self.batch_size : (self.step_in_epoch+1) * self.batch_size]
        # shape [batch_size, d]
        x = self.inputs[indices,:]
        # shape [batch_size]
        y = self.outputs[indices]

        self.step_in_epoch += 1
        if self.step_in_epoch == self.num_steps_per_epoch:
            self.epoch += 1
            self.step_in_epoch = 0
            self.epoch_order = self.generate_epoch_order()

        return x, y

    @property
    def total_num_steps(self):
        return self.num_steps_per_epoch * self.epoch + self.step_in_epoch