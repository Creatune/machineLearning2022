# The loss function, also referred to as the cost function, is the
# algorithm that quantifies how wrong a model is. Loss is the measure of this metric. Since loss is
# the model’s error, we ideally want it to be 0.

import math
import numpy as np

softmax_output = [0.7, 0.1, 0.2]

target_output = [1, 0, 0]  # one-hot vector (there is a 1 only in one "hot" spot, where the word hot indicates the
# desired output)

loss = -(math.log(softmax_output[0]) * target_output[0] + math.log(softmax_output[1]) * target_output[1] + math.log(
    softmax_output[2]) * target_output[2])  # categorical cross entropy loss function

print(loss)

loss = -math.log(softmax_output[0] * target_output[0])

print(loss)

loss = -math.log(softmax_output[0])

print(loss)

#  if confidence (softmax_output[i]) is higher, loss is lower and vice versa

softmax_outputs = np.array([[0.7, 0.2, 0.1],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

neg_log = -np.log(softmax_outputs[range(len(softmax_output)), class_targets])

average_loss = np.mean(neg_log)

print(neg_log, average_loss)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        # overall loss is always the mean of the losses of all samples
        data_loss = np.mean(sample_losses)

        return data_loss


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return (-np.log(correct_confidences))

#  resume from page 127 after thoroughly revising the above code
