import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# softmax activation function gives us the probability distribution for a layer's output
# softmax(x) = e^z/sigma(e^z)

# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        # self.output = probabilities

        output = probabilities
        return(output)

# softmax([1, 2, 3]) = [e^1/(e^(1+2+3), e^2/(e^(1+2+3), e^3/(e^(1+2+3)) = [0.09003057, 0.24472847, 0.66524096]
softmax = Activation_Softmax()
print(softmax.forward([[1,2,3]]))