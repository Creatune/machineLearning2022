x = [1.0, -2.0, 3.0]  # input values
w = [-3.0, -1.0, 2.0]  # weight values
b = 1.0  # bias

xw = []
for i in range(len(x)):
    xw.append(x[i] * w[i])

print(sum(xw) + b)

z = sum(xw) + b

# RelU function:
y = max(z, 0)

print(y)

# the above is a full forward pass of ONE neuron with a relu activation function

# example derivate from the next layer is 1
dvalue = 1.0

# derivative of relu and the chain rule
drelu_dz = dvalue * (1. if z > 0 else 0.)

print(drelu_dz)

# going further back, taking derivatives of the sum function (input1 * weight1 + bias1) + (input2 * weight2 + bias2)+...
dsum_dxw0 = 1  # derivative is 1 because do/do,x(x + y) = 1 + 0 = 1; where do/do,x represents partial derivative wrt x

drelu_dxw0 = drelu_dz * dsum_dxw0  # chain rule

print(drelu_dxw0)

# repeating the above 3 lines for the remaining set of weights/inputs and the bias

dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

# going further back, we have to differentiate the neuron output function = f(x, w, b) = x * w + b
dmul_dx0 = w[0]  # do/do,x0 (x0 * w0 + b0) = w0
drelu_dx0 = drelu_dxw0 * dmul_dx0  # chain rule
print(drelu_dx0)

# We perform the same operation for other inputs and weights
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2
print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

# the 3 immediate lines below are the gradients found by backpropagating using chain rule
dx = [drelu_dx0, drelu_dx1, drelu_dx2]  # gradients on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2]  # gradients on weights
db = drelu_db  # gradient on bias...just 1 bias here.

# changing the weights and biases using a negative fraction of the gradient to show that we can decrease the neuron
# output using backprop
print(w, b)
w[0] += -0.0001 * dw[0]  # an optimizer usually takes care of adjusting the weights and biases, but for now we just
# did a negative fraction of the gradient arbitrarily
w[0] += -0.01 * dw[0]
w[0] += -0.01 * dw[0]
b += -0.01 * db
print(w, b)
# performing a forward pass again to prove that we actually decreased the neuron output:
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
z = xw0 + xw1 + xw2 + b
y = max(z, 0)
print(y)  # this was previously 6, but now 5.969, which means that we successfully decreased the neuron's output

# NOTE: decreasing the neuron's output does not usually make sense but this was done for simplicity. we usually want
# to decrease how wrong our model is, which means we would have to take the derivative of the loss function

# ALL of the above has to be applied to each and every neuron in a neural network

# backprop for a set of neurons:
import numpy as np

# Passed in gradient from the next layer
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# sum weights of given input and multiply by the passed in gradient for this neuron
dinputs = np.dot(dvalues, weights.T)

print(dinputs)  # dinputs is a gradient of the neuron function with respect to inputs

# We have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])
# sum weights of given input
# and multiply by the passed in gradient for this neuron
dweights = np.dot(inputs.T, dvalues)

print(dweights)

# One bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list -
# we explained this in the chapter 4
dbiases = np.sum(dvalues, axis=0, keepdims=True)
print(dbiases)

# Example layer output
z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])
# ReLU activation's derivative
drelu = np.zeros_like(z)
drelu[z > 0] = 1

print(drelu)

# The chain rule
dvalues = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])
drelu *= dvalues
print(drelu)

# doing a full backprop for a set of neurons:


# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])
# We have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])
# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T
# One bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# forward pass:
layer_outputs = np.dot(inputs, weights) + biases

relu_outputs = np.maximum(layer_outputs, 0)

# Let's optimize and test backpropagation here
# ReLU activation - simulates derivative with respect to input values
# from next layer passed to current layer during backpropagation
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0

# Dense layer
# dinputs - multiply by weights
dinputs = np.dot(drelu, weights.T)
# dweights - multiply by inputs
dweights = np.dot(inputs.T, drelu)
# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list -
# we explained this in the chapter 4
dbiases = np.sum(drelu, axis=0, keepdims=True)

# Update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases
print(weights)
print(biases)