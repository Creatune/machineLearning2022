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

