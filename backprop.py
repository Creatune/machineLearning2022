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