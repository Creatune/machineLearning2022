import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')


def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


xs, ys = create_dataset(40, 10, 2, correlation='pos')


def best_fit_slope(xs, ys):
    m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) ** 2) - mean(xs ** 2))
    return m


def best_fit_intercept(xs, ys):
    b = mean(ys) - best_fit_slope(xs, ys) * mean(xs)
    return b


def squared_error(ys_og, ys_line):
    #  distance between line of best fit and actual point squared
    return sum((ys_line - ys_og) ** 2)


def coefficient_determination(ys_og, ys_line):
    y_mean_line = [mean(ys_og) for y in ys_og]
    squared_error_reggr = squared_error(ys_og, ys_line)
    squared_error_y_mean = squared_error(ys_og, y_mean_line)
    return 1 - (squared_error_reggr / squared_error_y_mean)


m = best_fit_slope(xs, ys)
print(m)
b = best_fit_intercept(xs, ys)
print(b)

predict_x = 8
predict_y = m * predict_x + b
print(predict_y)

regression_line = [(m * x) + b for x in xs]
r_squared = coefficient_determination(ys, regression_line)
print(r_squared)
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y)
plt.plot(xs, regression_line)
plt.show()
