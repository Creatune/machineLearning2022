from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('fivethirtyeight')

dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]


def k_nearest_neighbors(known_data, data_to_predict, k=3):
    if len(known_data) >= k:
        warnings.warn("K is set to a value less than total voting groups")

    distances = []
    for _class in known_data:
        for known_data_features in known_data[_class]:
            distance = np.linalg.norm(np.array(known_data_features) - np.array(data_to_predict))
            distances.append([distance, _class])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result


result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s=100, color=i)

plt.scatter(new_features[0], new_features[1], color=result)
plt.show()
