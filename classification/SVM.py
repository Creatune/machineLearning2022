import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', 10, inplace=True)
df.drop(['id'], 1, inplace=True)
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)
print(f'accuracy = {clf.score(X_test, y_test) * 100}%')

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]])
prediction = clf.predict(example_measures)
if prediction == [2]:
    print(f"benign tumor class={prediction}")
else:
    print(f"malignant tumor class={prediction}")