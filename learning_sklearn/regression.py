import pandas as pd
import quandl
import math
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import numpy as np

# regression is basic therefore not all features are correlated unlike deep learning where everything matters

df = quandl.get('WIKI/GOOGL')  # gives google stock price in a table

print(df.head())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]  # picking columns that we need

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0  # high minus low percent

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0  # daily percentage change

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]  # defining a new dataframe overwriting the old one to
# view later

forecast_col = 'Adj. Close'  # our forecasted price

df.fillna(value=-99999, inplace=True)

forecast_out = int(math.ceil(0.1 * len(df)))  # here we're taking the last 10 days data "0.1" to predict the price
# for next day

df['label'] = df[forecast_col].shift(-forecast_out)  # SHIFT is a Pandas function that shift the specific DF column
# either up(-) or down(+)

# So all he has done is created a column in the Pandas.DF (Pandas.DataFrame) and labeled it "Label" and in it displayed
# the shifted values of Adjusted Close UP(-) 34 ROWS. In this working example each ROW is equivalent to ONE DAY.

print(df.head())

X = np.array(df.drop(['label'], 1))  # It is a typical standard with machine learning in code to define X (capital x),
# as the features drops label column which is represented by 1 and everything else are features


X = preprocessing.scale(X)
# We could leave it at this, and move on to training and testing, but we're going to do some pre-processing.
# Generally, you want your features in machine learning to be in a range of -1 to 1. This may do nothing,
# but it usually speeds up processing and can also help with accuracy. Because this range is so popularly used,
# it is included in the preprocessing module of Scikit-Learn. To utilize this, you can apply preprocessing.scale to
# your X variable:

X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])  # y (lowercase y) as the label that corresponds to the features

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)  # this is a python feature
# to continously assign variables new incoming values
# made by the train_test_split func  ## called multiple assignment


clf = LinearRegression()

clf = LinearRegression(n_jobs=-1)  # n_jons uses as many threads as possible when its -1

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)

# SVM doesn't do better than linear regression

for k in ['linear', 'poly', 'rbf', 'sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k, confidence)
