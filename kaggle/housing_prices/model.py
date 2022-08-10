from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

home_data_train = pd.read_csv('train.csv')
features = ['LotArea', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']
home_data_test = pd.read_csv('test.csv')
train_X = home_data_train[features]
print(train_X.head())
train_y = home_data_train.SalePrice
test_X = home_data_test[features]
test_X = test_X.dropna(axis=0)
model = RandomForestClassifier()
model.fit(train_X, train_y)
y_pred = model.predict(test_X)
print(y_pred)

IDs = np.array([i for i in range(1461, 1461 * 2 - 3)])
res = pd.DataFrame(y_pred)
res.index += 1461
res.columns = ['SalePrice']
res.to_csv('predictions.csv')