from sklearn.ensemble import RandomForestRegressor
import pandas as pd

home_data_train = pd.read_csv('train.csv')
features = ['LotArea', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']
home_data_test = pd.read_csv('test.csv')
train_X = home_data_train[features]
print(train_X.head())

train_y = home_data_train.SalePrice
test_X = home_data_test[features]

print(test_X.describe())

test_X = test_X.fillna(3086, axis=0)
model = RandomForestRegressor()
model.fit(train_X, train_y)
y_pred = model.predict(test_X)
print(y_pred)

output = pd.DataFrame({'Id': home_data_test.Id,
                       'SalePrice': y_pred})
output.to_csv('predictions.csv', index=False)
