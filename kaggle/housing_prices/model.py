from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import pandas as pd

train_X = pd.read_csv('train.csv')
train_y = pd.read_csv('train.csv').SalePrice

val_X = pd.read_csv('test.csv')

model = RandomForestClassifier(random_state=1)
print('SalePrice' in val_X)
model.fit(train_X, train_y)

y_pred = model.predict(val_X)

# mean_squared_error(y_pred, val_y)