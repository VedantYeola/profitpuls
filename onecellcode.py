import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl

#Implementation of KNN classifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#implementation of knn regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import neighbors

import requests
import json

data = quandl.get("NSE/TATASTEEL")
data.head(10)
plt.figure(figure=(16,8))
plt.plot(data['Close'], label='Closing Price')
#classification problem:buy(+1) or sell(-1) the stock
data['Open-Close'] = data['Open']-data['Close']
data['High-Low'] = data['High']-data['Low']
data = data.dropna()
#input features to predict weather customer shuld buy or sell the stock
X = data[['Open-Close', 'High-Low']]
X.head()
#intention to store +1 for the buy  signal and -1 for the sell signal. the target variable is 'y' for classification task
Y = np.where(data['Close'].shift(-1)>data['Close'],1,-1)
Y
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 44)
#using gridsearch to find the best parameter

params = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn = neighbors.KNeighborsClassifier()
model = GridSearchCV(knn, params, cv=5)

#fit the model

model.fit(X_train, y_train)

#accurancy score

accuracy_train = accuracy_score(y_train, model.predict(X_train))
accuracy_test = accuracy_score(y_test, model.predict(X_test))

print('Train_data accuracy: %.2f' %accuracy_train)
print('Test_data accuracy: %.2f' %accuracy_test)
predictions_classification = model.predict(X_test)
actual_predicted_data = pd.DataFrame({'Actual Class': y_test, 'Predicted Class': predictions_classification})
actual_predicted_data.head(10)
#regression problem = Knn
y = data['Close']
y
#implementation of knn regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.25, random_state = 44)
#using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn_reg = neighbors.KNeighborsRegressor()
model_reg = GridSearchCV(knn_reg, params, cv=5)
#fit the model and make preddictions
model_reg.fit(X_train_reg, y_train_reg)
predictions = model_reg.predict(X_test_reg)
print(predictions)
#rmse
rms = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(predictions)),2)))
rms
valid = pd.DataFrame({'Actual Class': y_test, 'Predicted Close value': predictions})
valid.head(10)
