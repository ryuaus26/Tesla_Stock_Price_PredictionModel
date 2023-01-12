import csv
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

#Linear Regression Model

data = pd.read_csv("./Data/TSLA_Train_data.csv")

training_set = data.iloc[:,1:2].values

scaler = MinMaxScaler(feature_range=(-1,1))
scaled_data_set = scaler.fit_transform(training_set)

X_train = []
y_train = []

for i in range(60,1258): #trains from 1st day to 60th day. Model learns to predict
    X_train.append(scaled_data_set[i-60:i,0]) #selects only the values from 0th column of the data  
    y_train.append(scaled_data_set[i,0])

X_train  = np.array(X_train)
y_train = np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences=True,input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer='adam', loss ='mean_squared_error')
regressor.fit(X_train,y_train,epochs=100,batch_size=32)

dataset_test = pd.read_csv("./Data/TSLA_Train_data.csv")
actual_stock_price = dataset_test.iloc[:,1:2].values

dataset_total = pd.concat((data['Open'],data['Open']),axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test)-60].values

inputs = inputs.reshape(2,1)
inputs = scaler.transform(inputs)

X_test = []

for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plt.plot(actual_stock_price, color ='red',label = "Actual Tesla stock price")
plt.plot(predicted_stock_price,color = 'blue',label= "Predicted Tesla stock price")
plt.title("Tesla stock price prediction model")
plt.xlabel("time")
plt.ylabel("$ Price")
plt.legend()



