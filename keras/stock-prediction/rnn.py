# Step 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
training_set = pd.read_csv('dataset/Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting the inputs and the ouputs
X_train = training_set[:-1]
y_train = training_set[1:]

# Reshaping
X_train = np.reshape(X_train, (len(X_train), 1, 1))

# Step 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Initialising the RNN
model = Sequential()

# Adding the input layer and the LSTM layer
model.add(LSTM(4, activation='sigmoid', input_shape=(None, 1)))

# Adding the output layer
model.add(Dense(1))

# Compiling the RNN
model.compile(optimizer='adam', loss='mse')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, batch_size=32, nb_epoch=200)

# Step 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
test_set = pd.read_csv('dataset/Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

# Getting the predicted stock price of 2017
inputs = real_stock_price
inputs = sc.fit_transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predictions = model.predict(inputs)
predicted_stock_price = sc.inverse_transform(predictions)

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
