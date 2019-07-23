# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("all_2009-8.csv")
X = dataset.iloc[  :  , [2 , 4, 5 , 6]].values
y = dataset.iloc[ : , -1].values

"""
X = dataset[:, :-1]
y = dataset[:, -1]
"""


# Splitting the dataset into the Training set and Test set
"""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.08, random_state = 0)"""

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
"""
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""



# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(32,  activation = 'relu', input_dim = 4))

# Adding the second hidden layer
model.add(Dense(units = 2, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 4, activation = 'relu'))
model.add(Dense(units = 6, activation = 'relu'))
model.add(Dense(units = 8, activation = 'relu'))
model.add(Dense(units = 10, activation = 'relu'))
model.add(Dense(units = 12, activation = 'relu'))
model.add(Dense(units = 14, activation = 'relu'))
model.add(Dense(units = 16, activation = 'relu'))
model.add(Dense(units = 18, activation = 'relu'))
model.add(Dense(units = 20, activation = 'relu'))
model.add(Dense(units = 22, activation = 'relu'))
model.add(Dense(units = 24, activation = 'relu'))
model.add(Dense(units = 26, activation = 'relu'))
model.add(Dense(units = 28, activation = 'relu'))
model.add(Dense(units = 30, activation = 'relu'))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 34, activation = 'relu'))
model.add(Dense(units = 36, activation = 'relu'))
model.add(Dense(units = 38, activation = 'relu'))
model.add(Dense(units = 40, activation = 'relu'))
model.add(Dense(units = 42, activation = 'relu'))
model.add(Dense(units = 44, activation = 'relu'))

# Adding the output layer

model.add(Dense(units = 1))

#model.add(Dense(1))
# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
"""model.fit(X_train, y_train, batch_size = 10, epochs = 100)"""
model.fit(X, y, batch_size = 1, epochs = 100)

lower = 0
upper = 30   ##  changed this value

b = pd.read_csv("2018.csv")### changed this value
X_test = b.iloc[lower : upper  , [1 ,4, 6, 3]].values
X_test = sc.transform(X_test)

y_test = b.iloc[lower : upper , -5].values   
y_pred = model.predict(X_test)



from sklearn import metrics  
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
print('RMSE current:',np.sqrt(metrics.mean_squared_error(y_test,b.iloc[lower : upper, -3]))) 
 
"""
df=pd.DataFrame({'x': range(len(y_pred)), 'y1': y_pred, 'y2': b.iloc[lower : upper , -3], 'y3': y_test })
plt.grid(True) 

plt.plot( 'x', 'y1', data=df, color='red', linewidth=3, label = "Random forest")
plt.plot( 'x', 'y2', data=df, color='blue', linewidth=3, label = "Current Method")
plt.plot( 'x', 'y3', data=df, color='gold', linewidth=3, label = "Actual Runs")


# x-axis label 
plt.xlabel('x - axis (Over range)') 
# frequency label 
plt.ylabel('y - axis (Runs)') 
# plot title 
plt.title('Comparison Graph') 

#to print which line represent which values
plt.legend()
"""












"""
plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()
"""