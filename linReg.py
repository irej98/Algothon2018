# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 02:26:12 2018

@author: Jeri
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 23:13:59 2018

@author: Jeri
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

import quandl
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import sklearn.model_selection
import datetime, time



# GET DATA
quandl.ApiConfig.api_key = 'GsxZ6q5_MbB8DiGZmE48' #setting up your api-key

data = quandl.get('PERTH/USD_JPY_D')

model = Sequential()
model.add(Dense(1, activation='linear', input_dim=1))
#model.add(Dense(units=200, activation='relu'))
#model.add(Dense(units=1, activation='relu'))
model.compile(loss='mse', 
              optimizer='sgd', 
              metrics=['accuracy'])

#######################################################################################


data_numbers = data.select_dtypes(include=['int', 'int64', 'float64', 'float'])
print(data.shape)
#print(fbd)





#####################
print('Data')

data_numbers = data_numbers.fillna(0)
data = data.reset_index()

# Bid average
Y = np.array(data['Bid Average'][-100:-51])
Y = Y / 100

# Date

X = np.array(range(49))
plt.rcParams['figure.figsize'] = (15,15)
plt.scatter(X, Y, label = 'Original Data', s = 3 ** 2)
plt.legend()

X = X / 100



# Split dataset for training and initial testing
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# Check shape of each variable
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print('-----------------------------------------------')





print('X = ', X)
print('Y = ', Y)
model.fit(X_train, Y_train, epochs=300, batch_size=1)

print(X_test)
Y_prediction = model.predict(np.array(X_test))
print(Y_prediction)

##Y_prediction *= 100
plt.figure()
plt.rcParams['figure.figsize'] = (15,15)
plt.scatter(Y_test, Y_prediction, label='Test vs Prediction', s = 3 ** 2)
plt.legend()


X = X * 100
#X = X.astype(np.datetime64)

X_test = X_test * 100
#X_test = np.array(X_test.astype(np.datetime64))


plt.figure()
plt.rcParams['figure.figsize'] = (15,15)
plt.scatter(X_test, Y_prediction, label='Prediction', s = 3 ** 2)
plt.scatter(X_test, Y_test, label='Test vs Test', s = 3 ** 2)



X_train = X_train * 100


plt.scatter(X_train, Y_train, label='Original Data', s = 3 ** 2)
plt.legend()


###########################################
#### MAKE PREDICTION

futurePredictX = np.array(range(1,100)) / 100
futurePredictY = model.predict(futurePredictX)
### decent



X = np.array(range(49))
Y = np.array(data['Bid Average'][-100:-1])
Y = Y / 100
plt.figure()
plt.rcParams['figure.figsize'] = (15,15)
plt.scatter((futurePredictX*100), futurePredictY, s = 3 ** 2, label = '50 days learned + 50 days predicted')
plt.scatter((futurePredictX*100), Y, s = 3 ** 2, label = 'Actual')
plt.legend()


###########################################


####### NEW METHOD WHERE LAST 25 DAYS IS RELEARNED FOR EACH VALUE

#### Plot actual data:
plt.figure()
plt.rcParams['figure.figsize'] = (15,15)
Y = np.array(data['Bid Average'][-100:-1])
Y = Y / 100
futurePredictX = np.array(range(1,100)) / 100
plt.plot((futurePredictX*100), Y, label = 'Actual')# s = 3 ** 2, 
plt.legend()



#### For each additional point, retrain model



for i in range(25,100): #1,50
    print(i)
    # Bid average
    Y = np.array(data['Bid Average'][(-150+i):(-100+i)])#-75
    Y = Y / 100
    
    # Date
    
    X = np.array(range(50))#24#25
    
    #plt.scatter(X, Y, label = 'Original Data')
    #plt.legend()
    
    X = X / 100
    
    
    
    # Split dataset for training and initial testing
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    
    
    #print('X = ', X)
    #print('Y = ', Y)
    model.fit(X_train, Y_train, epochs=10, batch_size=1, verbose=2)
    
    
    X = X * 100
    #X = X.astype(np.datetime64)
    
    X_test = X_test * 100
    #X_test = np.array(X_test.astype(np.datetime64))
    
    
    X_train = X_train * 100
    
    ###########################################
    #### MAKE PREDICTION
    
    futurePredictX = np.array([i]) / 100 #(51+i)
    futurePredictY = model.predict(futurePredictX)
    ### decent
    
    
    '''
    X = np.array(range(49))
    Y = np.array(data['Bid Average'][-100:-1])
    Y = Y / 100
    '''
    #plt.figure()
    plt.scatter((futurePredictX*100), futurePredictY, s = 3 ** 2)#, label = '25 days learned + 1 days predicted')
    #plt.scatter((futurePredictX*100), Y, s = 3 ** 2, label = 'Actual')
    plt.legend()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    