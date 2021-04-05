import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

X=[1,2,3,4,5,6,7,8,9]
y=[11,22,33,44,53,66,77,87,95]

model = Sequential()

model.add(Dense(1,input_dim=1, activation='linear'))

sgd = optimizers.SGD(lr=0.01)

model.compile(optimizer=sgd, loss='mse', metrics=['mse'])

model.fit(X, y, batch_size=1, epochs=300, shuffle=False)

import matplotlib.pyplot as plt 

plt.plot(X, model.predict(X), 'b', X, y, 'k.')

print(model.predict([9.5]))


