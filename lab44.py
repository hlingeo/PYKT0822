import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import os
import matplotlib.pyplot as plt

dataset1 = np.loadtxt("data/diabetes.csv",
                      delimiter=",",
                      skiprows=1)
print(dataset1.shape)
inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]

model = Sequential()
model.add(Dense(14, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.summary()

history = model.fit(inputList, resultList, epochs=200, batch_size=20,
                    validation_split=0.1)
scores = model.evaluate(inputList, resultList)
print(model.metrics_names)
print(scores)
print("{}:{}".format(model.metrics_names[0], scores[0]))
print("{}:{}".format(model.metrics_names[1], scores[1]))

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(['val_loss','loss'])
plt.show()

plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.legend(['val_accuracy','accuracy'])
plt.show()