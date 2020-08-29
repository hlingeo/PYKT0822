import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import os

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

model.fit(inputList, resultList, epochs=200, batch_size=20,
          validation_split=0.2)
scores = model.evaluate(inputList, resultList)
print(model.metrics_names)
print(scores)
print("{}:{}".format(model.metrics_names[0], scores[0]))
print("{}:{}".format(model.metrics_names[1], scores[1]))

demo45
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import os
from keras.models import save_model, load_model
MODEL_LOC = 'models/lab45'
dataset1 = np.loadtxt("data/diabetes.csv",
                      delimiter=",",
                      skiprows=1)
print(dataset1.shape)
inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]


def createModel():
    # global model
    model = Sequential()
    model.add(Dense(14, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


model = createModel()

model.fit(inputList, resultList, epochs=200, batch_size=20)
save_model(model, MODEL_LOC)
scores = model.evaluate(inputList, resultList)
print(model.metrics_names)
print(scores)
print("{}:{}".format(model.metrics_names[0], scores[0]))
print("{}:{}".format(model.metrics_names[1], scores[1]))

model2 = createModel()
scores2 = model2.evaluate(inputList, resultList)
print("{}:{}".format(model2.metrics_names[0], scores2[0]))
print("{}:{}".format(model2.metrics_names[1], scores2[1]))

model3 = load_model(MODEL_LOC)
scores3 = model3.evaluate(inputList, resultList)
print("{}:{}".format(model3.metrics_names[0], scores3[0]))
print("{}:{}".format(model3.metrics_names[1], scores3[1]))
