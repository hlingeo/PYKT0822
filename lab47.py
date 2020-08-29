import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

dataset1 = np.loadtxt("data/diabetes.csv",
                      delimiter=",",
                      skiprows=1)
print(dataset1.shape)
inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]

fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
totalScore = []

for train, test in fiveFold.split(inputList, resultList):
    model = Sequential()
    model.add(Dense(14, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(inputList[train], resultList[train],
                        epochs=200, batch_size=20,
                        validation_split=0.1,verbose=0)
    scores = model.evaluate(inputList[test], resultList[test])
    print(model.metrics_names)
    print(scores)
    print("{}:{}".format(model.metrics_names[0], scores[0]))
    print("{}:{}".format(model.metrics_names[1], scores[1]))
    totalScore.append(scores[1] * 100)

print(totalScore)
print(np.mean(totalScore), np.std(totalScore))