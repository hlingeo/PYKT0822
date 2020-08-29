import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import os
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

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


model2 = KerasClassifier(build_fn=createModel,
                         epochs=200, batch_size=20, verbose=0)
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
result = cross_val_score(model2, inputList, resultList, cv=fiveFold)
print(result)
print(result.mean(), result.std())