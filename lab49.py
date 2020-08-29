import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import os
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import \
    StratifiedKFold, cross_val_score, GridSearchCV

dataset1 = np.loadtxt("data/diabetes.csv",
                      delimiter=",",
                      skiprows=1)
print(dataset1.shape)
inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]


def createModel(optimizer='adam', init='uniform'):
    # global model
    model = Sequential()
    model.add(Dense(14, kernel_initializer=init,
                    input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    return model


model2 = KerasClassifier(build_fn=createModel, verbose=0)
optimizers = ['adam', 'rmsprop', 'sgd']
inits = ['normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 15]
param_grid = dict(optimizer=optimizers,
                  epochs=epochs,
                  batch_size=batches,
                  init=inits)
grid = GridSearchCV(estimator=model2, param_grid=param_grid)
grid_result = grid.fit(inputList, resultList)
