from pandas import read_csv
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

df1 = read_csv("./data/iris.data", header=None)
dataset = df1.values
features = dataset[:, 0:4].astype(float)
labels = dataset[:, 4]
print(features.shape, labels.shape)
print(np.unique(labels, return_counts=True))
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
print(np.unique(encoded_Y, return_counts=True))
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y[:10], dummy_y[50:60], dummy_y[100:110])


def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model,
                            epochs=200,
                            batch_size=10,
                            verbose=1)
kfold = KFold(n_splits=3, shuffle=True)
results = cross_val_score(estimator, features,
                          dummy_y, cv=kfold)
print(results)