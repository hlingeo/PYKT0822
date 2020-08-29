import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset1 = np.loadtxt("data/diabetes.csv",
                      delimiter=",",
                      skiprows=1)
print(dataset1.shape)
inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]

feature_train, feature_test, label_train, label_test \
    = train_test_split(inputList, resultList, test_size=0.2, stratify=resultList)

for data in [resultList, label_train, label_test]:
    classes, counts = np.unique(data, return_counts=True)
    for cl, co in zip(classes, counts):
        print(f"{int(cl)}==>{co / sum(counts)}")

model = Sequential()
model.add(Dense(14, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.summary()

history = model.fit(feature_train, label_train, epochs=200, batch_size=20,
                    validation_data=(feature_test, label_test))

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(['val_loss', 'loss'])
plt.show()

plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.legend(['val_accuracy', 'accuracy'])
plt.show()
