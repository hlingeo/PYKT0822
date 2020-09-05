import numpy as np
from keras import models, layers
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = \
    imdb.load_data(num_words=10000)
print(train_data[0])
print(max([max(sequence) for sequence in train_data]))
word_to_digit_index = imdb.get_word_index()
print(type(word_to_digit_index))
# for index in word_to_digit_index:
#     print(index, word_to_digit_index[index])
reverse_index = dict([(v, k) for k, v
                      in word_to_digit_index.items()])


def decodeIMDB(x):
    return ' '.join([reverse_index.get(i - 3, '?')
                     for i in train_data[x]])


for i in range(5):   # 1 is good and 0 is bad
    print(train_labels[i])
    print(decodeIMDB(i))


import numpy as np
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = \
    imdb.load_data(num_words=10000)
print(train_data[0])
print(max([max(sequence) for sequence in train_data]))
word_to_digit_index = imdb.get_word_index()
print(type(word_to_digit_index))
# for index in word_to_digit_index:
#     print(index, word_to_digit_index[index])
reverse_index = dict([(v, k) for k, v
                      in word_to_digit_index.items()])


def decodeIMDB(x):
    return ' '.join([reverse_index.get(i - 3, '?')
                     for i in train_data[x]])


for i in range(5):
    print(train_labels[i])
    print(decodeIMDB(i))

# 05-Sep-2020
def vectorize_sequence(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1
    return results

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(32, activation='relu',
                       input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train,
                    epochs=30,
                    batch_size=500,
                    validation_data=(x_test, y_test))
import matplotlib.pyplot as plt
history_dict = history.history