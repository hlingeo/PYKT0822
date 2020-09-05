import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()


def plot_image(image, label):
    plt.imshow(image.reshape(28, 28), cmap='binary')
    plt.title(label)
    plt.show()


#for i in range(0, 10):
 #   label = f"the image marked as:{train_labels[i]}"
  #  plot_image(train_images[i], label)

FLATTEN_DIM = 28 * 28
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

trainImages = np.reshape(train_images, (TRAINING_SIZE, FLATTEN_DIM))
testImages = np.reshape(test_images, (TEST_SIZE, FLATTEN_DIM))
print(type(trainImages[0]))

trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)

trainImages /= 255
testImages /= 255

NUM_DIGITS = 10
print(train_labels[:10])
trainLabels = keras.utils.to_categorical(train_labels, NUM_DIGITS)
testLabels = keras.utils.to_categorical(test_labels, NUM_DIGITS)

model = Sequential()
model.add(Dense(128, activation=tf.nn.relu,
                input_shape=(FLATTEN_DIM,)))
model.add(Dense(10, activation=tf.nn.softmax))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print(model.summary())
from keras.callbacks import TensorBoard
tbCallbacks = TensorBoard(log_dir='logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
model.fit(trainImages, trainLabels, epochs=20,
          callbacks=[tbCallbacks])

predictLabels = model.predict_classes(testImages)
print("result=", predictLabels[:10])

loss, accuracy = model.evaluate(testImages, testLabels)
print("loss={}, accuracy={}".format(loss, accuracy))