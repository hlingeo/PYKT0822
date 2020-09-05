import tensorflow as tf
import keras
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.applications.densenet import layers

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) \
    = fashion_mnist.load_data()
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(12, 9))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
print(train_images.shape)
print(test_images.shape)
print(train_labels.shape)
print(test_labels.shape)

model = keras.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
print(model.summary())
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=30)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)

predictions = model.predict(test_images)
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    prediction_label = np.argmax(predictions_array)
    if prediction_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{}{:2.0f}({})".format(class_names[prediction_label],
                                      100 * np.max(predictions_array),
                                      class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.xticks(range(10))
    plt.yticks([])
    thisPlot = plt.bar(range(10), predictions_array, color="#888888")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisPlot[predicted_label].set_color('red')
    thisPlot[true_label].set_color('blue')


plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    pass
plt.show()