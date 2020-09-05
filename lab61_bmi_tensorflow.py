import pandas as pd
import keras
from sklearn.preprocessing import LabelBinarizer
from keras import callbacks

csv = pd.read_csv("data/bmi.csv")
csv['height'] = csv['height'] / 200
csv['weight'] = csv['weight'] / 100
print(csv.head(n=20))

encoder = LabelBinarizer()
transformedLabel = encoder.fit_transform(csv['label'])
print(csv['label'][:20])
print(transformedLabel[:20])

TEST_START = 250000
test_csv = csv[TEST_START:]
test_pat = test_csv[['weight', 'height']]
test_ans = transformedLabel[TEST_START:]
train_csv = csv[:TEST_START]
train_pat = train_csv[["weight", 'height']]
train_ans = transformedLabel[:TEST_START]
print(test_pat.shape)
print(test_ans.shape)
print(train_pat.shape)
print(train_ans.shape)

Sequential = keras.models.Sequential
Dense = keras.layers.Dense
backend = keras.backend

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(2,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd',
              metrics=['accuracy'])
tensorboard = callbacks.TensorBoard(log_dir='logs', histogram_freq=1)
print(model.summary())
model.fit(train_pat, train_ans, batch_size=100, epochs=50,
          verbose=1, validation_data=(test_pat, test_ans),
          callbacks=[tensorboard])
score = model.evaluate(test_pat, test_ans, verbose=0)
print("score[0]={}, score[1]={}".format(score[0], score[1]))
