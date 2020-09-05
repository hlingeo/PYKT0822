from keras import models
from keras.datasets import boston_housing
from keras import layers

(train_data, train_target), (test_data, test_target) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
print(train_data.shape, test_data.shape)
test_data -= mean
test_data /= std

def build_model():
    m = models.Sequential()
    m.add(layers.Dense(64, activation='relu',
                       input_shape=(train_data.shape[1],)))
    m.add(layers.Dense(64, activation='relu'))
    m.add(layers.Dense(1))
    return m


model = build_model()
model.compile(optimizer='adam', loss="mse", metrics=['mae'])
model.fit(train_data, train_target, validation_split=0.1,
          epochs=100, batch_size=5, verbose=1)

for (i, j) in zip(test_data, test_target):
    predict = model.predict(i.reshape(1, -1))
    print("predict as:{:.1f}, real is:{}, diff={:.1f}".format(predict[0][0], j, predict[0][0] - j))

