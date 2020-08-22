from sklearn import linear_model, datasets
import numpy as np

diabetes = datasets.load_diabetes()
print(type(diabetes))
print(dir(diabetes))
print(diabetes.DESCR)
print(diabetes.data.shape) # feature
print(diabetes.target.shape) # label
dataForTest = -50
data_train = diabetes.data[:dataForTest] #x
target_train = diabetes.target[:dataForTest] #y
print(f"features dim={data_train.shape}")
print(f"label dim={target_train.shape}")
data_test = diabetes.data[dataForTest:]
target_test = diabetes.target[dataForTest:]
print(f"testing feature dim={data_test.shape}")
print(f"testing label dim={target_test.shape}")

regression1 = linear_model.LinearRegression(normalize=True)
regression1.fit(data_train, target_train)
print(regression1.coef_)
print(regression1.intercept_)

for i, v in enumerate(regression1.coef_):
    print('feature:%0d, score:%.5f' % (i, v))

print("score=%.2f" % regression1.score(data_test, target_test))

for i in range(dataForTest, 0):
    dataArray = np.array(data_test[i]).reshape(1, -1)
    print("predict={:.1f},actual={}".format(
        regression1.predict(dataArray)[0],
        target_test[i]
    ))
mean_square_error = np.mean((regression1.predict(data_test) - target_test) ** 2)
mean_absolute_error = mean_square_error ** 0.5
print(mean_square_error)
print(mean_absolute_error)
