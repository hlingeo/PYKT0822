from sklearn import datasets

data1 = datasets.make_regression(10, 6, noise=5)
print(data1[0].shape)
print(data1[1].shape)
regressionX = data1[0]
r1 = sorted(regressionX, key=lambda t: t[0])
r2 = sorted(regressionX, key=lambda t: t[1])
r3 = sorted(regressionX, key=lambda t: t[2])
r4 = sorted(regressionX, key=lambda t: t[3])
r5 = sorted(regressionX, key=lambda t: t[4])
r6 = sorted(regressionX, key=lambda t: t[5])
print("finished")