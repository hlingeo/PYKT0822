import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
from sklearn.decomposition import PCA

iris = datasets.load_iris()
pca = PCA(n_components=2)
data = pca.fit(iris.data).transform(iris.data)
print(data.shape)

dataMax = data.max(axis=0) + 1
dataMin = data.min(axis=0) - 1
n = 1000
X, Y = np.meshgrid(np.linspace(dataMin[0], dataMax[0], n),
                   np.linspace(dataMin[1], dataMax[1], n))
# C=1, 10
# kernel='linear' # 0.9667,9733
# kernel='rbf' # 0.96, 9667
# kernel='poly' # 0.9467, 0.96
# kernel='sigmoid' # 0.86, 0.82
# modified C will make different
svc = svm.SVC(C=10, kernel='sigmoid')
svc.fit(data, iris.target)
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
plt.contour(X, Y, Z.reshape(X.shape), colors='#000000')

for c, s in zip([0, 1, 2], ['o', '^', '*']):
    d = data[iris.target == c]
    plt.scatter(d[:, 0], d[:, 1], c='k', marker=s)

print(f"score={svc.score(data, iris.target):.4f}")
plt.show()