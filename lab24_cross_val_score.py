import sklearn.datasets as datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import svm

iris = datasets.load_iris()
data = iris.data
target = iris.target

logisticRegression1 = LogisticRegression()
svc1 = svm.SVC()

estimators = [logisticRegression1, svc1]
for e in estimators:
    scores = model_selection.cross_val_score(e,
                                             data,
                                             target,
                                             cv=5)
    print(e)
    print(scores)
    print(scores.mean(), scores.std())
