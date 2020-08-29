from subprocess import check_call
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz

X = [[0, 0], [1, 1], [0, 1], [1, 0]]
Y = [0, 0, 1, 1]
colors = ['red', 'green']
marker = ['o', 'd']
index = 0
while index < len(X):
    type = Y[index]
    plt.scatter(X[index][0], X[index][1], c=colors[type],
                marker=marker[type])
    index += 1
plt.show()

classifier1 = tree.DecisionTreeClassifier()
classifier1.fit(X, Y)
print(classifier1.tree_)
# manual make a directory output
export_graphviz(classifier1, out_file='output/lab26.dot',
                filled=True, rounded=True,
                special_characters=True)
# -Tpng, -Tpdf
check_call(['dot', '-Tpng', 'output/lab26.dot',
            '-o', 'output/lab26.png'])
check_call(['dot', '-Tpdf', 'output/lab26.dot',
            '-o', 'output/lab26.pdf'])
check_call(['dot', '-Tsvg', 'output/lab26.dot',
            '-o', 'output/lab26.svg'])