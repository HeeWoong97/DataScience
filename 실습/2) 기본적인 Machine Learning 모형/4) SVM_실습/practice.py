import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# C: svm 의 regularization parameter
C = 1
cif = svm.SVC(kernel='linear', C=C)
cif.fit(X, y)

# 2,3번째 범주에 대해서는 정확도가 떨어진다
y_pred = cif.predict(X)
print('normal SVC', confusion_matrix(y, y_pred), sep='\n', end='\n\n')

####################
# kernel SVM 접합 및 비교
# linearSVC: 에러^2 을 최소화
# SVC: 에러를 최소화
cif = svm.LinearSVC(C=C, max_iter=10000)
cif.fit(X, y)
y_pred = cif.predict(X)
print('LinearSVC', confusion_matrix(y, y_pred), sep='\n', end='\n\n')

# radial basis function
# = Gaussian Kernel
# 차원을 무한대로 늘려서 초평면을 찾음
# gamma: rbf kernel 에서 필요한 매개변수
# max_iter: 10000차원까지 늘려서 초평면을 찾는다
cif = svm.SVC(kernel='rbf', gamma=0.7, C=C, max_iter=10000)
cif.fit(X, y)
y_pred = cif.predict(X)
print('rbf', confusion_matrix(y, y_pred), sep='\n', end='\n\n')

# polynomial kernel
# degree: 몇차원 공간으로?
# max_iter: 만약 접합이 안된다면 수를 늘려가기
cif = svm.SVC(kernel='poly', degree=3, C=C, gamma='auto')
cif.fit(X, y)
y_pred = cif.predict(X)
print('polynomial', confusion_matrix(y, y_pred), sep='\n', end='\n\n')


####################
# 시각적 비교
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

C = 1.0  # Regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
models = (clf.fit(X, y) for clf in models)

titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

# polynomial 의 경우 나눠진 영역들이 자연스럽지가 않다
# accuracy 가 좋아도 나눠진 영역을 보고 재고해볼 필요가 있음
