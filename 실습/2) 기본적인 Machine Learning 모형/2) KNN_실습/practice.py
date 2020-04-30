from sklearn import neighbors, datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

# 모델 구축
# K 값을 매개변수로 줌
cif = neighbors.KNeighborsClassifier(5)
cif.fit(X, y)

y_pred = cif.predict(X)
print(confusion_matrix(y, y_pred))

####################
# #k-fold cross-validation 을 이용한 K 값 찾기
# cross_val_score: cross validation 을 진행해서 score 가 높은 것으로 뽑아내겠다
# KNN 의 K 값 범위
k_range = range(1, 100)
k_scores = []

for k in k_range:
    knn = neighbors.KNeighborsClassifier(k)
    # fold 를 10개로(cv), 성능 검증에는 accuracy(10번 실행 중 정답 비율) 활용(scoring)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    # mean: fold 를 나눈대로 반복하기 때문. 평균을 낸다
    k_scores.append(scores.mean())

# K=45 근방에서 제일 좋은 성능을 낸다
plt.plot(k_range, k_scores)
plt.xlabel('value of K in KNN')
plt.ylabel('Cross-Validated accuracy')
plt.show()

####################
# #weight 를 준 KNN
# 거리가 가까울수록 가중치를 줌
n_neighbors = 40

# 분포(mesh grid)를 그릴때 최소단위
# 분포를 직교좌표계에 그리는데 최소 눈금단위가 0.2
h = .02

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# weight 를 안주거나(uniform) 거리대로 주거나(distance)
for weights in ['uniform', 'distance']:
    # 가중치 부여
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # mesh 내부의 점(xx.ravel()) 을 설명변수로 해서 예측
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()

# 새로운 데이터로 
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()
y[::5] += 1 * (0.5 - np.random.rand(8))

knn = neighbors.KNeighborsRegressor(n_neighbors)
y_ = knn.fit(X, y).predict(T)

n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, c='k', label='data')
    plt.plot(T, y_, c='g', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights))

plt.tight_layout()
plt.show()
