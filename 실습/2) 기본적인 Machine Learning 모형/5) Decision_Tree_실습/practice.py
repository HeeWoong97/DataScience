from sklearn import tree
from sklearn.datasets import load_iris
import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# ##########Decision Classification Tree##########
# ->y가 범주형일때

# iris.data: 데이터
# iris.feature_names: 각 변수의 의미
# iris.target: 종속 데이터
# iris.target_names: 종속변수의 의미
iris = load_iris()

# 트리 구축
# 구축 기준 기본값: 지니계수
cif = tree.DecisionTreeClassifier()
cif = cif.fit(iris.data, iris.target)

# 트리 시각화
# filled: 그래프의 색상
# rounded: 소숫점 보정
# special_characters: 특수문자 포함
dot_data = tree.export_graphviz(cif, out_file="tree.dot", feature_names=iris.feature_names,
                                class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)

# 엔트로피를 활용한 트리
# 엔트로피가 0이 될때까지 분화를 진행
cif2 = tree.DecisionTreeClassifier(criterion="entropy")
cif2.fit(iris.data, iris.target)

# 트리 시각화
dot_data2 = tree.export_graphviz(cif2, out_file="tree.dot", feature_names=iris.feature_names,
                                 class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
graph2 = graphviz.Source(dot_data)

# 프루닝
# ->모델을 그냥 만들어버리면(마지막 노드에 1 또는 0만 존재하도록) 너무 복잡해진다
# ->과접합 가능성이 높아지는데 이를 필요 없는 곳은 적당히 잘라주는게 프루닝
# max_depth: 노드의 단계를 제한. root node 밑으로 2단계만
# k-fold 교차검증 같은 방법을 사용해서 효과적인 depth 를 구할 수 있다
cif3 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
cif3.fit(iris.data, iris.target)

# 트리 시각화
dot_data3 = tree.export_graphviz(cif3, out_file="tree.dot", feature_names=iris.feature_names,
                                 class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
graph3 = graphviz.Source(dot_data)

# confusion matrix
# 프루닝을 이용한 모델의 예측력이 떨어진다
# 훈련 데이터에 대해 예측력이 떨어지기 때문에 검증 데이터에 대해서는 다를수도 있음
print('지니계수 이용', confusion_matrix(iris.target, cif.predict(iris.data)), sep='\n', end='\n\n')
print('엔트로피 이용', confusion_matrix(iris.target, cif2.predict(iris.data)), sep='\n', end='\n\n')
print('프루닝 이용', confusion_matrix(iris.target, cif3.predict(iris.data)), sep='\n', end='\n\n')

# 학습, 검증 데이터 구분
# 기본세팅: 0.75(학습), 0.25(검증)
# 위에서 학습 데이터에 대해서만 테스트 했을 경우에는 틀리는 경우가 없었지만
# 학습, 검증으로 나누어 테스트 해보니 예측이 틀리는 경우도 있다
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, stratify=iris.target, random_state=1)

cif4 = tree.DecisionTreeClassifier(criterion='entropy')
cif4.fit(X_train, y_train)
print('학습, 검증 분류', confusion_matrix(y_test, cif4.predict(X_test)), sep='\n', end='\n\n')

# ##########Decision Regression Tree##########
# ->y 가 연속형일때
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# 트리 구축
regr1 = tree.DecisionTreeRegressor(max_depth=2)
regr2 = tree.DecisionTreeRegressor(max_depth=5)

regr1.fit(X, y)
regr2.fit(X, y)

X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

y_1 = regr1.predict(X_test)
y_2 = regr2.predict(X_test)

# depth 시각화
# depth = 2인 간단한 모델: 큰 경향을 반영
# depth = 5인 복잡한 모델: 이상치에 민감함. 과접합
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

# 트리 시각화
dot_data4 = tree.export_graphviz(regr2, out_file=None, filled=True, rounded=True, special_characters=True)
graph4 = graphviz.Source(dot_data4)

dot_data5 = tree.export_graphviz(regr1, out_file=None, filled=True, rounded=True, special_characters=True)
graph5 = graphviz.Source(dot_data5)
