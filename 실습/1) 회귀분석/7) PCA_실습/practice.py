# Principal Component Analysis 실습
# 머신러닝을 모듈에 포함하고, 이에 대한 정보가 있는 사이트: https://scikit-learn.org
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# #데이터 전처리 및 데이터 파악
# 꽃 모양에 따른 꽃의 종류 data set 불러옴
iris = datasets.load_iris()

# iris 데이터의 구성요소
# data: 데이터들
# feature_names: 설명변수(X)
# target: 종속변수(y) 라고 추측 가능
print(dir(iris))

# 편의를 위해 독립변수 2개만 사용
X = iris.data[:, [0, 2]]
y = iris.target

# 데이터 shape 확인
print(X.shape)
feature_names = [iris.feature_names[0], iris.feature_names[2]]
df_X = pd.DataFrame(X)
print(df_X.head())

print(y.shape)
df_y = pd.DataFrame(y)
print(df_y.head())

# 결측치 확인
# 행이 column, 열이 null 값
# 결측치는 없다
print(df_X.isnull().sum())
print(df_y.isnull().sum())

# target 과 대응하는 target_names 확인
print(set(y))
print(iris.target_names)

# 데이터의 분포 파악
# 종속변수(출력변수, 반응변수)의 분포
# y는 범주형 변수라서 class imbalance 한 경우라면 일반적인 방법에서 원하는 결과를 못얻을수도
# 각각의 변수가 몇개씩 있는지 count
df_y[0].value_counts().plot(kind='bar')
plt.show()

# 독립변수(속성, 입력변수, 설명변수)의 분포
for i in range(df_X.shape[1]):
    sns.distplot(df_X[i])
    plt.title(feature_names[i])
plt.show()

# #PCA 함수 활용 및 아웃풋 의미파악
# 2차원으로 축소하는 PCA 객채를 만들고, X 대입
pca = PCA(n_components=2)
pca.fit(X)

# pca 의 eigen vectors
print(pca.components_)

# eigen values
# 정렬된 상태로 보여줌
# eigen value 가 크다: 분산을 가장 잘 설명하고 있다
print(pca.explained_variance_)

# PC score
# -> 회귀분석에 이용할 수 있음
# -> 점의 벡터와 eigen vector 를 곱한 형태(점에서 eigen vector 에 정사영을 내림)
PCscore = pca.transform(X)
print(PCscore[0:5])

# PC score 는 eigen vector 위에서의 새로운 좌표(스칼라, 정사영, 내적)
# eigen vector 에 transpose 를 시켜준 이후에 연산을 진행한다
# -> 그냥 벡터끼리 곱하면 외적
eigens_v = pca.components_.transpose()

# 데이터들을 centering 을 하고 진행해야됨
# -> 데이터들의 평균을 0으로 만듦
mX = np.matrix(X)
for i in range(X.shape[1]):
    mX[:, i] = mX[:, i] - np.mean(X[:, i])
df_mX = pd.DataFrame(mX)

# pca.transform(X)를 통해 구한 값과 동일하다
print((mX * eigens_v)[0:5])

# PC score 산점도 표시
# 축이 바뀐 PC score
plt.scatter(PCscore[:, 0], PCscore[:, 1])
plt.show()

# 원래 데이터
# X, y 사이에 상관관계가 크다(비례관계)
# 바로 위의 PC score 그래프
# -> 데이터들을 각각의 eigen vector 에 대해 정사형을 시킨 후의 결과
plt.scatter(df_mX[0], df_mX[1])
origin = [0, 0], [0, 0]
plt.quiver(*origin, eigens_v[0, :], eigens_v[1, :], color=['r', 'b'], scale=3)
plt.show()

# #PC 를 활용한 회귀분석
# 모든 독립변수를 이용하여 PC 를 뽑아냄
# PC 를 활용하는 목적
# -> 1. dimension reduction
# -> 2. 다중공선성
# 이번 실습에서는 dimension reduction 을 했을때 의미를 가지는 상황을

# 데이터 전처리
X2 = iris.data
pca2 = PCA(n_components=4)
pca2.fit(X2)

# eigen value 확인
# 첫번째 PC 값이 다른 값들에 비해 매우 크다
# -> 매우 중요한 변수이다
# -> dimension reduction 을 통해 의미있는 결과를 얻을것으로 기대
print(pca2.explained_variance_)

# PC score
# PC 값들이 차이가 많이 나서 많아야 2개 뽑는게 적당하다고 판단
PCs = pca2.transform(X2)[:, 0:2]

# 첫번째로, 모든 변수를 활용한 회귀분석을 해봄
# 종속변수가 범주형 자료형이기 때문에 Rogistic Regression 을 활용
# 종류가 3개인 번주형 자료형이기 때문에 일한 로지스틱이 아닌, g-rogistic 사용
# g-rogistic 은 일반적인 방법으로는 풀리지 않기 때문에 알고리즘을 따로 지정해야됨
# multi_class parameter: 종속변수가 3개 이상의 데이터로 이루어져있다
# 실행 결과, ConvergenceWarning
# -> 회귀계수들이 수렴하지 않는다. 회귀계수를 구할 수가 없다.
# 모델의 복잡성으로 인하여 기존 자료를 이용한 분석은 수렴하지 않는 모습
# -> 모델을 간단하게 만들 필요가 있음
# -> PC 를 활용하여 X 변수의 개수를 줄일 수 있다
cif = LogisticRegression(solver='sag', multi_class='multinomial').fit(X2, y)

# 아까 뽑아낸 PC 를 활용하여 다시 분석
cif2 = LogisticRegression(solver='sag', multi_class='multinomial').fit(PCs, y)
# train set 을 활용한 예측
print(cif2.predict(PCs))
# 원본 데이터와 비교(confusion matrix)
print(confusion_matrix(y, cif2.predict(PCs)))

# PC 를 활용한 모델과 임의로 변수 2개를 뽑아낸 모델 비교
# 임의로 변수 2개를 뽑아낸 모델
# confusion matrix 를 비교하면 성능이 하락했음을 알 수 있음
cif3 = LogisticRegression(solver='sag', max_iter=1000, random_state=0, multi_class='multinomial').fit(X2[:, 0:2], y)
print(confusion_matrix(y, cif3.predict(X2[:, 0:2])))

# 차원축소를 통해 모델의 복잡성을 줄이는 동시에 최대한 많은 정보를 활용하여 분석할 수 있음
