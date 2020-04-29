from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

# Gaussian Naive Bayes
# '설명변수'가 연속형인 경우
iris = datasets.load_iris()

df_X = pd.DataFrame(iris.data)
df_y = pd.DataFrame(iris.target)

print(df_X.head())

# model fitting
gnb = GaussianNB()
fitted = gnb.fit(iris.data, iris.target)
y_pred = fitted.predict(iris.data)

# 종속형 변수는 3개의 범주(0, 1, 2)를 가지고 있다(column).
# 1, 48, 51, 100번째 예측(row)에서 0, 1, 2가 나올 확률을 보여줌
print(fitted.predict_proba(iris.data)[[1, 48, 51, 100]])

# 1, 48, 51, 100번째 예측값을 보여줌
print(fitted.predict(iris.data)[[1, 48, 51, 100]])

# confusion matrix
print(confusion_matrix(iris.target, y_pred))

# prior 설정
# -> 특정 범주일 확률
# 1, 2, 3번째 범주가 나올 확률을 설정
# 3번째가 나올 확률을 높였으니 3번째 범주는 모두 맞춤
# 다른 변수들은 예측도가 낮아짐
gnb2 = GaussianNB(priors=[1 / 100, 1 / 100, 98 / 100])
fitted2 = gnb2.fit(iris.data, iris.target)
y_pred2 = fitted2.predict(iris.data)
print(confusion_matrix(iris.target, y_pred2))

# Multinomial Naive Bayes
# '설명변수'가 범주형인 경우
# X: 설명변수가 100개 있는 아무런 의미없는 데이터 집합
X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])

# 모델 생성
cif = MultinomialNB()
cif.fit(X, y)

# 피팅된 모델이 X의 2번(=3번째) 데이터 집합으로 3을 예측했다
# 성공적으로 예측
print(cif.predict_proba(X[2:3]))
print(cif.predict(X[2:3]))

# prior 변경
# 합이 1이 아니더라도 상대적 비율로 설정됨
cif2 = MultinomialNB(class_prior=[0.1, 0.1999, 0.0001, 0.1, 0.1, 0.1])
cif2.fit(X, y)

# 가중치를 많이 준 범주의 확률이 높아졌다
print(cif2.predict_proba(X[2:3]))
