import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

boston = pd.read_csv('Boston_house.csv')

boston_data = boston.drop(['Target'], axis=1)

# 변수들 설정
target = boston[['Target']]

# crim, rm, lstat 변수를 통한 다중회귀분석
x_data = boston[['CRIM', 'RM', 'LSTAT']]

# 상수항 추가
x_data_1 = sm.add_constant(x_data, has_constant='add')

# 회귀모델에 접합
multi_model = sm.OLS(target, x_data_1)
fitted_multi_model = multi_model.fit()

# crim, rm, lstat, b, tax, age, zn, nox, indus 변수를 통한 다중선형회귀분석
x_data1 = boston[['CRIM', 'RM', 'LSTAT', 'B', 'TAX', 'AGE', 'ZN', 'NOX', 'INDUS']]

# 상수항 추가
x_data1_1 = sm.add_constant(x_data1, has_constant='add')

# 회귀모델에 접합
multi_model1 = sm.OLS(target, x_data1_1)
fitted_multi_model1 = multi_model1.fit()

# 변수가 더 많이 추가되었음에도 crim, rm, lstat 만 사용한 모델보다 R^2값은 크게 증가하지 않았다
# 계수들의 값이 0에 가깝다.
# -> 변수들간에 상관성이 있다. 다중공선성
# 이런 상황에서는 변수 개수가 적은 모델을 사용하는게 훨씬 유익하다
# 변수를 선택할때 p-value 값을 보고 어느것을 선택할지 결정
# p-value 값이 애매한것은 분석가의 판단에 맡긴다(crim 이 집값에 영향을 미치는가? 그렇다. 선택)
# p-value 값이 높은 변수부터 제거하면서 분석하기
print(fitted_multi_model1.summary())

# 잔차 시각화
# 변수를 추가했어도 잔차가 크게 줄어들지 않았다
fitted_multi_model.resid.plot(label='3 vars')
fitted_multi_model1.resid.plot(label='9 vars')
plt.legend()
plt.show()


# 상관계수/산점도를 통해 다중공선성 확인 #
# 상관계수 행렬
print(x_data1.corr())

# 상관계수 시각화
cmap = sns.light_palette('darkgray', as_cmap=True)
sns.heatmap(x_data1.corr(), annot=True, cmap=cmap)
plt.show()

# 변수별 산점도 시각화
sns.pairplot(x_data1)
plt.show()

# VIF 를 통한 다중공선성 확인
# NOX 변수의 VIF 가 크게 나온다
# 일단 NOX 를 제거하고 VIF 를 다시 검사
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(x_data1.values, i) for i in range(x_data1.shape[1])]
vif['features'] = x_data1.columns
print(vif)

# NOX 제거 후 VIF 확인
# 영향을 주는 변수가 줄어들었기 때문에 다른 변수들의 VIF 값도 감소한다
# RM 변수의 VIF 가 크게 나온다.
# p-value 와 산점도를 봤을때는 여기에서 멈춰도 된다.
# 하지만 일단 RM을 제거하고 다시 검사
vif = pd.DataFrame()
x_data2 = x_data1.drop('NOX', axis=1)
vif['VIF Factor'] = [variance_inflation_factor(x_data2.values, i) for i in range(x_data2.shape[1])]
vif['features'] = x_data2.columns
print(vif)

# RM 제거 후 VIF 확인
# 이정도에서 멈취도 충분하다.
vif = pd.DataFrame()
x_data3 = x_data2.drop('RM', axis=1)
vif['VIF Factor'] = [variance_inflation_factor(x_data3.values, i) for i in range(x_data3.shape[1])]
vif['features'] = x_data3.columns
print(vif)

# 상수항 추가
x_data2_1 = sm.add_constant(x_data2, has_constant='add')
x_data3_1 = sm.add_constant(x_data3, has_constant='add')

# 회귀모델에 접합
multi_model2 = sm.OLS(target, x_data2_1)
multi_model3 = sm.OLS(target, x_data3_1)

fitted_multi_model2 = multi_model2.fit()
fitted_multi_model3 = multi_model3.fit()

# 모델 결과 비교
# model2: NOX 만 제거
# R^2값이 크게 변하지 않았다: NOX 는 변동성의 큰 부분을 차지하지 않는다.
# model3: NOX, RM 제거
# R^2값이 오히려 model2보다 작다: RM은 제거하면 안되는 중요한 변수이다.
# AIC 의 경우도 RM을 제거하기 전 값이 더 낮다.
# -> model2가 더 좋은 모델이다.
print(fitted_multi_model2.summary())
print(fitted_multi_model3.summary())


# 학습/검증 데이터 분할 #
# random_state 매개변수: 그 값으로는 항상 같은 결과. 다른 값을 주면 결과가 다르다
X = x_data1_1
y = target
train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=1)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

# train_x에 회귀모델 접합
# 이미 상수항이 추가된 set 을 학습 데이터로 썼기 때문에 상수항 추가는 필요 없다.
multi_train_model = sm.OLS(train_y, train_x)
fitted_train_model = multi_train_model.fit()

# 검증데이터에 대한 예측값과 true 값 비교
# 학습데이터로 회귀시킨 모델에 test_x 대입, test_y와 비교
# 어느정도의 패턴은 맞춰간다고 판단 가능
plt.subplot(1, 3, 1)
plt.title('All data')
plt.plot(np.array(fitted_train_model.predict(test_x)), label='pred')
plt.plot(np.array(test_y), label='true')
plt.legend()
plt.plot(1, 1)

# x_data2와 x_data3 학습, 검증데이터 분할
# 결과를 서로 비교하기 위해 테스트와 검증 데이터를 같은 index 의 데이터로 나누어야 한다.
# -> random_state 값을 동일하게
X = x_data2_1
y = target
train_x2, test_x2, train_y2, test_y2 = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=1)

X = x_data3_1
y = target
train_x3, test_x3, train_y3, test_y3 = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=1)

# train_x2, train_x3에 회귀모델 접합
multi_train_model2 = sm.OLS(train_y2, train_x2)
multi_train_model3 = sm.OLS(train_y3, train_x3)

fitted_train_model2 = multi_train_model2.fit()
fitted_train_model3 = multi_train_model3.fit()

# 검증데이터에 대한 예측값과 true 값 비교
# 학습데이터로 회귀시킨 모델에 test_x 대입, test_y와 비교
# NOX, RM 모두 제거한 모델이 가장 안좋다는 것을 알수있다.
plt.subplot(1, 3, 2)
plt.title('remove NOX')
plt.plot(np.array(fitted_train_model2.predict(test_x2)), label='pred')
plt.plot(np.array(test_y2), label='true')
plt.legend()
plt.plot(1, 2)

plt.subplot(1, 3, 3)
plt.title('remove NOX, RM')
plt.plot(np.array(fitted_train_model3.predict(test_x3)), label='pred')
plt.plot(np.array(test_y3), label='true')
plt.legend()
plt.plot(1, 3)
plt.show()

# MSE 를 통한 검증데이터에 대한 성능비교
# NOX 만 제거한 모델의 MSE 값이 가장 낮다
MSE_model = mean_squared_error(y_true=test_y['Target'], y_pred=fitted_train_model.predict(test_x))
MSE_model2 = mean_squared_error(y_true=test_y2['Target'], y_pred=fitted_train_model2.predict(test_x2))
MSE_model3 = mean_squared_error(y_true=test_y3['Target'], y_pred=fitted_train_model3.predict(test_x3))

print('All data: ', MSE_model)
print('Remove NOX: ', MSE_model2)
print('Remove NOX, RM: ', MSE_model3)

# 1: 9개의 변수들을 선택했다.
# 2: 회귀모델에 접합시키고 결과를 본 결과, 일부 변수들의 p-value 가 높았다.
# 3: 다중공선성을 확인
# 4: VIF 가 높은 데이터를 삭제하고 다시 모델에 접합시켰다.
# 5: 여전히 p-value 가 높은 변수들이 존재해서 다시 삭제하고 모델에 접합시켰다.
# 6: 하지만 R^2이 크게 변하지 않았다 -> 삭제한 변수는 중요한 부분을 차지하고 있다.

# 학습, 검증 데이터로 나누어서도 모델에 접합시켜봄
# 검증, 학습 데이터의 비교와 MSE 확인을 통해 변수 하나만 삭제한 모델이 가장 성능이 좋았다는 것을 확인
# p-value 가 높은 데이터를 무조건 삭제한다고 모델의 성능이 좋아지는것은 아니다.
