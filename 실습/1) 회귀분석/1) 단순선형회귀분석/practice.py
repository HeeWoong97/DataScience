import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

boston = pd.read_csv('Boston_house.csv')

boston_data = boston.drop(['Target'], axis=1)

target = boston[['Target']]
crim = boston[['CRIM']]
rm = boston[['RM']]
lstat = boston[['LSTAT']]

# 변수에 상수항 추가
crim1 = sm.add_constant(crim, has_constant='add')
rm1 = sm.add_constant(rm, has_constant='add')
lstat1 = sm.add_constant(lstat, has_constant='add')

# sm.OLS 접합시키기
model1 = sm.OLS(target, crim1)
model2 = sm.OLS(target, rm1)
model3 = sm.OLS(target, lstat1)

fitted_model1 = model1.fit()
fitted_model2 = model2.fit()
fitted_model3 = model3.fit()

# summary를 통해 결과출력
fitted_model1.summary()
fitted_model2.summary()
fitted_model3.summary()

# 매개변수값 출력
fitted_model1.params
fitted_model2.params
fitted_model3.params

# y_hat = beta0 + beta1*x 계산
# 1. 회귀계수 * 데이터(x)
np.dot(crim1, fitted_model1.params)

# 2. predict 함수를 이용해 y_hat 구하기
pred1 = fitted_model1.predict(crim1)

# 1번과 2번 방법의 차이
# 차이가 없다!
np.dot(crim1, fitted_model1.params) - pred1

pred2 = fitted_model2.predict(rm1)
pred3 = fitted_model3.predict(lstat1)

# 접합시킨 직선 시각화
plt.yticks(fontname="Arial")

plt.scatter(crim, target, label="data")
plt.plot(crim, pred1, label="result")
plt.legend()
plt.show()

plt.scatter(rm, target, label="data")
plt.plot(rm, pred2, label="result")
plt.legend()
plt.show()

plt.scatter(lstat, target, label="data")
plt.plot(lstat, pred3, label="result")
plt.legend()
plt.show()

plt.scatter(target, pred1)
plt.xlabel("real_value")
plt.ylabel("pred_value")
plt.show()

# residual 시각화
fitted_model1.resid.plot()
plt.xlabel("residual_number")
plt.show()

fitted_model2.resid.plot()
plt.xlabel("residual_number")
plt.show()

fitted_model3.resid.plot()
plt.xlabel("residual_number")
plt.show()

# 잔차의 합
np.sum(fitted_model1.resid)
np.sum(fitted_model2.resid)
np.sum(fitted_model3.resid)

# 한 화면에 합쳐서 보기
fitted_model1.resid.plot(label="crim")
fitted_model2.resid.plot(label="rm")
fitted_model3.resid.plot(label="lstat")
plt.legend()
