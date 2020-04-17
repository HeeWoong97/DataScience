import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from numpy import linalg

boston = pd.read_csv('Boston_house.csv')

boston_data = boston.drop(['Target'], axis=1)

# 변수들 설정
target = boston[['Target']]
x_data = boston[['CRIM', 'RM', 'LSTAT']]

# 상수항 추가: 잔차 e가 아닌 beta0를 추가하는 과정
x_data1 = sm.add_constant(x_data, has_constant='add')
print(x_data1)

# 회귀모델에 접합
multi_model = sm.OLS(target, x_data1)
fitted_multi_model = multi_model.fit()
# print(fitted_multi_model.summary())

# 공식으로 beta(회귀계수) 구하기: beta = (X'X)-1X'y
# fitted_multi_model.params 와 동일한 결과
# x_data1 대신 x_data 사용하지 않도록 유의
# print(np.dot(np.dot(linalg.inv(np.dot(x_data1.T, x_data1)), x_data1.T), target))

# y_hat 구하기
pred1 = fitted_multi_model.predict(x_data1)

# 잔차 시각화
plt.yticks(fontname="Arial")

fitted_multi_model.resid.plot()
plt.xlabel("residual_number")
plt.show()
