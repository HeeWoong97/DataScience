import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('kc_house_data.csv')
'''
id: 집 고유아이디
date: 집이 팔린 날짜 
price: 집 가격 (타겟변수)
bedrooms: 주택 당 침실 개수
bathrooms: 주택 당 화장실 개수
floors: 전체 층 개수
waterfront: 해변이 보이는지 (0, 1)
condition: 집 청소상태 (1~5)
grade: King County grading system 으로 인한 평점 (1~13)
yr_built: 집이 지어진 년도
yr_renovated: 집이 리모델링 된 년도
zipcode: 우편번호
lat: 위도
long: 경도
'''
# ncar: 데이터의 갯수
# nvar: 변수의 갯수
ncar = data.shape[0]
nvar = data.shape[1]

# ##########의미가 없다고 판단되는 변수 제거
data = data.drop(['id', 'date', 'zipcode', 'lat', 'long'], axis=1)

# ##########설명변수와 타겟변수 분리, 학습데이터와 검증데이터 분리
feature_columns = list(data.columns.difference(['price']))

X = data[feature_columns]
y = data['price']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

# ##########선형모델에 접합 후 검증
sm_train_X = sm.add_constant(train_X, has_constant='add')
sm_model = sm.OLS(train_y, sm_train_X)
fitted_sm_model = sm_model.fit()

# R^2: y 의 총 변동성 중 X 가 차지하는 비율
print(fitted_sm_model.summary())

sm_test_X = sm.add_constant(test_X, has_constant='add')
sm_model_predict = fitted_sm_model.predict(sm_test_X)

print(sqrt(mean_squared_error(sm_model_predict, test_y)))  # RMSE

# ##########for문을 사용한 bagging##########
# ##########Bagging 한 결과가 일반적인 결과보다 좋은지 확인
bagging_predict_result = []
for _ in range(10):
    data_index = [data_index for data_index in range(train_X.shape[0])]
    random_data_index = np.random.choice(data_index, train_X.shape[0])  # 복원추출. data_index list 에서 train_X.shape[0]번 추출
    print(len(set(random_data_index)))  # unique 한 것들의 갯수. 복원추출하면 전체 데이터의 60%정도가 뽑힌다는 것을 확인 가능

    sm_train_X = train_X.iloc[random_data_index,]
    sm_train_y = train_y.iloc[random_data_index,]

    sm_train_X = sm.add_constant(sm_train_X, has_constant='add')

    sm_model = sm.OLS(sm_train_y, sm_train_X)
    fitted_sm_model = sm_model.fit()

    pred = fitted_sm_model.predict(sm_test_X)
    bagging_predict_result.append(pred)
    print(sqrt(mean_squared_error(pred, test_y)))

bagging_predict = []
for lst2_index in range(test_X.shape[0]):
    temp_predict = []
    for lst_index in range(len(bagging_predict_result)):
        temp_predict.append(bagging_predict_result[lst_index].values[lst2_index])
    bagging_predict.append(np.mean(temp_predict))

# Ensemble 모델을 사용했을때의 RMSE
# 모델 하나를 사용했을때와 별 차이가 없다
# 상황에 따라 다름
print(sqrt(mean_squared_error(bagging_predict, test_y)))

# ##########패키지를 이용한 bagging##########
regression_model = LinearRegression()
linear_model1 = regression_model.fit(train_X, train_y)

# ##########Bagging 을 이용하여 선형 회귀 모형에 접합 후 평가 (Sampling 5번)
bagging_model = BaggingRegressor(base_estimator=regression_model, n_estimators=5)
linear_model2 = bagging_model.fit(train_X, train_y)
predict2 = linear_model2.predict(test_X)

print(sqrt(mean_squared_error(predict2, test_y)))

# ##########Sampling 을 많이 해봄
bagging_model = BaggingRegressor(base_estimator=regression_model, n_estimators=30)
linear_model2 = bagging_model.fit(train_X, train_y)
predict2 = linear_model2.predict(test_X)

print(sqrt(mean_squared_error(predict2, test_y)))

# ##########학습데이터를 의사결정나무모형에 접합 후 평가 데이터로 검증##########
decision_tree_model = DecisionTreeRegressor()
tree_model = decision_tree_model.fit(train_X, train_y)
predict_tree = tree_model.predict(test_X)

# 성능이 안좋아짐
print(sqrt(mean_squared_error(predict_tree, test_y)))

bagging_predict_result = []
for _ in range(10):
    data_index = [data_index for data_index in range(train_X.shape[0])]
    random_data_index = np.random.choice(data_index, train_X.shape[0])  # 복원추출. data_index list 에서 train_X.shape[0]번 추출
    print(len(set(random_data_index)))  # unique 한 것들의 갯수. 복원추출하면 전체 데이터의 60%정도가 뽑힌다는 것을 확인 가능

    sm_train_X = train_X.iloc[random_data_index,]
    sm_train_y = train_y.iloc[random_data_index,]

    decision_tree_model = DecisionTreeRegressor()
    tree_model = decision_tree_model.fit(sm_train_X, sm_train_y)
    predict_tree = tree_model.predict(test_X)

    bagging_predict_result.append(predict_tree)
    print(sqrt(mean_squared_error(predict_tree, test_y)))

bagging_predict = []
for lst2_index in range(test_X.shape[0]):
    temp_predict = []
    for lst_index in range(len(bagging_predict_result)):
        temp_predict.append(bagging_predict_result[lst_index][lst2_index])
    bagging_predict.append(np.mean(temp_predict))
print(sqrt(mean_squared_error(bagging_predict, test_y)))

# ##########패키지 이용##########
bagging_model = BaggingRegressor(base_estimator=decision_tree_model, n_estimators=10)
linear_model2 = bagging_model.fit(train_X, train_y)
predict2 = linear_model2.predict(test_X)

print(sqrt(mean_squared_error(predict2, test_y)))

bagging_model = BaggingRegressor(base_estimator=decision_tree_model, n_estimators=30)
linear_model2 = bagging_model.fit(train_X, train_y)
predict2 = linear_model2.predict(test_X)

print(sqrt(mean_squared_error(predict2, test_y)))

bagging_model = BaggingRegressor(base_estimator=decision_tree_model, n_estimators=40)
linear_model2 = bagging_model.fit(train_X, train_y)
predict2 = linear_model2.predict(test_X)

print(sqrt(mean_squared_error(predict2, test_y)))

# 횟수를 늘린다고 해도 성능이 극적으로 좋아지지는 않는다.
