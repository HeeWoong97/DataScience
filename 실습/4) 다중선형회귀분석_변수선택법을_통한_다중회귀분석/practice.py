# 다중선형회귀분석 실습 - 도요타 차량 가격 예측
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time
import itertools
from sklearn import metrics

# 데이터 불러오기
corolla = pd.read_csv('ToyotaCorolla.csv')

# 데이터 수와 변수의 수 확인
nCar = corolla.shape[0]
nVar = corolla.shape[1]

# 범주형 변수를 이진형 변수로 변환
# Fuel_Type 에 연료의 종류가 몇개 있는가?
# Petrol, Diesel, CNG 3가지 종류가 있다.
print(corolla.Fuel_Type.unique())

dummy_p = np.repeat(0, nCar)
dummy_d = np.repeat(0, nCar)
dummy_c = np.repeat(0, nCar)

# 인덱스 슬라이싱 후 (binary=1) 대입
p_idx = np.array(corolla.Fuel_Type == "Petrol")
d_idx = np.array(corolla.Fuel_Type == "Diesel")
c_idx = np.array(corolla.Fuel_Type == "CNG")

dummy_p[p_idx] = 1
dummy_d[d_idx] = 1
dummy_c[c_idx] = 1

# 불필요한 변수 제거 및 가변수 추가
# 차 가격을 예측하는데 불필요한 Id, Model, Fuel_Type 열을 제거하고 Fuel 추가
Fuel = pd.DataFrame({'Petrol': dummy_p, 'Diesel': dummy_d, 'CNG': dummy_c})

corolla_ = corolla.drop(['Id', 'Model', 'Fuel_Type'], axis=1, inplace=False)
mlr_data = pd.concat((corolla_, Fuel), axis=1)

# pandas get_dummies 함수를 이용해서 더 간단하게 가능
mlr_data = pd.get_dummies(corolla, columns=['Fuel_Type'])

# 상수항 추가
mlr_data_ = sm.add_constant(mlr_data, has_constant='add')

# 설명변수(X), 타겟변수(Y) 분리 및 학습데이터와 평가데이터 분할
# 우리가 예측할 Price 를 제외한 나머지 모두를 설명변수로
# y를 제외한 X 변수의 이름
feature_columns = list(mlr_data_.columns.difference(['Price']))

X = mlr_data_[feature_columns]
y = mlr_data_.Price

train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.7, test_size=0.3)

# 회귀모델 접합
# R^2 값이 높고, 대부분이 값들이 유의미하다
# Diesel 같은 변수들의 p-value 를 보면 결과에 큰 영향을 끼치지 않는 경우도 확인 가능
# 변수가 30개 이상으로 매우 많기 때문에 다중공선성도 높다고 추측 가능
full_model = sm.OLS(train_y, train_x)
fitted_full_model = full_model.fit()
print(fitted_full_model.summary())

# VIF 를 통한 다중공선성 확인
# 몇몇 변수는 inf(infinite)값을 가진다
# -> 다중공선성이 매우 심하다
# -> summary 를 통해 본 결과에서는 p-value 도 낮고 유의미한 변수였다
# -> VIF 값이 높다고 무조건 지우기에는 애매하다
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(
    mlr_data_.values, i) for i in range(mlr_data_.shape[1])]
vif["features"] = mlr_data_.columns
print(vif)

# 학습 데이터의 잔차 확인
# QQ-plot(정규성 확인)
# y=x 그래프를 벗어나는 데이터가 몇개 있지만 정규분포를 충분히 따른다고 할 수 있다
res = fitted_full_model.resid
flg = sm.qqplot(res, fit=True, line='45')
plt.show()

# residual pattern 확인
# 테스트 데이터에 대한 예측
# 잔차가 균등하게 있다
# 잔차가 y의 예측값에 따라서 변하지는 않음
pred_y = fitted_full_model.predict(train_x)
fig = plt.scatter(pred_y, res, s=4)
plt.xlim(4000, 30000)
plt.xlim(4000, 30000)
plt.xlabel('Fitted values')
plt.ylabel('Residual')
plt.show()

# 검증 데이터에 대한 예측
# 일부 데이터 제외하고 검증 데이터에 대해서 잘 맞는다고 판단 가능
pred_y2 = fitted_full_model.predict(test_x)
plt.plot(np.array(test_y - pred_y2), label='pred_full')
plt.legend()
plt.show()

# MSE 확인
print(mean_squared_error(y_true=test_y, y_pred=pred_y2))


# 변수선택법
# 모델을 return 해주는 함수
def processSubset(X, y, feature_set):
    model = sm.OLS(y, X[list(feature_set)])
    regr = model.fit()
    AIC = regr.aic
    return {'model': regr, 'AIC': AIC}


print(processSubset(X=train_x, y=train_y, feature_set=feature_columns[0:5]))


# 가장 낮은 AIC 를 가지는 모델 선택 및 저장하는 함수
# ex) k=2 : 변수들 중 2개만 뽑는 모든 조합을 시행하고 가장 낮은 AIC 를 가지는 모델을 반환
def getBest(X, y, k):
    tic = time.time()
    results = []
    for combo in itertools.combinations(X.columns.difference(['const']), k):  # 각 변수 조합을 고려한 경우의 수
        combo = (list(combo) + ['const'])
        results.append(processSubset(X, y, feature_set=combo))  # 모델링 된 것을 저장
    models = pd.DataFrame(results)  # 데이터프레임으로 변환
    # 가장 낮은 AIC 를 가지는 모델 선택 및 저장
    best_model = models.loc[models['AIC'].argmin()]  # index
    toc = time.time()  # 종료시간
    print('Processed', models.shape[0], 'models on', k, 'predictors in', (toc - tic), 'Seconds.')
    return best_model


print(getBest(X=train_x, y=train_y, k=2))

# for 문 작동방식 확인
# X 에서 const(상수항)을 떼고, 조합을 하고, 다시 const 추가
# X 는 이미 상수항이 추가된 상태
for combo in itertools.combinations(X.columns.difference(['const']), 2):  # 각 변수 조합을 고려한 경우의 수
    print(list(combo) + ['const'])

# 변수 선택에 따른 학습시간과 저장
models = pd.DataFrame(columns=['AIC', 'model'])
tic = time.time()
for i in range(1, 4):
    models.loc[i] = getBest(X=train_x, y=train_y, k=i)
toc = time.time()
print('Total elapsed time:', toc - tic, 'seconds.')

print(models)
print(models.loc[3, 'model'].summary())

# 모든 변수들 모델링 한것과 비교
# 모든 변수를 고려한 모델의 R^2가 높고, AIC 가 낮다
# -> 모든 변수를 고려한 모델이 더 좋다
print("full model Rsquared: ", "{:.5f}".format(fitted_full_model.rsquared))
print("full model AIC: ", "{:.5f}".format(fitted_full_model.aic))
print("full model MSE: ", "{:.5f}".format(fitted_full_model.mse_total))
print("selected model Rsquared: ", "{:.5f}".format(models.loc[3, "model"].rsquared))
print("selected model AIC: ", "{:.5f}".format(models.loc[3, "model"].aic))
print("selected model MSE: ", "{:.5f}".format(models.loc[3, "model"].mse_total))

# Plot the result
plt.figure(figsize=(20, 10))
plt.rcParams.update({'font.size': 18, 'lines.markersize': 10})

# Mallow Cp
# -> AIC 와 비슷하게 변수의 수만큼 패널티를 준 성능지표
# -> 낮을수록 좋다
plt.subplot(2, 2, 1)
Cp = models.apply(lambda row: (row[1].params.shape[0] + (row[1].mse_total - fitted_full_model.mse_total)
                               * (train_x.shape[0] - row[1].params.shape[0]) / fitted_full_model.mse_total), axis=1)
plt.plot(Cp)
plt.plot(Cp.argmin(), Cp.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('Cp')

# adj-rsquared plot
# adj-rsquared = Explained variation / Total variation
adj_rsquared = models.apply(lambda row: row[1].rsquared_adj, axis=1)
plt.subplot(2, 2, 2)
plt.plot(adj_rsquared)
plt.plot(adj_rsquared.argmax(), adj_rsquared.max(), "or")
plt.xlabel('# Predictors')
plt.ylabel('adjusted rsquared')

# aic
aic = models.apply(lambda row: row[1].aic, axis=1)
plt.subplot(2, 2, 3)
plt.plot(aic)
plt.plot(aic.argmin(), aic.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('AIC')

# bic
bic = models.apply(lambda row: row[1].bic, axis=1)
plt.subplot(2, 2, 4)
plt.plot(bic)
plt.plot(bic.argmin(), bic.min(), "or")
plt.xlabel(' # Predictors')
plt.ylabel('BIC')

# Mallow Cp 를 제외한 값들은 변수를 많이 선택할수록 좋은 값이 나왔다
plt.show()


# ###########전진선택법(Feedforward Selection)##########
# 처음부터 하나씩 증가하며 학습시키다가 AIC 가 높아지면 stop
# predictors 매개변수: 현재 선택되어 있는 변수
def forward(X, y, predictors):
    # 데이터 변수들이 미리정의된 predictors 에 있는지 없는지 확인 및 분류
    remaining_predictors = [p for p in X.columns.difference(['const']) if p not in predictors]
    tic = time.time()
    results = []
    for p in remaining_predictors:
        results.append(processSubset(X=X, y=y, feature_set=predictors + [p] + ['const']))
    # 데이터프레임으로 변환
    models = pd.DataFrame(results)

    # AIC가 가장 낮은 것을 선택
    best_model = models.loc[models['AIC'].argmin()]  # index
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors) + 1, "predictors in", (toc - tic))
    print('Selected predictors:', best_model['model'].model.exog_names, ' AIC:', best_model[0])
    return best_model


# 전진선택법 모델
def forward_model(X, y):
    Fmodels = pd.DataFrame(columns=["AIC", "model"])
    tic = time.time()
    # 미리 정의된 데이터 변수
    predictors = []
    # 변수 1~10개 : 0~9 -> 1~10
    for i in range(1, len(X.columns.difference(['const'])) + 1):
        Forward_result = forward(X=X, y=y, predictors=predictors)  # forward 함수를 통해 선택된 모델 하나
        if i > 1:
            if Forward_result['AIC'] > Fmodel_before:  # Fmodel_before: 이전에 선택된 모델의 AIC
                break  # 이전 모델보다 AIC 가 높아지면 break
        Fmodels.loc[i] = Forward_result
        predictors = Fmodels.loc[i]["model"].model.exog_names  # 그때의 매개변수와 AIC 를 저장
        Fmodel_before = Fmodels.loc[i]["AIC"]
        predictors = [k for k in predictors if k != 'const']  # 선택된 predictor 로 업데이트
    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")

    return (Fmodels['model'][len(Fmodels['model'])])


# 변수 3개 모두 선택하는 것보다 시간이 더 짧다
Forward_best_model = forward_model(X=train_x, y=train_y)
print(Forward_best_model.aic)

# R^2값은 모든 변수로 그냥 학습시킨거랑 별 차이가 없다
# p-value 가 높아서 문제가 있다고 생각되던 변수들은 삭제된것을 확인 가능
# 일일히 다 지우지 않아도 변수선택법을 통해 삭제할 수 있다
print(Forward_best_model.summary())


# ##########후진소거법(Backward Elimination)##########
# 모든 변수를 학습시키고 하나씩 제가하다가 AIC 가 높아지면 stop
def backward(X, y, predictors):
    tic = time.time()
    results = []
    # 데이터 변수들이 미리정의된 predictors 조합 확인
    for combo in itertools.combinations(predictors, len(predictors) - 1):
        results.append(processSubset(X=X, y=y, feature_set=list(combo) + ['const']))
    models = pd.DataFrame(results)
    # 가장 낮은 AIC를 가진 모델을 선택
    best_model = models.loc[models['AIC'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors) - 1, "predictors in",
          (toc - tic))
    print('Selected predictors:', best_model['model'].model.exog_names, ' AIC:', best_model[0])
    return best_model


# 후진 소거법 모델
# 전체적인 구조는 전진선택법과 거의 동일
def backward_model(X, y):
    Bmodels = pd.DataFrame(columns=["AIC", "model"], index=range(1, len(X.columns)))
    tic = time.time()
    predictors = X.columns.difference(['const'])
    Bmodel_before = processSubset(X, y, predictors)['AIC']
    while (len(predictors) > 1):
        Backward_result = backward(X=train_x, y=train_y, predictors=predictors)  # 전체변수를 다 넣고 fitting
        if Backward_result['AIC'] > Bmodel_before:  # AIC 가 이전 모델보다 높아지면 break
            break
        Bmodels.loc[len(predictors) - 1] = Backward_result
        predictors = Bmodels.loc[len(predictors) - 1]["model"].model.exog_names
        Bmodel_before = Backward_result['AIC']
        predictors = [k for k in predictors if k != 'const']

    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")
    return (Bmodels['model'].dropna().iloc[0])


Backward_best_model = backward_model(X=train_x, y=train_y)
print(Backward_best_model.aic)

# 전진선택법과 거의 유사한 결과
# 변수선택법은 변수가 엄청 많지 않은 이상 유사한 결과가 나올 확률이 높다
print(Backward_best_model.summary())


# ##########단계적 선택법(Stepwise)##########
# feedforward, backward elimination 을 번갈아가며 실행
def Stepwise_model(X, y):
    Stepmodels = pd.DataFrame(columns=["AIC", "model"])
    tic = time.time()
    predictors = []
    Smodel_before = processSubset(X, y, predictors + ['const'])['AIC']
    # 변수 1~10개 : 0~9 -> 1~10
    for i in range(1, len(X.columns.difference(['const'])) + 1):
        Forward_result = forward(X=X, y=y, predictors=predictors)  # constant added
        print('forward')
        Stepmodels.loc[i] = Forward_result
        predictors = Stepmodels.loc[i]["model"].model.exog_names
        predictors = [k for k in predictors if k != 'const']
        Backward_result = backward(X=X, y=y, predictors=predictors)
        if Backward_result['AIC'] < Forward_result['AIC']:
            Stepmodels.loc[i] = Backward_result
            predictors = Stepmodels.loc[i]["model"].model.exog_names
            Smodel_before = Stepmodels.loc[i]["AIC"]
            predictors = [k for k in predictors if k != 'const']
            print('backward')
        if Stepmodels.loc[i]['AIC'] > Smodel_before:  # AIC 가 높아지면 break
            break
        else:
            Smodel_before = Stepmodels.loc[i]["AIC"]
    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")
    return (Stepmodels['model'][len(Stepmodels['model'])])


# Stepwise 는 전진선택법과 후진소거법을 동시에 실행하기 때문에 시간이 더 오래 걸린다
Stepwise_best_model = Stepwise_model(X=train_x, y=train_y)
print(Stepwise_best_model.aic)
print(Stepwise_best_model.summary())

# ##########성능평가##########
# the number of params
print(Forward_best_model.params.shape, Backward_best_model.params.shape, Stepwise_best_model.params.shape)

# 모델에 의해 예측된/추정된 값 <->  test_y
pred_y_full = fitted_full_model.predict(test_x)
pred_y_forward = Forward_best_model.predict(test_x[Forward_best_model.model.exog_names]) # 전진선택법에서 선택된 변수들을 선택
pred_y_backward = Backward_best_model.predict(test_x[Backward_best_model.model.exog_names])
pred_y_stepwise = Stepwise_best_model.predict(test_x[Stepwise_best_model.model.exog_names])

perf_mat = pd.DataFrame(columns=["ALL", "FORWARD", "BACKWARD", "STEPWISE"],
                        index=['MSE', 'RMSE', 'MAE', 'MAPE'])


# MAE 는 0~무한대 값을 가지기 때문에 성능지표를 비교하기 어렵다
# MAPE 는 이를 %화 해서 만든것. 객관적인 성능지표 비교를 할 수 있게 해준다
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# 성능지표
# 모든 변수들을 고려한 모델의 MSE 가 가장 낮다
# MAE 지표를 보면 선택법을 사용한 모델의 값이 좋게 나왔다
# -> 모델들의 차이가 그렇게 나지 않는다
# -> 변수의 수가 그렇게 많지 않다면 feedforward, backward, stepwise 의 성능 차이는 그렇게 나지 않는다
# -> 하지만 선택법을 사용한 모델들에서 고려한 변수의 개수가 더 적다
# -> 비슷한 성능을 내는 모델이라면 적은 변수를 고려한 모델을 선택하는게 더 바람직하다
perf_mat.loc['MSE']['ALL'] = metrics.mean_squared_error(test_y, pred_y_full)
perf_mat.loc['MSE']['FORWARD'] = metrics.mean_squared_error(test_y, pred_y_forward)
perf_mat.loc['MSE']['BACKWARD'] = metrics.mean_squared_error(test_y, pred_y_backward)
perf_mat.loc['MSE']['STEPWISE'] = metrics.mean_squared_error(test_y, pred_y_stepwise)

perf_mat.loc['RMSE']['ALL'] = np.sqrt(metrics.mean_squared_error(test_y, pred_y_full))
perf_mat.loc['RMSE']['FORWARD'] = np.sqrt(metrics.mean_squared_error(test_y, pred_y_forward))
perf_mat.loc['RMSE']['BACKWARD'] = np.sqrt(metrics.mean_squared_error(test_y, pred_y_backward))
perf_mat.loc['RMSE']['STEPWISE'] = np.sqrt(metrics.mean_squared_error(test_y, pred_y_stepwise))

perf_mat.loc['MAE']['ALL'] = metrics.mean_absolute_error(test_y, pred_y_full)
perf_mat.loc['MAE']['FORWARD'] = metrics.mean_absolute_error(test_y, pred_y_forward)
perf_mat.loc['MAE']['BACKWARD'] = metrics.mean_absolute_error(test_y, pred_y_backward)
perf_mat.loc['MAE']['STEPWISE'] = metrics.mean_absolute_error(test_y, pred_y_stepwise)

perf_mat.loc['MAPE']['ALL'] = mean_absolute_percentage_error(test_y, pred_y_full)
perf_mat.loc['MAPE']['FORWARD'] = mean_absolute_percentage_error(test_y, pred_y_forward)
perf_mat.loc['MAPE']['BACKWARD'] = mean_absolute_percentage_error(test_y, pred_y_backward)
perf_mat.loc['MAPE']['STEPWISE'] = mean_absolute_percentage_error(test_y, pred_y_stepwise)

print(perf_mat)

print(len(fitted_full_model.params))
print(len(Forward_best_model.params))
print(len(Backward_best_model.params))
print(len(Stepwise_best_model.params))

# 1. 데이터를 전처리
# 2. summary 를 통해 어떤 변수에서 어느정도 성능이 나오는지를 확인
# 3. 제거할 변수는 제거하되, 도메인 지식에 의거해서 할것
# 4. VIF 같은 지표를 통해 다중공선성을 확인
# 5. 연관성이 있는 변수가 많다면 변수선택법을 통해 지우는게 일반적
# 6. 모델을 다 만들고 나서는 잔차의 경향을 확인하는게 필요
# 7. 최종적으로는 validation data 를 통해 꼭 확인해보기
