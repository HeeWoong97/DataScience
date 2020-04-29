import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
import time

# 경력, 수입 등등의 데이터를 통해 대출을 할지 말지 예측하는 모델
ploan = pd.read_csv('Personal Loan.csv')
print(ploan.head())
'''
Experience 경력
Income 수입
Famliy 가족단위
CCAvg 월 카드사용량 
Education 교육수준 (1: undergrad; 2, Graduate; 3; Advance )
Mortgage 가계대출
Securities account 유가증권계좌유무
CD account 양도예금증서 계좌 유무
Online 온라인계좌유무
CreidtCard 신용카드유무 

'''

# 의미없는 변수 제거(ID, ZIP Code)
ploan_processed = ploan.dropna().drop(['ID', 'ZIP Code'], axis=1, inplace=False)

# 상수항 추가
ploan_processed = sm.add_constant(ploan_processed, has_constant='add')

# 설명변수(X), 타겟변수(y)분리 및 학습데이터와 평가데이터
# 예측할 변수: Personal Loan
# 대출여부: 1 or 0
feature_columns = ploan_processed.columns.difference(['Personal Loan'])

X = ploan_processed[feature_columns]
y = ploan_processed['Personal Loan']

# stratify 매개변수: 원래 데이터의 1과 0의 비율을 train 과 test 데이터에서도 유지
train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y, train_size=0.7, test_size=0.3, random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

# 로지스틱 회귀모형 모델링 y=f(x)
# 로지스틱 모형 접합
# sm.Logit 함수 사용
# method='newton': 뉴턴법을 사용해서 접합
model = sm.Logit(train_y, train_x)
results = model.fit(method='newton')
print(results.summary())

# 회귀계수 출력
# params 를 통해 출력하는것은 Logit 방식
print(results.params)

# exponential 를 취해서 Odds 방식으로 해석하는게 더 편하다
# # 나이가 한살 많을수록록 대출할 확률이 1.024 높다.
# # 수입이 1단위 높을소룩 대출할 확률이 1.05배 높다
# # 가족 구성원수가 1많을수록 대출할 확률이 2.13배 높다
# # 경력이 1단위 높을수록 대출할 확률이 0.99배 높다(귀무가설 채택)
# # Experience,  Mortgage는 제외할 필요성이 있어보임
print(np.exp(results.params))

# y_hat 예측
# logistic 함수는 확률값을 반환한다
# threshold 값을 설정할 필요가 있음
pred_y = results.predict(test_x)


print(pred_y)

# threshold 값을 기준으로 분류하는 함수
def cutoff(y, threshold):
    Y = y.copy()
    Y[Y > threshold] = 1
    Y[Y <= threshold] = 0
    return (Y.astype(int))


pred_Y = cutoff(pred_y, 0.5)
print(pred_Y)

# confusion matrix
# -> 예측값과 실제값을 나타낸 표
# 행: 실제값, 열: 예측값
cfmat = confusion_matrix(test_y, pred_Y)
print(cfmat)

# accuracy 계산하기
# -> (옳게 분류된 데이터 수)/(전체 데이터의 수)
accuracy = (cfmat[0, 0] + cfmat[1, 1]) / len(pred_Y)


print(accuracy)


# accuracy 계산하는 함수
def acc(cfmat):
    return (cfmat[0, 0] + cfmat[1, 1]) / (cfmat[0, 0] + cfmat[0, 1] + cfmat[1, 0] + cfmat[1, 1])


print(acc(cfmat))

# threshold 에 따른 성능지표 비교
# 0.5~0.6 사이의 값으로 설정하는 것이 성능이 제일 좋다
threshold = np.arange(0, 1, 0.1)
table = pd.DataFrame(columns=['ACC'])
for i in threshold:
    pred_Y = cutoff(pred_y, i)
    cfmat = confusion_matrix(test_y, pred_Y)
    table.loc[i] = acc(cfmat)
table.index.name = 'threshold'
table.columns.name = 'performance'
print(table)

# sklearn ROC 패키지
# fpr: false positive rate(가짜 양성률)
# tpr: true positive rate(진짜 양성률)
# threshold: fpr 과 tpr 을 계산하는데 사용한 threshold 값
# pos_lable parameter: positive 로 설정될 값
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y, pos_label=1)

# ROC curve 출력
# 커브가 왼쪽 위로 볼록할수록 성능이 좋은 모델이다
plt.plot(fpr, tpr)
plt.show()

# AUC 값 출력
# 1에 가까운 값이다
# np.trapz: trapezoidal rule(사다리꼴 공식)로 넓이를 구함
auc = np.trapz(tpr, fpr)
print(auc)

# p-value 값이 높게 나온 Experience, Mortgage 변수를 제거하고 접합
feature_columns = list(ploan_processed.columns.difference(['Personal Loan', 'Experience', 'Mortgage']))
X = ploan_processed[feature_columns]
y = ploan_processed['Personal Loan']

train_x2, test_x2, train_y, test_y = train_test_split(X, y, stratify=y, train_size=0.7, test_size=0.3,
                                                      random_state=42)
print(train_x2.shape, test_x2.shape, train_y.shape, test_y.shape)

# 로지스틱 모델 접합
model = sm.Logit(train_y, train_x2)
result2 = model.fit(method='newton')

# 이전 모델과 비교
# 다른 변수들은 값이 크게 변하지 않았다
# -> Experience, Mortgage 는 다른 변수들과 다중공선성이 거의 없다
# -> 또한 y 값을 예측하는데 큰 도움이 되지 않는다
print(results.summary())
print(result2.summary())

# 예측
pred_y = result2.predict(test_x2)

# threshold = 0.5
pred_Y2 = cutoff(pred_y, 0.5)

# confusion matrix
# 변수를 제거해도 검증데이터의 성능은 크게 증가하지 않았다
cfmat = confusion_matrix(test_y, pred_Y2)
print(acc(cfmat))

# 아까와 비슷하게 0.5~0.6 사이에서 가장 높은 성능을 낸다
threshold = np.arange(0, 1, 0.1)
table = pd.DataFrame(columns=['ACC'])
for i in threshold:
    pred_Y2 = cutoff(pred_y, i)
    cfmat = confusion_matrix(test_y, pred_Y2)
    table.loc[i] = acc(cfmat)
table.index.name = 'threshold'
table.columns.name = 'performance'
print(table)

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_Y2, pos_label=1)
plt.plot(fpr, tpr)
auc = np.trapz(tpr, fpr)

# ##########변수선택법##########
# 다중회귀분석에서 한 원리와 다르지 않음
feature_columns = list(ploan_processed.columns.difference(["Personal Loan"]))
X = ploan_processed[feature_columns]
y = ploan_processed['Personal Loan']  # 대출여부: 1 or 0

train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y, train_size=0.7, test_size=0.3, random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)


def processSubset(X, y, feature_set):
    model = sm.Logit(y, X[list(feature_set)])
    regr = model.fit()
    AIC = regr.aic
    return {"model": regr, "AIC": AIC}


# 전진선택법
def forward(X, y, predictors):
    # 데이터 변수들이 미리정의된 predictors에 있는지 없는지 확인 및 분류
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


def forward_model(X, y):
    Fmodels = pd.DataFrame(columns=["AIC", "model"])
    tic = time.time()
    # 미리 정의된 데이터 변수
    predictors = []
    # 변수 1~10개 : 0~9 -> 1~10
    for i in range(1, len(X.columns.difference(['const'])) + 1):
        Forward_result = forward(X=X, y=y, predictors=predictors)
        if i > 1:
            if Forward_result['AIC'] > Fmodel_before:
                break
        Fmodels.loc[i] = Forward_result
        predictors = Fmodels.loc[i]["model"].model.exog_names
        Fmodel_before = Fmodels.loc[i]["AIC"]
        predictors = [k for k in predictors if k != 'const']
    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")

    return (Fmodels['model'][len(Fmodels['model'])])


# 후진소거법
def backward(X, y, predictors):
    tic = time.time()
    results = []

    # 데이터 변수들이 미리정의된 predictors 조합 확인
    for combo in itertools.combinations(predictors, len(predictors) - 1):
        results.append(processSubset(X=X, y=y, feature_set=list(combo) + ['const']))
    models = pd.DataFrame(results)

    # 가장 낮은 AIC 를 가진 모델을 선택
    best_model = models.loc[models['AIC'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors) - 1, "predictors in",
          (toc - tic))
    print('Selected predictors:', best_model['model'].model.exog_names, ' AIC:', best_model[0])
    return best_model


def backward_model(X, y):
    Bmodels = pd.DataFrame(columns=["AIC", "model"], index=range(1, len(X.columns)))
    tic = time.time()
    predictors = X.columns.difference(['const'])
    Bmodel_before = processSubset(X, y, predictors)['AIC']
    while (len(predictors) > 1):
        Backward_result = backward(X=train_x, y=train_y, predictors=predictors)
        if Backward_result['AIC'] > Bmodel_before:
            break
        Bmodels.loc[len(predictors) - 1] = Backward_result
        predictors = Bmodels.loc[len(predictors) - 1]["model"].model.exog_names
        Bmodel_before = Backward_result['AIC']
        predictors = [k for k in predictors if k != 'const']

    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")
    return (Bmodels['model'].dropna().iloc[0])


# 단계적 선택법
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
        if Stepmodels.loc[i]['AIC'] > Smodel_before:
            break
        else:
            Smodel_before = Stepmodels.loc[i]["AIC"]
    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")
    return (Stepmodels['model'][len(Stepmodels['model'])])


Forward_best_model = forward_model(X=train_x, y=train_y)
Backward_best_model = backward_model(X=train_x, y=train_y)
Stepwise_best_model = Stepwise_model(X=train_x, y=train_y)

# 로지스틱 회귀 접합
pred_y_full = results.predict(test_x)  # full model
pred_y_forward = Forward_best_model.predict(test_x[Forward_best_model.model.exog_names])
pred_y_backward = Backward_best_model.predict(test_x[Backward_best_model.model.exog_names])
pred_y_stepwise = Stepwise_best_model.predict(test_x[Stepwise_best_model.model.exog_names])

# threshold 기준으로 값 나눔
pred_Y_full = cutoff(pred_y_full, 0.5)
pred_Y_forward = cutoff(pred_y_forward, 0.5)
pred_Y_backward = cutoff(pred_y_backward, 0.5)
pred_Y_stepwise = cutoff(pred_y_stepwise, 0.5)

# confusion matrix 만들기
cfmat_full = confusion_matrix(test_y, pred_Y_full)
cfmat_forward = confusion_matrix(test_y, pred_Y_forward)
cfmat_backward = confusion_matrix(test_y, pred_Y_backward)
cfmat_stepwise = confusion_matrix(test_y, pred_Y_stepwise)

# 각각의 accuracy
print('accuracy of full: ', acc(cfmat_full))
print('accuracy of forward: ', acc(cfmat_forward))
print('accuracy of backward: ', acc(cfmat_backward))
print('accuracy of stepwise: ', acc(cfmat_stepwise))

# ROC curve 출력
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y_full, pos_label=1)
# Print ROC curve
plt.plot(fpr, tpr)
plt.show()
# Print AUC
auc = np.trapz(tpr, fpr)
print('AUC:', auc)


fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y_forward, pos_label=1)
# Print ROC curve
plt.plot(fpr, tpr)
plt.show()
# Print AUC
auc = np.trapz(tpr, fpr)
print('AUC:', auc)

fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y_backward, pos_label=1)
# Print ROC curve
plt.plot(fpr, tpr)
plt.show()
# Print AUC
auc = np.trapz(tpr, fpr)
print('AUC:', auc)

fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y_stepwise, pos_label=1)
# Print ROC curve
plt.plot(fpr, tpr)
plt.show()
# Print AUC
auc = np.trapz(tpr, fpr)
print('AUC:', auc)

# 성능면에서는 네 모델이 큰 차이가 없음
print(len(Forward_best_model.model.exog_names))
print(len(Backward_best_model.model.exog_names))
print(len(Stepwise_best_model.model.exog_names))
