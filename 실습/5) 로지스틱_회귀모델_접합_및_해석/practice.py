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
# print(ploan.head())
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
# print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

# 로지스틱 회귀모형 모델링 y=f(x)
# 로지스틱 모형 접합
# sm.Logit 함수 사용
# method='newton': 뉴턴법을 사용해서 접합
model = sm.Logit(train_y, train_x)
results = model.fit(method='newton')
# print(results.summary())

# 회귀계수 출력
# params 를 통해 출력하는것은 Logit 방식
# print(results.params)

# exponential 를 취해서 Odds 방식으로 해석하는게 더 편하다
# # 나이가 한살 많을수록록 대출할 확률이 1.024 높다.
# # 수입이 1단위 높을소룩 대출할 확률이 1.05배 높다
# # 가족 구성원수가 1많을수록 대출할 확률이 2.13배 높다
# # 경력이 1단위 높을수록 대출할 확률이 0.99배 높다(귀무가설 채택)
# # Experience,  Mortgage는 제외할 필요성이 있어보임
# print(np.exp(results.params))

# y_hat 예측
# logistic 함수는 확률값을 반환한다
# threshold 값을 설정할 필요가 있음
pred_y = results.predict(test_x)


# print(pred_y)

# threshold 값을 기준으로 분류하는 함수
def cutoff(y, threshold):
    Y = y.copy()
    Y[Y > threshold] = 1
    Y[Y <= threshold] = 0
    return (Y.astype(int))


pred_Y = cutoff(pred_y, 0.5)
# print(pred_Y)

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
