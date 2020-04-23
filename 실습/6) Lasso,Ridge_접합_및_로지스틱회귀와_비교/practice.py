import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet


# threshold 값을 기준으로 분류하는 함수
def cutoff(y, threshold):
    Y = y.copy()
    Y[Y > threshold] = 1
    Y[Y <= threshold] = 0
    return Y.astype(int)


# accuracy 계산하는 함수
def acc(cfmat):
    return (cfmat[0, 0] + cfmat[1, 1]) / (cfmat[0, 0] + cfmat[0, 1] + cfmat[1, 0] + cfmat[1, 1])


ploan = pd.read_csv('Personal Loan.csv')

# 데이터 전처리
ploan_processed = ploan.dropna().drop(['ID', 'ZIP Code'], axis=1, inplace=False)

feature_columns = list(ploan_processed.columns.difference(['Personal Loan']))
X = ploan_processed[feature_columns]
y = ploan_processed['Personal Loan']

train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y, train_size=0.7, test_size=0.3, random_state=42)

# 정규화: 항상 필요있는 변수를 남기고 필요없는 변수들은 0으로 보내지는 않는다
#        선형회귀에서 p-value 가 좋게 나왔던 변수들이 없어지는 경우도 있음
# logistic regression 문제에도 정규화를 적용할 수 있다

# #Lasso 접합
# alpha: 람다값
# alpha 값이 커질수록 모든 계수들이 0으로 수렴
ll = Lasso(alpha=0.01)
ll.fit(train_x, train_y)

# 회귀계수 출력
# Lasso 접합이기 때문에 계수가 아예 0이 되어버린다
# 일반 회귀모델과는 다르게 정규화는 summary 함수를 지원하지 않는다
# 상수항을 추가하지 X
print(ll.coef_)

# test 데이터 학습
# logistic regression 에 대한 문제이니 cutoff 함수 적용
pred_y_lasso = ll.predict(test_x)
pred_Y_lasso = cutoff(pred_y_lasso, 0.5)
cfmat = confusion_matrix(test_y, pred_Y_lasso)
print('Accuracy of Lasso:', acc(cfmat))

# plot roc curve
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y_lasso, pos_label=1)
plt.plot(fpr, tpr)
# plt.show()

# 정규화를 사용하지 않았을 때보다 auc 값이 감소했다(성능이 안좋아졌다)
# 너무 많은 변수를 0으로 만든것이 원인일수도
auc = np.trapz(tpr, fpr)
print('AUC of Lasso:', auc)

# #Ridge 접합
rr = Ridge(alpha=0.01)
rr.fit(train_x, train_y)

# 회귀계수 출력
# 같은 alpha(람다)값을 주었지만 ridge 의 회귀계수는 0으로 가지 않는다
# Lasso 를 통한 회귀계수보다는 큰 경향
print(rr.coef_)

# test 데이터 학습 및 accuracy 계산
# acc 값은 줄어들었다
pred_y_ridge = rr.predict(test_x)
pred_Y_ridge = cutoff(pred_y_ridge, 0.5)
cmat = confusion_matrix(test_y, pred_Y_ridge)
print('Accuracy of Ridge:', acc(cmat))

# plot roc curve
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y_ridge, pos_label=1)
plt.plot(fpr, tpr)
# plt.show()

# auc 값은 Lasso 보다 증가
auc = np.trapz(tpr, fpr)
print('AUC of Ridge:', auc)

# #lambda 값에 따른 회귀계수 / accuracy 계산
# 10^-3부터 10까지 log 단위로 5개구간 나눔
alpha = np.logspace(-3, 1, 5)

# #Lasso
# lambda 값 0.001~10까지 범위설정
# enumerate: 리스트의 순서와 값을 전달
# lasso.intercept_: 회귀모델의 상수항
# lasso.coef_: 회귀모델의 변수
# np.hstack: 가로로 증가하는 stack(column 방향)
data = []
acc_table = []
for i, a in enumerate(alpha):
    lasso = Lasso(alpha=a).fit(train_x, train_y)
    data.append(pd.Series(np.hstack([lasso.intercept_, lasso.coef_])))
    pred_y = lasso.predict(test_x)
    pred_Y = cutoff(pred_y, 0.5)
    cfmat = confusion_matrix(test_y, pred_Y)
    acc_table.append((acc(cfmat)))

# alpha 가 증가할수록 계수가 0으로 수렴하는 변수들이 많다
df_lasso = pd.DataFrame(data, index=alpha).T
print('Lasso', df_lasso, sep='\n')

# accuracy 도 alpha  0.01일때가 가장 높다
acc_table_lasso = pd.DataFrame(acc_table, index=alpha).T
print('acc of Lasso', acc_table_lasso, sep='\n')

# #Ridge
# lambda 값 0.001~10까지 범위설정
# enumerate: 리스트의 순서와 값을 전달
data = []
acc_table = []
for i, a in enumerate(alpha):
    ridge = Ridge(alpha=a).fit(train_x, train_y)
    data.append(pd.Series(np.hstack([ridge.intercept_, ridge.coef_])))
    pred_y = ridge.predict(test_x)
    pred_Y = cutoff(pred_y, 0.5)
    cfmat = confusion_matrix(test_y, pred_Y)
    acc_table.append((acc(cfmat)))

# alpha 가 증가할수록 계수가 0으로 수렴하는 변수들이 많다
df_ridge = pd.DataFrame(data, index=alpha).T
print('Ridge', df_ridge, sep='\n')

# accuracy 는 거의 비슷
# alpha 값을 증가시켜도 계수들은 큰 변화가 없다
# Ridge 로는 더이상 축소할 계수가 없다는 의미
acc_table_ridge = pd.DataFrame(acc_table, index=alpha).T
print('acc of Ridge', acc_table_ridge, sep='\n')

# #lambda 값의 변화에 따른 회귀계수 축소 시각화
# 람다값이 증가하면서 계수들이 0으로 가까워지는것을 확인할 수 있다
# Ridge 의 감소량은 매우 작은데, 데이터에 따라 이렇게 나올수도 있다
ax1 = plt.subplot(1, 2, 1)
plt.semilogx(df_ridge.T)
plt.xticks(alpha)
plt.title('Ridge')

ax2 = plt.subplot(1, 2, 2)
plt.semilogx(df_lasso.T)
plt.xticks(alpha)
plt.title('Lasso')

plt.show()

# 결론: 이 데이터의 경우에는 변수가 그렇게 많지 않기 때문에 정규화 기법을 사용하는 것은 큰 효과가 없을수도 있다
#       p-value 가 높은 변수들을 직접 제거하거나, 변수선택법을 사용하는것이 더 효과가 좋을수도
#       하지만 변수의 수가 매우 많은 경우라면 정규화 기법을 사용하는게 더 효과적일수도 있다
