import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import minmax_scale


mpl.rcParams.update(mpl.rcParamsDefault)

df = pd.read_csv('./tax.csv')
df.columns = ['year', 'T', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
df.head()

yr = df['year'].tolist()
TAX = df['T'].tolist()

# plt.plot(yr, TAX)
# plt.xticks(yr,rotation=60)
# plt.title('Tax Revenue during 1997-2017')
# plt.xlabel('Year')
# plt.ylabel('Tax Revenue')
# plt.show()
# plt.close()

# calculate the relative coefficience between T and explain parameters

corr_pearson = df.corr(method='pearson')['year']
corr_kendall = df.corr(method='kendall')['year']
corr_spearman = df.corr(method='spearman')['year']
# X4, X5 are not relevant

# MLR
y = df['T'].to_numpy()
x = df[['X1', 'X2', 'X3', 'X6', 'X7', 'X8']].to_numpy()

# normalization
# scale the output between 0-1
# minmax_scale(x)
# minmax_scale(y)
x_norm = x
y_norm = y

# Multi Linear Regression
reg = LinearRegression()
reg.fit(x_norm, y_norm)
cof_tax = reg.coef_
pre = reg.predict(x_norm)

fig, ax = plt.subplots()
plt.plot(yr, y_norm)
plt.plot(yr, pre)
plt.legend(['TAX', 'mlr_fit'])
plt.xticks(yr,rotation=60)
plt.title('Tax Revenue during 1997-2017')
plt.xlabel('Year')
plt.ylabel('Tax Revenue')
plt.grid()
plt.show()
plt.close()

# SVR

# 线性核函数配置支持向量机
linear_svr = SVR(kernel="poly")
# 训练
linear_svr.fit(x_norm, y_norm)
# 预测 保存预测结果
linear_svr_y_predict = linear_svr.predict(x_norm)

fig, ax = plt.subplots()
plt.plot(yr, y_norm)
plt.plot(yr, linear_svr_y_predict)
plt.legend(['TAX', 'fit'])
plt.xticks(yr, rotation=60)
plt.title('Tax Revenue during 1997-2017 by poly svr')
plt.xlabel('Year')
plt.ylabel('Tax Revenue')
plt.grid()
plt.show()
# svr_reg = SVR(kernel='rbf', gamma=0.002, C=101, epsilon=0.001)
# svr_reg.fit(x, y)
# svr_pre = svr_reg.predict(x)
# fig, ax = plt.subplots()
# plt.plot(yr, y_norm)
# plt.plot(yr, svr_pre)
# plt.legend(['TAX', 'svr_fit'])
# plt.xticks(yr, rotation=60)
# plt.title('Tax Revenue during 1997-2017')
# plt.xlabel('Year')
# plt.ylabel('Tax Revenue')
# plt.grid()
# plt.show()
# plt.close()


# sns.set(style='whitegrid', context='notebook')
# cols = ['T', 'X1', 'X2',
#              'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
# sns.pairplot(df[cols], size=2.5)
# plt.show()

# cm = np.corrcoef(df[cols].values.T)
# sns.set(font_scale=1.5)
# hm = sns.heatmap(cm,
#                 cbar=True,
#                 annot=True,
#                 square=True,
#                 fmt='.2f',
#                 annot_kws={'size': 12},
#                 yticklabels=cols,
#                 xticklabels=cols)
# plt.show()