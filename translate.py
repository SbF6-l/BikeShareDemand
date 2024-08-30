
"""
Bike Sharing Demand 进行方向
1) 训练和测试数据集的形式以及列的属性数据值的了解
2) 数据预处理和可视化
3) 应用回归模型
4) 得出结论

函数使用时的小贴士

当应用函数时，如果不知道内部参数，可以利用 Anaconda Prompt 或 Windows PowerShell 中的 REPL python 命令行
例如：如果想了解 pandas.to_numeric() 函数的内部参数
可以使用 help(pandas.to_numeric) 来查看函数的使用方法等文档
=> 我非常常用这个方法!!

"""

"""需要的库调用"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 
import seaborn as sns #用于可视化的库
import matplotlib.pyplot as plt
import calendar 
from datetime import datetime

import os
print(os.listdir("../input"))

"""
1) 训练和测试数据集的概况以及数据的列属性和数据值的数量
"""

#加载训练数据和测试数据集
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#概述训练数据集的情况
train.head()

#数据集中的列属性说明

"""
datetime - 每小时日期 + 时间戳  
season -  1 = 春季, 2 = 夏季, 3 = 秋季, 4 = 冬季 
holiday - 是否为假日
workingday - 是否为工作日（既不是周末也不是假日）
weather - 1: 晴天, 少量云, 部分多云, 部分多云 
2: 薄雾 + 多云, 薄雾 + 碎云, 薄雾 + 少量云, 薄雾 
3: 小雪, 小雨 + 雷暴 + 零星云, 小雨 + 零星云 
4: 大雨 + 冰雹 + 雷暴 + 薄雾, 雪 + 雾 
temp - 摄氏温度
atemp - "体感" 温度（摄氏度）
humidity - 相对湿度
windspeed - 风速
casual - 非注册用户租赁数量
registered - 注册用户租赁数量
count - 总租赁数量
"""

#查看训练数据集每列的数据类型和数据值的数量
train.info()

#输出测试数据集的概况
test.head()

""" 2) 数据预处理和可视化 """

#使用 split 函数将 datetime 属性拆分为年-月-日 和 时间
train['tempDate'] = train.datetime.apply(lambda x:x.split())

#使用拆分出的 tempDate 提取出年-月-日，从中提取 year, month, day 和 weekday 列。
# split() 内置函数说明: https://wikidocs.net/13 [字符数据类型_ 字符串拆分] <=> join() [字符数据类型_ 字符串插入]
train['year'] = train.tempDate.apply(lambda x:x[0].split('-')[0])
train['month'] = train.tempDate.apply(lambda x:x[0].split('-')[1])
train['day'] = train.tempDate.apply(lambda x:x[0].split('-')[2])
# weekday 使用 calendar 包和 datetime 包
# calendar.day_name 用法: https://stackoverflow.com/questions/36341484/get-day-name-from-weekday-int
# datetime.strptime 文档: https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
# Python 日期和时间处理: https://datascienceschool.net/view-notebook/465066ac92ef4da3b0aba32f76d9750a/ 
train['weekday'] = train.tempDate.apply(lambda x:calendar.day_name[datetime.strptime(x[0],"%Y-%m-%d").weekday()])

train['hour'] = train.tempDate.apply(lambda x:x[1].split(':')[0])

#提取出的属性是字符串属性，因此需要转换为数值型数据。
# pandas.to_numeric(): https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_numeric.html
train['year'] = pd.to_numeric(train.year,errors='coerce')
train['month'] = pd.to_numeric(train.month,errors='coerce')
train['day'] = pd.to_numeric(train.day,errors='coerce')
train['hour'] = pd.to_numeric(train.hour,errors='coerce')

#可以看到 year, month, day, hour 已经被转换为数值型数据。
train.info()

#删除已无用的 tempDate 列
train = train.drop('tempDate',axis=1)

#分析每个属性与预测结果 count 的关系

# 年份与 count
fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.barplot(x='year',y='count',data=train.groupby('year')['count'].mean().reset_index())

# 月份与 count
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.barplot(x='month',y='count',data=train.groupby('month')['count'].mean().reset_index())

# 日期与 count
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.barplot(x='day',y='count',data=train.groupby('day')['count'].mean().reset_index())

# 小时与 count
ax4 = fig.add_subplot(2,2,4)
ax4 = sns.barplot(x='hour',y='count',data=train.groupby('hour')['count'].mean().reset_index())

# 季节与 count
fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.barplot(x='season',y='count',data=train.groupby('season')['count'].mean().reset_index())

# 假日与 count
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.barplot(x='holiday',y='count',data=train.groupby('holiday')['count'].mean().reset_index())

# 工作日与 count
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.barplot(x='workingday',y='count',data=train.groupby('workingday')['count'].mean().reset_index())

# 天气与 count
ax4 = fig.add_subplot(2,2,4)
ax4 = sns.barplot(x='weather',y='count',data=train.groupby('weather')['count'].mean().reset_index())

"""
该部分由于笔者自己在查看数据时感到异常，因此进行了预处理。
因为在最初导入的数据集中，1月1日的季节列是1，即春季，
但实际上1月是冬季。为了纠正这个错误，使用了下面的 badToRight 函数来修正季节列。
这可能会导致与参考的内核的准确性有所不同。
"""

def badToRight(month):
    if month in [12,1,2]:
        return 4
    elif month in [3,4,5]:
        return 1
    elif month in [6,7,8]:
        return 2
    elif month in [9,10,11]:
        return 3

# apply() 内置函数是必须掌握的函数之一，类似于 split(), map(), join(), filter() 等。
train['season'] = train.month.apply(badToRight)

#与之前的可视化一样，比较一个列与结果值的关系

# 季节与 count
fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.barplot(x='season',y='count',data=train.groupby('season')['count'].mean().reset_index())

# 假日与 count
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.barplot(x='holiday',y='count',data=train.groupby('holiday')['count'].mean().reset_index())

# 工作日与 count
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.barplot(x='workingday',y='count',data=train.groupby('workingday')['count'].mean().reset_index())

# 天气与 count
ax4 = fig.add_subplot(2,2,4)
ax4 = sns.barplot(x='weather',y='count',data=train.groupby('weather')['count'].mean().reset_index())

#通过剩余的分布图，比较好的列与 count 的关系

# 温度与 count
fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.distplot(train.temp,bins=range(train.temp.min().astype('int'),train.temp.max().astype('int')+1))

# 平均温度与 count
ax2 = fig.add_subplot(2,2,2

)
ax2 = sns.distplot(train.atemp,bins=range(train.atemp.min().astype('int'),train.atemp.max().astype('int')+1))

# 湿度与 count
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.distplot(train.humidity,bins=range(train.humidity.min().astype('int'),train.humidity.max().astype('int')+1))

# 风速与 count
ax4 = fig.add_subplot(2,2,4)
ax4 = sns.distplot(train.windspeed,bins=range(train.windspeed.min().astype('int'),train.windspeed.max().astype('int')+1))

#通过热图展示各列之间的相关系数

fig = plt.figure(figsize=[20,20])
ax = sns.heatmap(train.corr(),annot=True,square=True)

#根据热图中的相关性，与之前的可视化不同，展示两个不同列对 count 的影响

# 时间和季节对 count 的影响
fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.pointplot(x='hour',y='count',hue='season',data=train.groupby(['season','hour'])['count'].mean().reset_index())

# 时间和假日对 count 的影响
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.pointplot(x='hour',y='count',hue='holiday',data=train.groupby(['holiday','hour'])['count'].mean().reset_index())

# 时间和星期几对 count 的影响
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.pointplot(x='hour',y='count',hue='weekday',hue_order=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],data=train.groupby(['weekday','hour'])['count'].mean().reset_index())

# 时间和天气对 count 的影响
ax4 = fig.add_subplot(2,2,4)
ax4 = sns.pointplot(x='hour',y='count',hue='weather',data=train.groupby(['weather','hour'])['count'].mean().reset_index())

#最后检查是否存在异常值

train[train.weather==4]

#月份和天气对 count 的影响
fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,1,1)
ax1 = sns.pointplot(x='month',y='count',hue='weather',data=train.groupby(['weather','month'])['count'].mean().reset_index())

#每月 count
ax2 = fig.add_subplot(2,1,2)
ax2 = sns.barplot(x='month',y='count',data=train.groupby('month')['count'].mean().reset_index())

"""
在风速分布图中，发现风速为0的值非常多，
这可能是实际风速为0或者测量值为0的两种情况。
考虑到后一种情况，使用我们的数据来填充 windspeed 值。
"""
# 训练机器学习模型时，字符串值是不被允许的，因此需要将字符串分类，并将其转换为数字值
train['weekday']= train.weekday.astype('category')

print(train['weekday'].cat.categories)

# 0: Sunday --> 6: Saturday
train.weekday.cat.categories = ['5','1','6','0','4','2','3']

"""
利用随机森林（RandomForest）填充风速（Windspeed）值
将数据分为风速为0的和风速不为0的数据集，
在风速不为0的数据集中，提取出风速作为目标值（Series）以及其他特征数据（DataFrame）进行训练，
然后在风速为0的数据集中提取相同的特征数据，使用训练好的模型进行预测，最后将预测结果填入原数据集中风速为0的部分。
"""
from sklearn.ensemble import RandomForestRegressor

# 风速为0的数据集
windspeed_0 = train[train.windspeed == 0]
# 风速不为0的数据集
windspeed_Not0 = train[train.windspeed != 0]

# 从风速为0的数据集中去除不需要的列
windspeed_0_df = windspeed_0.drop(['windspeed','casual','registered','count','datetime'],axis=1)

# 从风速不为0的数据集中去除不需要的列，保留风速列作为目标值
windspeed_Not0_df = windspeed_Not0.drop(['windspeed','casual','registered','count','datetime'],axis=1)
windspeed_Not0_series = windspeed_Not0['windspeed'] 

# 训练随机森林模型
rf = RandomForestRegressor()
rf.fit(windspeed_Not0_df,windspeed_Not0_series)
# 使用训练好的模型预测风速为0的数据集中的风速
predicted_windspeed_0 = rf.predict(windspeed_0_df)
# 将预测结果填入原数据集中
windspeed_0['windspeed'] = predicted_windspeed_0

# 恢复原始数据集
train = pd.concat([windspeed_0,windspeed_Not0],axis=0)

# 将字符串类型的 datetime 转换为 datetime 类型
train.datetime = pd.to_datetime(train.datetime,errors='coerce')

# 按 datetime 排序
train = train.sort_values(by=['datetime'])

# 修正风速后重新分析相关系数
# 与预期不同，风速与 count 的相关性仅在 0.1 到 0.11 之间变化不大
fig = plt.figure(figsize=[20,20])
ax = sns.heatmap(train.corr(),annot=True,square=True)

fig = plt.figure(figsize=[5,5])
sns.distplot(train['windspeed'],bins=np.linspace(train['windspeed'].min(),train['windspeed'].max(),10))
plt.suptitle("Filled by Random Forest Regressor")
print("Min value of windspeed is {}".format(train['windspeed'].min()))

"""现在对测试集进行相同的数据预处理"""
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

combine = pd.concat([train,test],axis=0)

combine.info()

combine['tempDate'] = combine.datetime.apply(lambda x:x.split())
combine['weekday'] = combine.tempDate.apply(lambda x: calendar.day_name[datetime.strptime(x[0],"%Y-%m-%d").weekday()])
combine['year'] = combine.tempDate.apply(lambda x: x[0].split('-')[0])
combine['month'] = combine.tempDate.apply(lambda x: x[0].split('-')[1])
combine['day'] = combine.tempDate.apply(lambda x: x[0].split('-')[2])
combine['hour'] = combine.tempDate.apply(lambda x: x[1].split(':')[0])

combine['year'] = pd.to_numeric(combine.year,errors='coerce')
combine['month'] = pd.to_numeric(combine.month,errors='coerce')
combine['day'] = pd.to_numeric(combine.day,errors='coerce')
combine['hour'] = pd.to_numeric(combine.hour,errors='coerce')

combine.info()

combine['season'] = combine.month.apply(badToRight)

combine.head()

combine.weekday = combine.weekday.astype('category')

combine.weekday.cat.categories = ['5','1','6','0','4','2','3']

dataWind0 = combine[combine['windspeed']==0]
dataWindNot0 = combine[combine['windspeed']!=0]

dataWind0.columns

dataWind0_df = dataWind0.drop(['windspeed','casual','registered','count','datetime','tempDate'],axis=1)

dataWindNot0_df = dataWindNot0.drop(['windspeed','casual','registered','count','datetime','tempDate'],axis=1)
dataWindNot0_series = dataWindNot0['windspeed']

dataWindNot0_df.head()

dataWind0_df.head()

rf2 = RandomForestRegressor()
rf2.fit(dataWindNot0_df,dataWindNot0_series)
predicted = rf2.predict(dataWind0_df)
print(predicted)

dataWind0['windspeed'] = predicted

combine = pd.concat([dataWind0,dataWindNot0],axis=0)

# 将分类列转换为分类数据类型，并删除不需要的列
categorizational_columns = ['holiday','humidity','season','weather','workingday','year','month','day','hour']
drop_columns = ['datetime','casual','registered','count','tempDate']

# 转换为分类数据类型
for col in categorizational_columns:
    combine[col] = combine[col].astype('category')

# 根据 count 列将数据集分为训练集和测试集，并按 datetime 排序
train = combine[pd.notnull(combine['count'])].sort_values(by='datetime')
test = combine[~pd.notnull(combine['count'])].sort_values(by='datetime')

# 训练时需要的标签值
datetimecol = test['datetime']
yLabels = train['count'] # count
yLabelsRegistered = train['registered'] # 注册用户
yLabelsCasual = train['casual'] # 临时用户

# 删除不需要的列后，得到训练集和测试集
train = train.drop(drop_columns,axis=1)
test = test.drop(drop_columns,axis=1)

"""
在这个问题中，使用 RMSLE（均方对数误差）来评估预测效果。
RMSLE 的使用方法参考链接：https://programmers.co.kr/learn/courses/21/lessons/943#

RMSLE
对高估的预测项惩罚较小，而对低估的预测项惩罚较大。
通过对误差进行平方，取均值，再开方，得到的值越小，精度越高。
结果接近0时表示精度更高。
"""
# y 是预测值，y_ 是实际值
def rmsle(y, y_, convertExp=True):
    if convertExp:
        y = np.exp(y)
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

# 线性回归模型
# 线性回归模型没有需要调整的内部属性
from sklearn.linear_model import LinearRegression, Ridge, Lasso

lr = LinearRegression()

"""
为什么使用 np.log1p 而不是 np.log 来对 yLabels 进行对数变换？
np.log1p 与 np.log(1+x) 等效。原因是如果 x 为0，直接使用 np.log 会导致结果趋近于负无穷，因此使用 np.log1p 更为稳健。
参考链接: https://ko.wikipedia.org/wiki/%EB%A1%9C%EA%B7%B8 
"""
yLabelslog = np.log1p(yLabels)
# 使用线性模型进行训练
lr.fit(train, yLabelslog)
# 预测结果
preds = lr.predict(train)
# rmsle 函数中的 np.exp() 是因为我们的 preds 是经过对数变换的结果，需要还原到原始值
print('RMSLE Value For Linear Regression: {}'.format(rmsle(np.exp(yLabelslog), np.exp(preds), False)))

"""
为什么在训练数据时使用对数值？
因为 count 值的范围非常大，如果不进行对数变换，结果可能会出现无限大（inf）。
"""

# count 值的分布
sns.distplot(yLabels, bins=range(yLabels.min().astype('int'), yLabels.max().astype('int')))

# 原训练数据集中 count 的数量
print(yLabels.count()) #10886

""" 
使用 3 sigma 方法检测异常值
参考链接: https://ko.wikipedia.org/wiki/68-95-99.7_%EA%B7%9C%EC%B9%99
"""
# 使用 3 sigma 方法检测异常值后的 count 数量
yLabels[np.logical_and(yLabels.mean() - 3 * yLabels.std() <= yLabels, yLabels.mean() + 3 * yLabels.std() >= yLabels)].count() #10739
# 当存在异常值时，使用对数变换进行处理

"""
使用 GridSearchCV 可以帮助我们找到最佳参数配置

GridSearchCV 参考链接:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
https://datascienceschool.net/view-notebook/ff4b5d491cc34f94aea04baca86fbef8/
"""
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# Ridge 模型是具有 L2 正则化的线性回归模型，主要调整 alpha 参数
ridge = Ridge()

# 调整 Ridge 模型的参数，其中特定参数的数组值会被测试，以找到最佳配置
ridge_params = {'max_iter': [3000], 'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)
grid_ridge = GridSearchCV(ridge, ridge_params, scoring=rmsle_scorer, cv=5)

grid_ridge.fit(train, yLabelslog)
preds = grid_ridge.predict(train)
print(grid_ridge.best_params_)
print('RMSLE Value for Ridge Regression {}'.format(rmsle(np.exp(yLabelslog), np.exp(preds), False)))

# 可以通过 grid_ridge 的 cv_results_ 变量查看不同 alpha 值下的平均得分
df = pd.DataFrame(grid_ridge.cv_results_)

df.head()

# Lasso 模型是具有 L1 正则化的线性回归模型，主要调整 alpha 参数
lasso = Lasso()

lasso_params = {'max_iter': [3000], 'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_lasso = GridSearchCV(lasso, lasso_params, scoring=rmsle_scorer, cv=5)
grid_lasso.fit(train, yLabelslog)
preds = grid_lasso.predict(train)
print('RMSLE Value for Lasso Regression {}'.format(rmsle(np.exp(yLabelslog), np.exp(preds), False)))

rf = RandomForestRegressor()

rf_params = {'n_estimators': [1, 10, 100]}
grid_rf = GridSearchCV(rf, rf_params, scoring=rmsle_scorer, cv=5)
grid_rf.fit(train, yLabelslog)
preds = grid_rf.predict(train)
print('RMSLE Value for RandomForest {}'.format(rmsle(np.exp(yLabelslog), np.exp(preds), False)))

from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor()
gb_params = {'max_depth': range(1, 11, 1), 'n_estimators': [1, 10, 100]}
grid_gb = GridSearchCV(gb, gb_params, scoring=rmsle_scorer, cv=5)
grid_gb.fit(train, yLabelslog)
preds = grid_gb.predict(train)
print('RMSLE Value for GradientBoosting {}'.format(rmsle(np.exp(yLabelslog), np.exp(preds), False)))

predsTest = grid_gb.predict(test)
fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.set_size_inches(12, 5)
sns.distplot(yLabels, ax=ax1, bins=50)
sns.distplot(np.exp(predsTest), ax=ax2, bins=50)

submission = pd.DataFrame({
        "datetime": datetimecol,
        "count": [max(0, x) for x in np.exp(predsTest)]
    })
submission.to_csv('bike_predictions_gbm_separate_without_fe.csv', index=False)
