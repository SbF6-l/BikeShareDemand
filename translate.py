
"""
Bike Sharing Demand 进行方向
1) 训练和测试数据集的形式以及列的属性数据值的了解
2) 数据预处理和可视化
3) 应用回归模型
4) 得出结论

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

#加载训练数据和测试数据集
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

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
ax1 = sns.histplot(train['temp'], bins=range(train['temp'].min().astype('int'), train['temp'].max().astype('int') + 1))

# 平均温度与 count
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.histplot(train['atemp'],bins=range(train['atemp'].min().astype('int'),train['atemp'].max().astype('int')+1))

# 湿度与 count
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.histplot(train['humidity'],bins=range(train['humidity'].min().astype('int'),train['humidity'].max().astype('int')+1))

# 风速与 count
ax4 = fig.add_subplot(2,2,4)
ax4 = sns.histplot(train['windspeed'],bins=range(train['windspeed'].min().astype('int'),train['windspeed'].max().astype('int')+1))

#通过热图展示各列之间的相关系数

fig = plt.figure(figsize=[20,20])
ax = sns.heatmap(train.drop(columns=['datetime','weekday']).corr(),annot=True,square=True)

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

