import os
import pandas as pd

# 用脚本所在位置定位数据
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "bike+sharing+dataset", "day.csv"))  # 把表格读进来，存到一个叫 df 的表里
print(df.head() ) #打印前五行
df.info()  #看看每列是什么类型、有没有空值
print (df['cnt'].describe()) #看总用车辆这一列的统计指标

#检查是否有缺失值
missing_data = df.isnull().sum()
print("缺失值:",missing_data)

#检查是否有重复值
duplicate_data = df.duplicated().sum()
print("重复值:",duplicate_data)

#逐月总量时序图
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


#画出逐月cnt时序曲线图
#先将yr列的0和1映射为2011和2012
df['year_label'] = df['yr'].map({0:'2011',1:"2012"})
#分组聚合，做数据透视表
monthly_data = df.groupby(['mnth','year_label'])['cnt'].sum().unstack()

#设置画布大小
plt.figure(figsize=(10,6)) #宽10高6
#用seaborn画两条折线
sns.lineplot(data=monthly_data,markers=True,linewidth=2)

#添加图表细节
plt.title('2011-2012 cnt distribution monthly by year',fontsize=15)
plt.xlabel('month',fontsize=12)
plt.ylabel('cnt',fontsize=12)
plt.xticks(range(1,13))
plt.grid(True,linestyle='--',alpha=0.6)
plt.legend(title='year')
plt.tight_layout()
plt.show()

#各季节分布箱线图
season_map = {1:'Spring', 2:'Summer', 3:'Autumn', 4:'winter'}
df['season_map'] = df['season'].map(season_map)
plt.figure(figsize=(10,6))
sns.boxplot(data=df,x='season_map',y='cnt',hue='year_label',palette='Set2')

plt.title('2011-2012 cnt by season',fontsize=15)
plt.xlabel('season',fontsize=12)
plt.ylabel('cnt',fontsize=12)
plt.legend(title='year')
plt.grid(axis='y',linestyle='--',alpha=0.7)
plt.tight_layout()
plt.show()

#工作日和节假日分布图
work_map = {0:"nowork", 1:"work"}
df['work_map'] = df['workingday'].map(work_map)
work_stats = df.groupby(['year_label','work_map'])['cnt'].agg(['mean','sum','count']).round(1)
print(work_stats)

plt.figure(figsize=(10,6))
sns.barplot(data=df,x='work_map',y='cnt',hue='year_label',palette='Blues',errorbar=None)
plt.title('2011-2012 cnt mean by work',fontsize=15)
plt.xlabel('work',fontsize=12)
plt.ylabel('cnt',fontsize=12)
plt.legend(title='year')
plt.grid(axis='y',linestyle='--',alpha=0.7)
plt.tight_layout()
plt.show()


#不同天气状况分布图
weather_map = {1:'sunny',2:'cloudy',3:'rainy'}
df['weather_map'] = df['weathersit'].map(weather_map)
weather_stats = df.groupby(['year_label','weather_map'])['cnt'].agg(['mean','sum','count']).round(1)
print(weather_stats)

plt.figure(figsize=(10,6))
sns.barplot(data=df,x='weather_map',y='cnt',hue='year_label',palette='Blues',errorbar=None)
plt.title('2011-2012 cnt mean by weather',fontsize=15)
plt.xlabel('weather',fontsize=12)
plt.ylabel('cnt',fontsize=12)
plt.legend(title='year')
plt.grid(axis='y',linestyle='--',alpha=0.7)
plt.tight_layout()
plt.show()

#cnt和各连续变量的相关性分析
#散点图
cols = ['cnt','temp','atemp','hum','windspeed']
fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(14,10))  #设置一个2行2列的图表大画布
axes = axes.flatten() #把四个子图展平成一个一维数组，方便下面循环画图
weather_vals=['temp','atemp','hum','windspeed']
titles = ['temp','atemp','hum','windspeed']

for i,var in enumerate(weather_vals):
    sns.scatterplot(data=df,x=var,y='cnt',alpha=0.5,ax=axes[i],color='steelblue')
    axes[i].set_title(titles[i],fontsize=14)
    axes[i].set_ylabel('cnt',fontsize=14)
    axes[i].set_xlabel(var,fontsize=12)
    axes[i].grid(True,linestyle='--',alpha=0.5)

plt.tight_layout()
plt.show()

#计算相关系数矩阵
corr_matrix = df[cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt='.3f',vmin=-1,vmax=1,linewidth=0.5)
plt.title('Correlation matrix',fontsize=15)
plt.tight_layout()
plt.show()

from scipy import stats #引入统计检验库
#切分数据
good_weather = df[df['weathersit']==1]['cnt'] #从整个大表里，把天气是晴天的那些行全部跳出来
bad_weather = df[df['weathersit'].isin([2,3])]['cnt'] #阴/雾/小雨
#打印两组数据
print("好天气",good_weather)
print("坏天气",bad_weather)

#计算t检验
t_state,p_value = stats.ttest_ind(good_weather,bad_weather,equal_var=False)
print(f"\nt 统计量: {t_state:.3f}")  #t的公式大致是（晴天均值-雨天均值）/数据的标准误差
print(f"p 值: {p_value:.3e}")  #若天气根本不影响，那恰好抽到晴天多雨天少的样本概率有多大

#下结论
if p_value < 0.05:
    print("拒绝原假设。好天气与差天气的用车辆存在显著差异")
else:
    print("无法拒绝原假设，好天气与差天气的用车辆差异不显著")


import statsmodels.api as sm
import numpy as np

#准备特征变量
features = ['temp','hum','windspeed','workingday','season','weathersit']
x = df[features]
y=df['cnt']

#把分类变量伪装成数字，让线性回归模型能读懂
x = pd.get_dummies(x,columns=['season','weathersit'],drop_first=True,dtype=int)
#添加基准线，拟合的数据更贴近真实情况
x=sm.add_constant(x)
#建立OLS（普通最小二乘法）模型并拟合
model = sm.OLS(y,x).fit()
print(model.summary())

#提取系数和置信区间
coefs = model.params
conf_int = model.conf_int()
errors = np.abs(conf_int[1] - coefs)  #计算误差棒的长度

plt.figure(figsize=(10,6))
plt.errorbar(coefs,coefs.index,xerr=errors,fmt='o',color='black',capsize=5,ecolor='gray')
plt.axvline(x=0,color='red',linestyle='--')

plt.title('OLS',fontsize=15)
plt.xlabel('coefficients',fontsize=12)
plt.grid(axis='y',linestyle='--',alpha=0.6)
plt.tight_layout()
plt.show()
