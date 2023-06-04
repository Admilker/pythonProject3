import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# from imblearn.over_sampling import SMOTE

# import graphviz
from sklearn import tree


df = pd.read_csv('./custom.csv')
data = df.copy()
pd.set_option('display.max_columns', None)
# print(data.head())

missingDf = data.isnull().sum().sort_values(ascending=False).reset_index()
missingDf.columns = ['feature', 'missing_num']
missingDf['missing_percentage'] = missingDf['missing_num'] / data.shape[0]
print(missingDf.head())

print(data[data['TotalCharges'] == ' '])

# 将特征TotalCharges转为数值型
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'],errors='coerce')


### 选取性别、是否老年人、是否有配偶、是否有家属等基本特征对客户流失有较大影响的特征
baseCols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']

for i in baseCols:
    cnt = pd.crosstab(data[i], data['Churn'])    # 构建特征与目标变量的列联表
    cnt.plot.bar(stacked=True)    # 绘制堆叠条形图，便于观察不同特征值流失的占比情况
    plt.show()    # 展示图像

# 业务特征方面
# 电话业务
posDf = data[data['PhoneService'] == 'Yes']
negDf = data[data['PhoneService'] == 'No']

fig = plt.figure(figsize=(10,4)) # 建立图像

ax1 = fig.add_subplot(121)
p1 = posDf['Churn'].value_counts()
ax1.pie(p1,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax1.set_title('Churn of (PhoneService = Yes)')

ax2 = fig.add_subplot(122)
p2 = negDf['Churn'].value_counts()
ax2.pie(p2,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax2.set_title('Churn of (PhoneService = No)')

plt.tight_layout(pad=0.5)    # 设置子图之间的间距
plt.show() # 展示饼状图

# 删除明显与预测值无关的特征
data.drop(['customerID', 'gender', 'PhoneService', 'StreamingTV', 'StreamingMovies'],inplace=True,axis=1)
# data.shape
data = data.drop(['TotalCharges'], axis=1)

for fea in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport']:
    data.loc[data[fea] == 'No internet service',fea] = 'No'
data.loc[data['MultipleLines'] == 'No phone service','MultipleLines'] = 'No'
for fea in ['Partner','Dependents','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',"PaperlessBilling",'Churn']:
    data[fea] = data[fea].apply(lambda x :1 if x == 'Yes' else 0)
for fea in  ['InternetService', 'Contract', 'PaymentMethod']:
    labels = data[fea].unique().tolist()
    data[fea] = data[fea].apply(lambda x:labels.index(x))


# 切分特征与标签
X = data.iloc[:,0:-1]
y = data['Churn']

# 切分训练集与测试集，比例为7：3
X = pd.DataFrame(X)
y = pd.DataFrame(y)

X_train,X_test,Y_train,Y_test = train_test_split(X, y, test_size=0.3,random_state=0)
for i in [X_train,X_test,Y_train,Y_test]:
    i.index = range(i.shape[0])

# 模型构建与训练
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train,Y_train)
score = clf.score(X_test,Y_test)
print(score)

# 导入精确率、召回率、F1值等评价指标
from sklearn.metrics import precision_score, recall_score, f1_score

pred = clf.predict(X_test)
r = recall_score(Y_test,pred)
p = precision_score(Y_test,pred)
f1 = f1_score(Y_test,pred)
print(r, p, f1)


