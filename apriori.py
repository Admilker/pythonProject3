import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori  # 生成频繁项集
from mlxtend.frequent_patterns import association_rules  # 生成强关联规则
import warnings

warnings.filterwarnings("ignore")  # 用于排除警告


def loaddata():
    order_data = pd.read_csv('C:\\Users\\贾星辰\\Desktop\\GoodsOrder.csv', header=0, encoding='gbk')

    print(order_data.isnull().any())

    # 转换数据格式
    order_data['Goods'] = order_data['Goods'].apply(lambda x: "," + x)
    order_data = order_data.groupby('id').sum().reset_index()
    order_data['Goods'] = order_data['Goods'].apply(lambda x: [x[1:]])
    order_data_list = list(order_data['Goods'])
    # print(order_data_list)
    # 分割商品名为每一个元素
    data_translation = []
    for i in order_data_list:
        # 列表中元素为1个字符串，所以用0索引
        p = i[0].split(",")
        data_translation.append(p)
    return data_translation


if __name__ == '__main__':
    dataSet = loaddata()
    column_list = []
    for var in dataSet:
        column_list = set(column_list) | set(var)
    print('转换原数据到0-1矩阵')
    data = pd.DataFrame(np.zeros((len(dataSet), 169)), columns=list(column_list))
    for i in range(len(dataSet)):
        for j in dataSet[i]:
            data.loc[i, j] += 1
    # apriori算法
    frequent_itemsets = apriori(data, min_support=0.02, use_colnames=True)
    print(pd.DataFrame(frequent_itemsets))
    pd.DataFrame(frequent_itemsets).to_csv('frequent_itemsets.csv')
    # 生成关联准则
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.35)
    rules2 = association_rules(frequent_itemsets, metric="lift", min_threshold=1.25)
    print(pd.DataFrame(rules))
    print(pd.DataFrame(rules2))
    pd.DataFrame(rules).to_csv('rules.csv')

