import pandas as pd
from sklearn.cluster import KMeans

# # 创建示例数据
# data = {
#     'Category': ['A', 'A', 'B', 'B', 'B', 'C'],
#     'Text': ['This is a long text', 'Short text', 'Another text', 'A text', 'A short text', 'Text']
# }
# df = pd.DataFrame(data)

# def myfunc(row):
#     print(row.str.len().argmin())
#     return row.iloc[row.str.len().argmin()]

# # 创建聚类器
# # kmeans = KMeans(n_clusters=2)

# # 使用聚类器对Category列进行聚类
# # df['Cluster'] = kmeans.fit_predict(df[['Category']])

# # 根据Cluster列的值选择对应的Text列的最短文本作为新增列
# # df['Shortest_Text'] = df.groupby('Category')['Text'].transform(lambda x: x.iloc[x.str.len().argmin()])
# df['Shortest_Text'] = df.groupby('Category')['Text'].transform(myfunc)
# # 打印结果
# print(df)


# 创建示例数据框
data = {
    'Category': ['A', 'A', 'B', 'B', 'B'],
    'Value': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)

# 使用transform函数计算每个组的均值
df['Mean'] = df.groupby('Category')['Value'].transform('mean')

# 使用apply函数对整个数据框应用自定义函数
def multiply_by_two(x):
    return x * 2

df['Value_doubled'] = df['Value'].apply(multiply_by_two)

# 打印结果
print(df)