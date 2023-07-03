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


# # 创建示例数据框
# data = {
#     'Category': ['A', 'A', 'B', 'B', 'B'],
#     'Value': [1, 2, 3, 4, 5]
# }
# df = pd.DataFrame(data)

# # 使用transform函数计算每个组的均值
# df['Mean'] = df.groupby('Category')['Value'].transform('mean')

# # 使用apply函数对整个数据框应用自定义函数
# def multiply_by_two(x):
#     return x * 2

# df['Value_doubled'] = df['Value'].apply(multiply_by_two)

# # 打印结果
# print(df)


######################################################

from sentence_transformers import SentenceTransformer, util
import numpy as np

# 并查集
class UnionFind():
    def __init__(self, length: int):
        self.union = list(range(length))
        self.length = length

    def find(self, x):
        if (self.union[x] == x):
            return x
        else:
            return self.find(self.union[x])

    def merge(self, x, y):
        self.union[self.find(x)] = self.find(y)

#计算两两相似度

sentence_embeddings = np.array([[12, 23, 2, 23, 21, 1, 9],
                                [23, 23, 3, 53, 5, 11, 93],
                                [45, 53, 7, 11, 22, 15, 93],
                                [3, 3, 78, 24, 87, 2, 2]], dtype=np.float32)



similarity_matrix = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings) # 计算余弦相似度矩阵
similarity_matrix = similarity_matrix.to('cpu').numpy()

print(f"similarity_matrix: {similarity_matrix}")

mask = 1 - np.eye(similarity_matrix.shape[0]) # 使用掩码将对角线元素清0
similarity_matrix = similarity_matrix * mask
max_similarity = similarity_matrix.max(axis=1) # 每个词与其他所有词的最大相似度
max_index = np.argmax(similarity_matrix, axis=1) # 每个词与其最相似的下标

print(f"max_similarity: {max_similarity}, max_index: {max_index}")

# 合并相似度大于等于阈值
threshold = 0.85
dsu = UnionFind(sentence_embeddings.shape[0])
print(f"1. dsu:{dsu.union}")

for i in range(sentence_embeddings.shape[0]):
    if max_similarity[i] >= threshold:
        dsu.merge(i, max_index[i])

print(f"2. dsu:{dsu.union}")