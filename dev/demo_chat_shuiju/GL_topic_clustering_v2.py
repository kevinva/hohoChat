from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredMarkdownLoader, CSVLoader
from langchain.text_splitter import MarkdownTextSplitter, CharacterTextSplitter ,TextSplitter

import pandas as pd
import torch

from sklearn.cluster import DBSCAN

EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
EMBEDDING_MODEL = "/root/autodl-tmp/models/sentence_pair_sim"

# 用FAISS.from_documents 去加载文本
topic_data = pd.read_excel('./outputs/topictask_20230704140642.xlsx')
print('未删除空值前', topic_data.shape[0])
topic_data = topic_data.dropna()
print('删除空值后', topic_data.shape[0])


topic = [t.replace('主题：','').replace('客户关心的','')
         for t in topic_data['主题']]


# embeddings = HuggingFaceEmbeddings(model_name = "nghuyong/ernie-3.0-base-zh",
#                                    model_kwargs={'device': EMBEDDING_DEVICE})

#从csv加载
# text_splitter = CharacterTextSplitter()
# docs = []
# loader = CSVLoader('对话主题总结.csv',  encoding='gbk')
# docs += loader.load_and_split(text_splitter)

# vector_store = FAISS.from_texts(topic, embeddings)
#
# import faiss
# Index = faiss.IndexFlatL2(768)


#从sentence_transformers 进行加载
# 加载模型，将数据进行向量化处理
from sentence_transformers import SentenceTransformer, util
import numpy as np
model_name = EMBEDDING_MODEL #'hfl/chinese-roberta-wwm-ext'
model = SentenceTransformer(model_name)
#sent_model/sentence_pair_sim/   hfl/chinese-roberta-wwm-ext

sentence_embeddings = model.encode(topic)
print(f"sentence_embeddings: {sentence_embeddings.shape}")

#采用AffinityPropagation 层次聚类
# from sklearn.cluster import AffinityPropagation
# af = AffinityPropagation(preference=90)
# clustering = af.fit(sentence_embeddings)
# topic_data['主题分类'] = list(clustering.labels_)
# print(len(set(list(clustering.labels_))))

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
similarity_matrix = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings) # 计算余弦相似度矩阵
similarity_matrix = similarity_matrix.to('cpu').numpy()


mask = 1 - np.eye(similarity_matrix.shape[0]) # 使用掩码将对角线元素清0
similarity_matrix = similarity_matrix * mask
max_similarity = similarity_matrix.max(axis=1) # 每个词与其他所有词的最大相似度
max_index = np.argmax(similarity_matrix, axis=1) # 每个词与其最相似的下标

# 合并相似度大于等于阈值
threshold = 0.7
dsu = UnionFind(sentence_embeddings.shape[0])
for i in range(sentence_embeddings.shape[0]):
    if max_similarity[i] >= threshold:
        dsu.merge(i, max_index[i])

_topic_group = {}
for i in range(sentence_embeddings.shape[0]):
    if dsu.find(i) not in _topic_group:
        _topic_group[dsu.find(i)] = []
    _topic_group[dsu.find(i)].append(topic[i])

dict_topic = {}
for k,v in _topic_group.items():
    for i in v:
        dict_topic[i] = k

print(f"dict_topic: {len(dict_topic)}")

clusterring = DBSCAN(eps = 13, min_samples = 100).fix(sentence_embeddings)
print(f"clusterring labels: {clusterring.labels_}")


# topic_data['主题分类'] = [dict_topic[k] for k in topic]
# topic_data['主题'] = topic
# topic_data['主题长度'] = topic_data['主题'].apply(lambda x:len(x))
# topic_data = topic_data[topic_data['主题长度'] <= 30]
# topic_data['原对话'].apply(lambda x:x.replace(
#     '：请基于以下客服与客户对话，进行总结任务——客户关心的主题，总结字数在10个字以内，返回格式：主题：对应主题 对话如下：""""',
#     ''))

# #topic_data[['主题', '主题长度']].groupby(['主题长度']).min()
# topic_data['分类后主题'] = topic_data.groupby('主题分类')['主题'].transform(lambda x: x.iloc[x.str.len().argmin()])
# topic_data.to_excel('./outputs/topicclassification-{}-{}.xlsx'.format(model_name.replace('/','_'), threshold))


