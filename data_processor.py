import numpy as np
from sklearn.cluster import KMeans


class ProcessData:
    """
    price_data： 价格数据
    n： 一个储存多个移动窗口长度的向量，维度为1 * 向量个数
    num_cluster： 生成聚类的个数
    num_effective_cluster： 选取最有效的聚类的个数

    """
    def __init__(self, price_data, n, num_cluster, num_effective_cluster):
        self.price_data = price_data
        self.n = n
        self.num_cluster = num_cluster
        self.num_effective_cluster = num_effective_cluster

    """
    把原始价格数据分别按照不同的窗口长度n划分后组成矩阵
    把每个矩阵存在list中，list的每个元素代表一个按照某种窗口长度划分成时间序列后组成的矩阵
    """
    def generate_time_series(self):
        num_of_matrices = len(self.n)
        list_matrix = [[] for k in range(num_of_matrices)]
        for i in range(len(self.n)):
            num_n = self.n[i]
            num_row = len(self.price_data[0]) - num_n
            ts = np.empty((num_row, num_n + 1))

            for j in range(num_row):
                ts[j, :num_n] = self.price_data[0, j:j+num_n]
                ts[j, num_n] = self.price_data[0, j + num_n] - self.price_data[0, j + num_n - 1]

            list_matrix[i].append(ts)

        return list_matrix

    """
    用划分好的时间序列进行聚类
    返回聚类
    """
    def find_clusters(self):
        time_series = self.generate_time_series()
        num_matrices = len(self.n)
        list_clusters = [[] for k in range(num_matrices)]
        for i in range(num_matrices):
            generate_clusters = KMeans(n_clusters=self.num_cluster, random_state=20)
            generate_clusters.fit(time_series[i][0])
            list_clusters[i].append(generate_clusters.cluster_centers_)
        return list_clusters

    """
    将每个聚类的特征最大值和最小值求差，把聚类按照极差值从大到小排列，并根据需求选出前几个聚类
    返回选出的聚类
    """
    def select_effective_clusters(self):
        clusters = self.find_clusters()
        num_cluster_list = len(self.n)
        list_effective_clusters = [[] for k in range(num_cluster_list)]
        for i in range(num_cluster_list):
            cluster = clusters[i][0]
            list_effective_clusters[i].append(cluster[np.argsort(np.ptp(cluster, axis=1))[-self.num_effective_cluster:]])

        return list_effective_clusters
