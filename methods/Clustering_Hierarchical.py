import math
import pylab as pl
from sklearn import datasets


def dist(node1, node2):
    """
    计算欧几里得距离,node1,node2分别为两个元组
    :param node1:
    :param node2:
    :return:
    """
    return math.sqrt(math.pow(node1[0]-node2[0], 2)+math.pow(node1[1]-node2[1], 2))


def dist_min(cluster_x, cluster_y):
    """
    Single Linkage
    又叫做 nearest-neighbor ，就是取两个类中距离最近的两个样本的距离作为这两个集合的距离。
    :param cluster_x:
    :param cluster_y:
    :return:
    """
    return min(dist(node1, node2) for node1 in cluster_x for node2 in cluster_y)


def dist_max(cluster_x, cluster_y):
    """
    Complete Linkage
    这个则完全是 Single Linkage 的反面极端，取两个集合中距离最远的两个点的距离作为两个集合的距离。
    :param cluster_x:
    :param cluster_y:
    :return:
    """
    return max(dist(node1, node2) for node1 in cluster_x for node2 in cluster_y)


def dist_avg(cluster_x, cluster_y):
    """
    Average Linkage
    这种方法就是把两个集合中的点两两的距离全部放在一起求均值，相对也能得到合适一点的结果。
    :param cluster_x:
    :param cluster_y:
    :return:
    """
    return sum(dist(node1, node2) for node1 in cluster_x for node2 in cluster_y)/(len(cluster_x)*len(cluster_y))


def find_min(distance_matrix):
    """
    找出距离最近的两个簇下标
    :param distance_matrix:
    :return:
    """
    min = 1000
    x = 0
    y = 0
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j and distance_matrix[i][j] < min:
                min = distance_matrix[i][j]
                x = i
                y = j
    return x, y, min


def AGNES(dataset, distance_method, k):
    """
    聚类算法模型
    :param dataset: 数据集
    :param distance_method: 计算簇类聚类的方法
    :param k: 目标簇类数目
    :return:
    """
    # 初始化簇类集合和距离矩阵
    cluster_set = []
    distance_matrix = []
    for node in dataset:
        cluster_set.append([node])
    print('original cluster set:', cluster_set)
    for cluster_x in cluster_set:
        distance_list = []
        for cluster_y in cluster_set:
            distance_list.append(distance_method(cluster_x, cluster_y))
        distance_matrix.append(distance_list)
    q = len(dataset)
    # 合并更新
    while q > k:
        id_x, id_y, min_distance = find_min(distance_matrix)
        cluster_set[id_x].extend(cluster_set[id_y])
        cluster_set.remove(cluster_set[id_y])
        distance_matrix = []
        for cluster_x in cluster_set:
            distance_list = []
            for cluster_y in cluster_set:
                distance_list.append(distance_method(cluster_x, cluster_y))
            distance_matrix.append(distance_list)
        q -= 1
    return cluster_set


def draw(cluster_set):
    """
    画图
    :param cluster_set:
    :return:
    """
    color_list = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for cluster_idx, cluster in enumerate(cluster_set):
        coo_x = []  # x坐标列表
        coo_y = []  # y坐标列表
        for node in cluster:
            coo_x.append(node[0])
            coo_y.append(node[1])
        pl.scatter(coo_x, coo_y, marker='x', color=color_list[cluster_idx % len(color_list)], label=cluster_idx)
    pl.legend(loc='upper right')
    pl.show()


iris = datasets.load_iris()
cluster_set = AGNES(iris.data.tolist(), dist_max, 4)
print("final cluster set:", cluster_set)
draw(cluster_set)