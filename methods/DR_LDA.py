import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification


# 获取两类原始数据，根据0，1标签来切分
def split_data(X, Y):
    Y0_index = np.array(np.where(Y == 0)).squeeze()
    Y1_index = np.array(np.where(Y == 1)).squeeze()
    return X[Y0_index], X[Y1_index]


# 基于特征值的大小，对特征值以及特征向量进行排序。倒序排列
def sortByEigenValue(Eigenvalues, EigenVectors):
    idx = Eigenvalues.argsort()[::-1]
    EigenVectors = EigenVectors[:, idx]
    Eigenvalues.sort()
    return Eigenvalues, EigenVectors


def lda(X, Y, k):
    """
    线性判别分析LDA简易方法实现，二分类
    :param k: 降维的维数
    :param X: 原始数据
    :param Y: 数据的相应标签：0,1
    :return: 降维后的特征向量
    """
    X0, X1 = split_data(X, Y)
    # 分别获取两类数据集的均值向量
    U_0 = np.mean(X0, axis=0)
    U_1 = np.mean(X1, axis=0)
    # 分别获取两类数据集的散列矩阵
    Z_0 = np.matmul((X0 - U_0).transpose(), (X0 - U_0))
    Z_1 = np.matmul((X1 - U_1).transpose(), (X1 - U_1))
    # 获得类内散度矩阵
    S_w = Z_0 + Z_1
    # 获得类间散度矩阵
    S_b = np.matmul((U_0 - U_1).transpose(), (U_0 - U_1))
    # 得到新的特征值和特征向量，并逆序排序
    vals, vets = np.linalg.eig(np.linalg.inv(S_w) * S_b)
    vals, vets = sortByEigenValue(vals, vets)
    new_vets = vets[:, :k]
    # 分别将原始二分类数据转换到新的低维空间中
    Low_X0 = np.matmul(X0, new_vets)
    Low_X1 = np.matmul(X1, new_vets)
    # 最后重建数据
    recont_X0 = np.matmul(Low_X0, new_vets.transpose())
    recont_X1 = np.matmul(Low_X1, new_vets.transpose())
    return recont_X0, recont_X1


if '__main__' == __name__:
    # 产生分类数据
    n_samples = 500
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_classes=2,
                               n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)
    recont_X0, recont_X1 = lda(X, y, 1)

    # 原始数据
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    ax.scatter(recont_X0[:, 0], recont_X0[:, 1], marker='o', c="r")
    ax.scatter(recont_X1[:, 0], recont_X1[:, 1], marker='o', c="b")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
