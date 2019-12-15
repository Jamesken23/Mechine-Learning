import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loadDataSet(filename):
    df = pd.read_table(filename, sep='\t')
    return np.array(df)


def showData(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0], dataMat[:, 1], c="green")
    ax.scatter(reconMat[:, 0], reconMat[:, 1], c="red")
    plt.show()


def pca(dataMat, topNfeat=99):
    # 对所有样本中心化
    meanVals = np.mean(dataMat, axis=0)
    new_data = dataMat - meanVals
    # 计算样本的协方差矩阵，若rowvar=False，表示将每一列看做Variable
    covmat = np.cov(new_data, rowvar=False)
    print("协方差矩阵为:", covmat)
    # 对协方差矩阵做特征分解，求得特征值和特征向量，并将特征值从大到小排序，筛选出前topNfeat个
    eigVals, eigVects = np.linalg.eig(covmat)
    idx = eigVals.argsort()[::-1]
    eigVects = eigVects[:, idx]
    eigVects = eigVects[:, :topNfeat]
    print("new_data shape", new_data.shape)
    print("eigVects", eigVects)
    # 将数据转换到新的低维空间中
    lowDDataMat = np.matmul(new_data, eigVects)  # 降维之后的数据
    reconMat = np.matmul(lowDDataMat, eigVects.T) + meanVals  # 重构数据，可在原数据维度下进行对比查看
    return np.array(lowDDataMat), np.array(reconMat)


if __name__ == "__main__":
    # Load the dataset
    dataMat = loadDataSet('./dataset/testSet.txt')
    lowDDataMat, reconMat = pca(dataMat, 1)
    # showData(dataMat, lowDDataMat)
    showData(dataMat, reconMat)
    print(lowDDataMat)