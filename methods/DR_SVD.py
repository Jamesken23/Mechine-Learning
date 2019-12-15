import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loadDataSet(filename):
    df = pd.read_table(filename, sep='\t')
    return np.array(df)


# 基于特征值的大小，对特征值以及特征向量进行排序。倒序排列
def sortByEigenValue(Eigenvalues, EigenVectors):
    idx = Eigenvalues.argsort()[::-1]
    EigenVectors = EigenVectors[:, idx]
    Eigenvalues.sort()
    return Eigenvalues, EigenVectors


def showData(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0], dataMat[:, 1], c="green")
    ax.scatter(reconMat[:, 0], reconMat[:, 1], c="red")
    plt.show()


# 详细版SVD算法
def svd(imgMat, k):
    XTX = np.matmul(imgMat.transpose(), imgMat)
    XXT = np.matmul(imgMat, imgMat.transpose())
    # 获取右奇异特征值以及向量
    right_vals, right_vet = np.linalg.eig(XTX)
    # 将特征值和特征向量逆序排列
    right_vals, right_vet = sortByEigenValue(right_vals, right_vet)
    # 得到奇异向量的特征值
    Z_vals = np.sqrt(right_vals)
    # 获取大于0的特征值个数
    eff_z_num = len(list(filter(lambda x: x > 0, Z_vals)))

    # 初始化左奇异特征向量
    left_vet = np.zeros(shape=[imgMat.shape[0], eff_z_num])
    # 遍历更新值
    for index in range(eff_z_num):
        left_vet[:, index] = np.transpose(np.matmul(imgMat, right_vet[:, index]) / Z_vals[index])

    if k > eff_z_num:
        print("不符合有效降维")
        return
    else:
        U = left_vet[:, :k]
        print("z", Z_vals)
        Z = np.diag(Z_vals[:k])
        V = right_vet[:, :k]
        # 重构矩阵
        new_Mat = U * Z * V.transpose()
        return new_Mat


def recoverBySVD(imgMat, k):
    # singular value decomposition
    U, s, V = np.linalg.svd(imgMat)
    # choose top k important singular values (or eigens)
    Uk = U[:, 0:k]
    Sk = np.diag(s[0:k])
    Vk = V[0:k, :]
    # recover the image
    imgMat_new = Uk * Sk * Vk
    return imgMat_new


if __name__ == "__main__":
    # Load the dataset
    dataMat = loadDataSet('./dataset/testSet.txt')
    # imgMat_new = recoverBySVD(dataMat, 1)
    imgMat_new = svd(dataMat, 1)
    print("imgMat_new shape", imgMat_new.shape)
    # showData(dataMat, lowDDataMat)
    showData(dataMat, imgMat_new)