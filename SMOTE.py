from sklearn.neighbors import NearestNeighbors
from base_sampler import *
import numpy as np


# 使用K-近邻方法产生新样本
def make_sample(old_feature_data, diff):
    # 获取每一个少数类样本点周围最近的n_neighbors-1个点的位置矩阵
    nns = NearestNeighbors(n_neighbors=6).fit(old_feature_data).kneighbors(old_feature_data, return_distance=False)[:,1:]
    # 随机产生diff个随机数作为之后产生新样本的选取的样本下标值
    samples_indices = np.random.randint(low=0, high=np.shape(old_feature_data)[0], size=diff)
    # 随机产生diff个随机数作为之后产生新样本的间距值
    steps = np.random.uniform(size=diff)
    cols = np.mod(samples_indices, nns.shape[1])
    reshaped_feature = np.zeros((diff, old_feature_data.shape[1]))
    for i, (col, step) in enumerate(zip(cols, steps)):
        row = samples_indices[i]
        reshaped_feature[i] = old_feature_data[row] - step * (old_feature_data[row] - old_feature_data[nns[row, col]])
    # 将原少数类样本点与新产生的少数类样本点整合
    new_min_feature_data = np.vstack((reshaped_feature, old_feature_data))
    return new_min_feature_data


# 对不平衡的数据集imbalanced_data_arr2进行SMOTE采样操作，返回平衡数据集
# :param imbalanced_data_arr2: 非平衡数据集
# :return: 平衡后的数据集
def SMOTE(imbalanced_data_arr2):
    # 将数据集分开为少数类数据和多数类数据
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_data_arr2)
    # print(minor_data_arr2.shape)
    # 计算多数类数据和少数类数据之间的数量差,也是需要过采样的数量
    diff = major_data_arr2.shape[0] - minor_data_arr2.shape[0]
    # 原始少数样本的特征集
    old_feature_data = minor_data_arr2[:, : -1]
    # 原始少数样本的标签值
    old_label_data = minor_data_arr2[0][-1]
    # 使用K近邻方法产生的新样本特征集
    new_feature_data = make_sample(old_feature_data, diff)
    # 使用K近邻方法产生的新样本标签数组
    new_labels_data = np.array([old_label_data] * np.shape(major_data_arr2)[0])
    # 将类别标签数组合并到少数类样本特征集，构建出新的少数类样本数据集
    new_minor_data_arr2 = np.column_stack((new_feature_data, new_labels_data))
    # print(new_minor_data_arr2[:,-1])
    # 将少数类数据集和多数据类数据集合并，并对样本数据进行打乱重排，
    balanced_data_arr2 = concat_and_shuffle_data(new_minor_data_arr2, major_data_arr2)
    return balanced_data_arr2


# 测试
if __name__ == '__main__':
    imbalanced_data = np.load('imbalanced_train_data_arr2.npy')
    print(imbalanced_data.shape)
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_data)
    print(minor_data_arr2.shape)
    print(major_data_arr2.shape)
    # 测试SMOTE方法
    balanced_data_arr2 = SMOTE(imbalanced_data)
    print(balanced_data_arr2)
    print(balanced_data_arr2.shape)