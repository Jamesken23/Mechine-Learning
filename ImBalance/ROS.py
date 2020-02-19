"""
Random Over-Sampling (ROS)方法的使用接口
"""
from sklearn.utils import safe_indexing
from base_sampler import *
import numpy as np


def make_sample(old_feature_data, diff):
    indice = np.random.randint(low=0, high=np.shape(old_feature_data)[0], size=diff)
    reshaped_feature = np.vstack((old_feature_data, safe_indexing(old_feature_data, indice)))
    return reshaped_feature


def ROS(imbalanced_data_arr2):
    """
    对不平衡的数据集imbalanced_data_arr2进行ROS采样操作，返回平衡数据集
    :param imbalanced_data_arr2: 非平衡数据集
    :return: 平衡后的数据集
    """
    # 将数据集分开为少数类数据和多数类数据
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_data_arr2)
    # print(minor_data_arr2.shape)
    # 计算多数类数据和少数类数据之间的数量差,也是需要过采样的数量
    diff = major_data_arr2.shape[0] - minor_data_arr2.shape[0]
    # 原始少数样本的特征集
    old_feature_data = minor_data_arr2[:, : -1]
    # 原始少数样本的标签值
    old_label_data = minor_data_arr2[0][-1]
    # 使用随机复制方法产生的新样本特征集
    new_feature_data = make_sample(old_feature_data, diff)
    # 使用随机复制方法产生的新样本标签数组
    new_labels_data = np.array([old_label_data] * np.shape(major_data_arr2)[0])
    # 将类别标签数组合并到少数类样本特征集，构建出新的少数类样本数据集
    new_minor_data_arr2 = np.column_stack((new_feature_data, new_labels_data))
    # print(new_minor_data_arr2[:,-1])
    # 将少数类数据集和多数据类数据集合并，并对样本数据进行打乱重排，
    balanced_data_arr2 = concat_and_shuffle_data(new_minor_data_arr2, major_data_arr2)
    return balanced_data_arr2


#测试
if __name__=='__main__':
    imbalanced_data = np.load('imbalanced_train_data_arr2.npy')
    print(imbalanced_data.shape)
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_data)
    print(minor_data_arr2.shape)
    print(major_data_arr2.shape)
    # 测试SMOTE方法
    balanced_data_arr2 = ROS(imbalanced_data)
    print(balanced_data_arr2.shape)