"""
采样器的基础代码,可用于后面采样器的复用
"""
import numpy as np
import os


def seperate_minor_and_major_data(imbalanced_data_arr2):
    """
    将训练数据分开为少数据类数据集和多数类数据集
    :param imbalanced_data_arr2: 非平衡数集
    :return: 少数据类数据集和多数类数据集
    """

    # 提取类别标签一维数组，并提取出两类类别标签标记
    labels_arr1 = imbalanced_data_arr2[:, -1]
    unique_labels_arr1 = np.unique(labels_arr1)
    if len(unique_labels_arr1) != 2:
        print('数据类别大于2，错误！')
        return

    # 找出少数类的类别标签
    minor_label = unique_labels_arr1[0] if np.sum(labels_arr1 == unique_labels_arr1[0]) \
                                           < np.sum(labels_arr1 == unique_labels_arr1[1]) else unique_labels_arr1[1]

    [rows, cols] = imbalanced_data_arr2.shape  # 获取数据二维数组形状
    minor_data_arr2 = np.empty((0, cols))  # 建立一个空的少数类数据二维数组
    major_data_arr2 = np.empty((0, cols))  # 建立一个空的多数类数据二维数组

    # 遍历每个样本数据，分开少数类数据和多数类数据
    for row in range(rows):
        data_arr1 = imbalanced_data_arr2[row, :]
        if data_arr1[-1] == minor_label:
            # 如果类别标签为少数类类别标签，则将数据加入少数类二维数组中
            minor_data_arr2 = np.row_stack((minor_data_arr2, data_arr1))
        else:  # 否则，将数据加入多数类二维数组中
            major_data_arr2 = np.row_stack((major_data_arr2, data_arr1))

    return minor_data_arr2, major_data_arr2


def concat_and_shuffle_data(data1_arr2, data2_arr2):
    """
    对两个numpy二维数组进行0轴连接，并对行向量进行打乱重排，
    :param data1_arr2: numpy二维数组
    :param data2_arr2: numpy二维数组
    :return:
    """
    data_arr2 = np.concatenate((data1_arr2, data2_arr2), axis=0)  # 数组0轴连接
    np.random.shuffle(data_arr2)  # 行向量shuffle
    return data_arr2


if __name__ == '__main__':
    imbalanced_train_data_path = '../../data/clean_data/imbalanced_train_data_arr2.npy'
    imbalanced_train_data_arr2 = np.load(imbalanced_train_data_path)
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_train_data_arr2)
    print(minor_data_arr2.shape)
    print(major_data_arr2.shape)
