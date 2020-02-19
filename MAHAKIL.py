"""
MAHAKIL方法的实现和使用接口
"""
import math
from base_sampler import *
import numpy as np


def get_mahalanobis_distances_to_center(data_arr2):
    """
    计算样本数据中所有样本到样本空间中心点的马氏距离
    :param data_arr2: 样本数据集，n*m二维数组，其中n为样本数量，m为特征数量
    :return: 所有样本到样本空间中心点的马氏距离，为n维行向量
    """
    S_arr2 = np.cov(data_arr2.T)  # 求特征之间的协方差矩阵
    SI_arr2 = np.linalg.inv(S_arr2)  # 求协方差矩阵的逆矩阵
    # print(S_arr2.shape)
    u_arr1 = np.mean(data_arr2, axis=0) #计算样本空间中心点
    rows=data_arr2.shape[0] #行数，样本数量
    mahalanobis_distance_arr1=np.empty((rows,)) #所有样本到样本空间中心点的马氏距离，为n维行向量
    for row in range(0,rows):
    #对于每个样本数据
        data_arr1=data_arr2[row,:]  #该数据的特征向量
        distance_arr1=data_arr1-u_arr1  #该样本到中心点的差值向量
        mahalanobis_distance_arr1[row]=np.sqrt(np.dot(np.dot(distance_arr1,SI_arr2),distance_arr1.T))   #该样本到中心点的马氏距离
    return mahalanobis_distance_arr1

def sorted_data_arr2_by_distance_arr1(data_arr2,distance_arr1):
    """
    根据每个样本到中心点的距离，将样本集进行降序排序
    :param data_arr2: 二维数组
    :param distance_arr1: 所有样本到样本空间中心点的距离，为n维行向量
    :return: 排序后的样本集
    """
    data_arr1_distance_zip=zip(data_arr2,distance_arr1) #将每个样本及其距离进行zip
    sorted_data_arr1_distance_tup=sorted(data_arr1_distance_zip,key=lambda d:d[1],reverse=True)    #根据距离对样本进行降序排序
    sorted_data_arr1_tup,distance_tup=zip(*sorted_data_arr1_distance_tup)  #将排序好的样本和距离分开为各自的元组
    sorted_data_arr2=np.array(sorted_data_arr1_tup) #将排序好的样本元组转为numpy二维数组
    return sorted_data_arr2


def inheritant_sample(sorted_data_arr2,sample_number):
    """
    根据排序好的样本数据集进行遗传采样
    :param sorted_data_arr2: 排序好的样本数据集
    :param sample_number: 需要采样的数量
    :return: 采样好的
    """

    # 先根据降序排序好的样本特征集，将样本特征集分为相同数量的2个子样本特征集+1个样本特征，
    # 2个子样本特征集指的是第一组和第二组样本特征集，第一组离中心点远，第二组离中心点近，组成一个列表
    # 1个样本特征指的是，如果中间多了一个，就返回那个那多的，否则返回空，但可以直接拼接
    rows, cols = sorted_data_arr2.shape  # 样本特征集的形状

    data_arr2s=list()    #多个子样本特征集组成的一个列表
    extra_data_arr1 = np.empty((0, cols))  # 可能会多余的一个样本特征，初始化为空，如果没有，那就是空

    middle_number = math.floor(rows / 2)  # 找排序在中间的样本位置,也是每个子特征集中最大的样本数量
    # print(middle_number)
    data_arr2s.append(sorted_data_arr2[:middle_number, :])  # 前一半离中心点较远的子特征集加入特征集列表作为第一个元素
    # print(sorted_data_arr2s[0].shape)
    if middle_number < rows / 2:  # 如果样本有奇数个
        extra_data_arr1 = np.row_stack((extra_data_arr1,sorted_data_arr2[middle_number, :]))  # 中间的样本特征作为多余的那个特征
        data_arr2s.append(sorted_data_arr2[middle_number + 1:, :])  #后一半离中心点较近的子特征集加入特征集列表作为第二个元素
    else:  # 如果样本有偶数个
        data_arr2s.append(sorted_data_arr2[middle_number:, :])  #后一半离中心点较近的子特征集加入特征集列表作为第二个元素

    # 开始生成新的样本特征
    i=0 #新样本数量
    # print('sample number:',sample_number)
    while i < sample_number:
    #如果生成样本特征数量少于所需，就继续生成
        temp_data_arr2s = list()    #临时的特征集列表
        for j in range(0,len(data_arr2s)-1):
        #遍历特征集列表中的子特征集，最多遍历到倒数第二个
            temp_data_arr2s.append(data_arr2s[j])   #将当前的子特征集加入列表中
            if sample_number-i>middle_number:
            #如果基于相邻子特征集新生成的完整的子特征集大小小于还需要生成的样本特征数量
                generated_data_arr2=0.5*(data_arr2s[j]+data_arr2s[j+1]) #利用相邻子特征集生成新的完整的子特征集
                temp_data_arr2s.append(generated_data_arr2) #加入到临时列表中
                i += middle_number  #更新i
            else:
                n=sample_number-i   #需要最后生成的样本特征数量
                generated_data_arr2=0.5*(data_arr2s[j][:n,:]+data_arr2s[j+1][:n,:]) #利用相邻的子特征集生成部分的新子特征集
                temp_data_arr2s.append(generated_data_arr2) #加入到临时列表中
                i += middle_number  #更新i
                break   #最后的样本特征生成并加入临时列表后，需要中断循环了，此时data_arr2s中可能还有大量子特征集

        # print('i:',i)
        temp_data_arr2s.extend(data_arr2s[j+1:]) #中断循环后，data_arr2s中可能还有大量子特征集，而不止最后一个
        # print('length of temp_data_arr2s:', len(temp_data_arr2s))
        data_arr2s=temp_data_arr2s  #更新data_arr2s
        # print('length of data_arr2s:',len(data_arr2s))
    data_arr2s.append(extra_data_arr1)  #将可能会多余的一个样本特征加入data_arr2s中
    return np.concatenate(data_arr2s,axis=0)    #把样本特征集列表进行拼接，形成完整的样本特征集进行返回

def MAHAKIL(imbalanced_data_arr2):
    """
    对不平衡的数据集imbalanced_data_arr2进行MAHAKIL采样操作，返回平衡数据集
    :param imbalanced_data_arr2: 非平衡数据集
    :return: 平衡后的数据集
    """
    #将数据集分开为少数类数据和多数类数据
    minor_data_arr2, major_data_arr2=seperate_minor_and_major_data(imbalanced_data_arr2)
    # print(minor_data_arr2.shape)
    #计算多数类数据和少数类数据之间的数量差,也是需要过采样的数量
    diff=major_data_arr2.shape[0]-minor_data_arr2.shape[0]
    # print(major_data_arr2.shape[0])
    # print(minor_data_arr2.shape[0])
    # print(diff)
    #计算所有少数类样本点到中心点的马氏距离
    minor_mahalanobis_distance_arr1=get_mahalanobis_distances_to_center(minor_data_arr2[:, :-1])
    # print(mahalanobis_distance_arr1.shape)
    #根据所有少数类样本点在样本空间中对于样本中心的距离对少数类样本数据进行降序排序
    sorted_minor_data_arr2=sorted_data_arr2_by_distance_arr1(minor_data_arr2,minor_mahalanobis_distance_arr1)
    #根据染色体遗传算法根据排序好的少数类样本特征集生成新的少数类样本特征集
    new_minor_feature_arr2=inheritant_sample(sorted_minor_data_arr2[:,:-1],diff)
    # print(new_minor_feature_arr2.shape)
    #构建新的符合少数类样本长度的类别标签数组
    new_minor_label_arr1=np.full((new_minor_feature_arr2.shape[0],),sorted_minor_data_arr2[0,-1])
    # print(label_arr1.shape)
    #将类别标签数组合并到少数类样本特征集，构建出新的少数类样本数据集
    new_minor_data_arr2=np.column_stack((new_minor_feature_arr2,new_minor_label_arr1))
    # print(new_minor_data_arr2[:,-1])
    #将少数类数据集和多数据类数据集合并，并对样本数据进行打乱重排，
    balanced_data_arr2=concat_and_shuffle_data(new_minor_data_arr2,major_data_arr2)
    return balanced_data_arr2


#测试
if __name__=='__main__':
    #读取本地的非平衡数据集
    imbalanced_train_data_path = 'imbalanced_train_data_arr2.npy'
    imbalanced_train_data_arr2 = np.load(imbalanced_train_data_path)
    print(imbalanced_train_data_arr2.shape)
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_train_data_arr2)
    print(minor_data_arr2.shape)
    print(major_data_arr2.shape)
    #测试MAHAKIL方法
    balanced_data_arr2=MAHAKIL(imbalanced_train_data_arr2)
    print(balanced_data_arr2.shape)