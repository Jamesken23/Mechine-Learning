import numpy as np
import pandas as pd

MIN_VAL = 10e-8
np.set_printoptions(2)


def get_confusion_matrix_df(true_labels, pred_labels, unique_labels=None):
    """
    根据真实标记和预测标记，构建混淆矩阵，返回混淆矩阵的datafame形式，行列索引为label
    然后计算总体accuracy、每个类别以及所有类别平均的precision、recall、F1
    :param true_labels: 真实标记列表，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记列表，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    """
    get_confusion_matrix_df.__name__ = 'confusion_matrix'
    if unique_labels is None:  # 如果unique为空
        unique_labels = set(true_labels) | set(pred_labels)
    # 排序的标记集合，索引号即为混淆矩阵中对应的索引号
    sort_unique_labels = sorted(unique_labels)
    cm_df = pd.DataFrame(data=np.zeros(shape=(len(sort_unique_labels), len(sort_unique_labels))),
                         index=sort_unique_labels,
                         columns=sort_unique_labels)  # 建立一个全0的混淆矩阵
    for true_label, pred_label in zip(true_labels, pred_labels):
        # 遍历每一对标记
        cm_df.loc[true_label, pred_label] += 1

    # # 将true_labels,pred_labels中的标记转化为索引
    # true_classes = [sort_unique_labels.index(true_label) for true_label in true_labels]
    # pred_classes = [sort_unique_labels.index(pred_label) for pred_label in pred_labels]
    # cn = len(sort_unique_labels)  # 类别数量 class number
    # cm = np.zeros((cn, cn))  # 定义混淆矩阵 confusion matrix
    # for true_class, pred_class in zip(true_classes, pred_classes):
    #     cm[true_class, pred_class] += 1
    # cm_df=pd.DataFrame(data=cm,index=sort_unique_labels,columns=sort_unique_labels)

    return cm_df


def get_overall_accuracy(true_labels, pred_labels, unique_labels=None):
    """
    根据真实标记和预测标记，构建混淆矩阵，计算分类的总体accuracy
    :param true_labels:  真实标记，序列，例如['A','B','C','A']
    :param pred_labels:  预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    """
    get_overall_accuracy.__name__ = 'overall_accuracy'
    # get_overall_accuracy.metric = 'overall_accuracy'
    cm_df = get_confusion_matrix_df(true_labels, pred_labels, unique_labels)  # 构建混淆矩阵
    accuracy = np.sum(np.diag(cm_df)) / np.sum(np.array(cm_df))  # 不转换成ndarray不行
    return accuracy


def get_precision_series(true_labels, pred_labels, unique_labels=None):
    '''
    根据真实标记和预测标记，构建混淆矩阵，然后计算每个类别的precision并构建成series结构
    :param true_labels: 真实标记，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    get_precision_series.__name__ = 'precision'
    # get_precision_series.metric = 'precision'
    cm_df = get_confusion_matrix_df(true_labels, pred_labels, unique_labels)  # 构建混淆矩阵
    pred_labels_series = np.sum(cm_df, axis=0)  # series
    precision_series = pd.Series(
        [cm_df.iloc[i, i] / (pred_labels_series.values[i] + MIN_VAL) for i in range(0, len(pred_labels_series))],
        index=cm_df.index.tolist())  # 计算每个类的prescision组成series数据结构
    return precision_series


def get_average_precision(true_labels, pred_labels, unique_labels=None, is_weight=False):
    '''
    根据真实标记和预测标记，计算所有类别平均的precision
    :param true_labels: 真实标记，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :return:
    '''
    get_average_precision.__name__ = 'average_precision'
    # get_average_precision.metric = 'average_precision'
    precision_series = get_precision_series(true_labels, pred_labels, unique_labels)
    if is_weight == True:
        cm_df = get_confusion_matrix_df(true_labels, pred_labels, unique_labels)  # 构建混淆矩阵
        class_ratio_series = np.sum(cm_df, axis=1) / np.sum(np.array(cm_df))  # 不转换成ndarray不行
        average_precision = float(np.sum(precision_series * class_ratio_series))
    else:
        average_precision = float(np.mean(precision_series))
    return average_precision


def get_recall_series(true_labels, pred_labels, unique_labels=None):
    '''
    根据真实标记和预测标记，构建混淆矩阵，然后计算每个类别的recall并构建成series结构
    recall貌似就是每个类预测正确实例的正确率，recall就是sensitivity！！！
    :param true_labels: 真实标记，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    get_recall_series.__name__ = 'recall'
    # get_recall_series.metric = 'recall'
    cm_df = get_confusion_matrix_df(true_labels, pred_labels, unique_labels)  # 构建混淆矩阵
    true_labels_series = np.sum(cm_df, axis=1)  # series
    recall_series = pd.Series(
        [cm_df.iloc[i, i] / (true_labels_series.values[i] + MIN_VAL) for i in range(0, len(true_labels_series))],
        index=cm_df.index.tolist())  # 计算每个类的recall组成series数据结构
    return recall_series


def get_average_recall(true_labels, pred_labels, unique_labels=None, is_weight=False):
    '''
    根据真实标记和预测标记，计算所有类别平均的recall
    :param true_labels: 真实标记，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :return:
    '''
    get_average_recall.__name__ = 'average_recall'
    # get_average_recall.metric = 'average_recall'
    recall_series = get_recall_series(true_labels, pred_labels, unique_labels)
    if is_weight == True:
        cm_df = get_confusion_matrix_df(true_labels, pred_labels, unique_labels)  # 构建混淆矩阵
        class_ratio_series = np.sum(cm_df, axis=1) / np.sum(np.array(cm_df))  # 不转换成ndarray不行
        average_recall = float(np.sum(recall_series * class_ratio_series))
    else:
        average_recall = float(np.mean(recall_series))
    return average_recall


def get_F_score_series(true_labels, pred_labels, unique_labels=None, alpha=1.):
    '''
    根据真实标记和预测标记，构建混淆矩阵，然后计算每个类别的precision和recalld的series数据结构
    然后根据precision和recall，以及给出的调和值alpha计算F-score
    :param true_labels: 真实标记，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param alpha: 综合precision和recall的调和值，默认1，即求F1-score
    :return:
    '''
    get_F_score_series.__name__ = 'F_score'
    # get_F_score_series.metric = 'F_score'
    precision_series = get_precision_series(true_labels, pred_labels, unique_labels)  # precision series
    recall_series = get_recall_series(true_labels, pred_labels)  # recall series
    F_score_series = (1 + pow(alpha, 2)) * precision_series * recall_series / \
                     (pow(alpha, 2) * precision_series + recall_series + MIN_VAL)
    return F_score_series


def get_average_F_score(true_labels, pred_labels, unique_labels=None, is_weight=False, alpha=1.):
    '''
    根据真实标记和预测标记，以及给出的调和值alpha,计算所有类别平均的F_score
    :param true_labels: 真实标记，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :param alpha: 综合precision和recall的调和值，默认1，即求F1-score
    :return:
    '''
    get_average_F_score.__name__ = 'average_F_score'
    # get_average_F_score.metric = 'average_F_score'
    F_score_series = get_F_score_series(true_labels, pred_labels, unique_labels, alpha)
    if is_weight == True:
        cm_df = get_confusion_matrix_df(true_labels, pred_labels, unique_labels)  # 构建混淆矩阵
        class_ratio_series = np.sum(cm_df, axis=1) / np.sum(np.array(cm_df))  # 不转换成ndarray不行
        average_F_score = float(np.sum(F_score_series * class_ratio_series))
    else:
        average_F_score = float(np.mean(F_score_series))
    return average_F_score


def get_F1_score_series(true_labels, pred_labels, unique_labels=None):
    '''
    根据真实标记和预测标记，构建混淆矩阵，然后计算每个类别的precision和recall的series数据结构
    然后根据precision和recall计算F1 score
    :param true_labels: 真实标记，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    get_F1_score_series.__name__ = 'F1_score'
    # get_F1_score_series.metric = 'F1_score'
    # precision_series=get_precision_series(true_labels, pred_labels)   #precision series
    # recall_series=get_recall_series(true_labels, pred_labels) #recall series
    # F1_score_series = 2.0 * precision_series * recall_series / (precision_series + recall_series + MIN_VAL)
    # return F1_score_series
    F1_score_series = get_F_score_series(true_labels, pred_labels, unique_labels, alpha=1)
    return F1_score_series


def get_average_F1_score(true_labels, pred_labels, unique_labels=None, is_weight=False):
    '''
    根据真实标记和预测标记，计算所有类别平均的F1_score
    :param true_labels: 真实标记，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :return:
    '''
    get_average_F1_score.__name__ = 'average_F1_score'
    # get_average_F1_score.metric = 'average_F1_score'
    F1_score_series = get_F1_score_series(true_labels, pred_labels, unique_labels)
    if is_weight == True:
        cm_df = get_confusion_matrix_df(true_labels, pred_labels, unique_labels)  # 构建混淆矩阵
        class_ratio_series = np.sum(cm_df, axis=1) / np.sum(np.array(cm_df))  # 不转换成ndarray不行
        average_F1_score = float(np.sum(F1_score_series * class_ratio_series))
    else:
        average_F1_score = float(np.mean(F1_score_series))
    return average_F1_score


def get_sensitivity_series(true_labels, pred_labels, unique_labels=None):
    '''
    根据真实标记和预测标记，算每个类别的敏感性或灵敏性sensitivity, 即召回率或查全率recall
    :param true_labels: 真实标记，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    get_sensitivity_series.__name__ = 'sensitivity'
    # get_sensitivity_series.metric = 'sensitivity'
    sensitivity_series = get_recall_series(true_labels, pred_labels, unique_labels)
    return sensitivity_series


def get_average_sensitivity(true_labels, pred_labels, unique_labels=None, is_weight=False):
    '''
    根据真实标记和预测标记，计算所有类别平均的sensitivity,即平均recall
    :param true_labels: 真实标记，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :return:
    '''
    get_average_sensitivity.__name__ = 'average_sensitivity'
    # get_average_sensitivity.metric = 'average_sensitivity'
    average_sensitivity = get_average_recall(true_labels, pred_labels, unique_labels, is_weight)
    return average_sensitivity


def get_specificity_series(true_labels, pred_labels, unique_labels=None):
    '''
    根据真实标记和预测标记，构建混淆矩阵，然后计算所有类别的特异性specificity
    :param true_labels: 真实标记，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :return:
    '''
    get_specificity_series.__name__ = 'specificity'
    # get_specificity_series.metric = 'specificity'
    cm_df = get_confusion_matrix_df(true_labels, pred_labels, unique_labels)  # 构建混淆矩阵
    # specificity_series=pd.Series(index=cm_df.index.tolist())
    diag_arr1 = np.diag(cm_df)  # 取出对角线
    diag_sum = np.sum(diag_arr1)  # 对角线求和
    pred_labels_arr1 = np.sum(cm_df, axis=0).values  # 每一类预测到的数量，并转成ndarray
    # cn = cm_df.shape[0] #不同类别的数量
    # specificities=[]
    # for i in range(0,cn):
    #     #根据公式计算每类的特异性，公式里是加，这里反着用减
    #     specificities.append((diag_sum-diag_arr1[i])/(pred_labels_arr1[i]-diag_arr1[i]+diag_sum-diag_arr1[i]))
    specificities = [(diag_sum - diag_arr1[i]) / (pred_labels_arr1[i] + diag_sum - 2 * diag_arr1[i])
                     for i in range(0, cm_df.shape[0])]  # 根据公式计算每类的特异性，公式里是加，这里反着用减
    # 将list转换成series
    specificity_series = pd.Series(data=specificities, index=cm_df.index.tolist())
    return specificity_series


def get_average_specificity(true_labels, pred_labels, unique_labels=None, is_weight=False):
    '''
    根据真实标记和预测标记，计算所有类别平均的specificity
    :param true_labels: 真实标记，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :return:
    '''
    get_average_specificity.__name__ = 'average_specificity'
    # get_average_specificity.metric = 'average_specificity'
    specificity_series = get_specificity_series(true_labels, pred_labels, unique_labels)
    if is_weight == True:
        cm_df = get_confusion_matrix_df(true_labels, pred_labels, unique_labels)  # 构建混淆矩阵
        class_ratio_series = np.sum(cm_df, axis=1) / np.sum(np.array(cm_df))  # 不转换成ndarray不行
        average_specificity = float(np.sum(specificity_series * class_ratio_series))
    else:
        average_specificity = float(np.mean(specificity_series))
    return average_specificity


def get_balanced_accuracy_series(true_labels, pred_labels, unique_labels=None):
    '''
    根据真实标记和预测标记，计算每个类别的sensitivity和specificity
    继而计算每个类别的balanced_accuracy，并构建成series结构
    balanced_accuracy=（sensitivity+specificity）/2
    :param true_labels: 真实标记，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    get_balanced_accuracy_series.__name__ = 'balanced_accuracy'
    # get_balanced_accuracy_series.metric = 'balanced_accuracy'
    sensitivity_series = get_sensitivity_series(true_labels, pred_labels, unique_labels)
    specificity_series = get_specificity_series(true_labels, pred_labels, unique_labels)
    balanced_accuracy = (sensitivity_series + specificity_series) / 2
    return balanced_accuracy


def get_average_balanced_accuracy(true_labels, pred_labels, unique_labels=None, is_weight=False):
    '''
    根据真实标记和预测标记，计算所有类别平均的balanced_accuracy
    :param true_labels: 真实标记，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :return:
    '''
    get_average_balanced_accuracy.__name__ = 'average_balanced_accuracy'
    # get_average_balanced_accuracy.metric = 'average_balanced_accuracy'
    balanced_accuracy_series = get_balanced_accuracy_series(true_labels, pred_labels, unique_labels)
    if is_weight == True:
        cm_df = get_confusion_matrix_df(true_labels, pred_labels, unique_labels)  # 构建混淆矩阵
        class_ratio_series = np.sum(cm_df, axis=1) / np.sum(np.array(cm_df))  # 不转换成ndarray不行
        average_balanced_accuracy = float(np.sum(balanced_accuracy_series * class_ratio_series))
    else:
        average_balanced_accuracy = float(np.mean(balanced_accuracy_series))
    return average_balanced_accuracy


# 函数的__name__属性在函数运行前不会改变，要改变只有两种方式，要么函数运行后，要么定义在函数后面
get_confusion_matrix_df.__name__ = 'confusion_matrix'
get_overall_accuracy.__name__ = 'overall_accuracy'
get_precision_series.__name__ = 'precision'
get_average_precision.__name__ = 'average_precision'
get_recall_series.__name__ = 'recall'
get_average_recall.__name__ = 'average_recall'
get_F_score_series.__name__ = 'F_score'
get_average_F_score.__name__ = 'average_F_score'
get_F1_score_series.__name__ = 'F1_score'
get_average_F1_score.__name__ = 'average_F1_score'
get_sensitivity_series.__name__ = 'sensitivity'
get_average_sensitivity.__name__ = 'average_sensitivity'
get_specificity_series.__name__ = 'specificity'
get_average_specificity.__name__ = 'average_specificity'
get_balanced_accuracy_series.__name__ = 'balanced_accuracy'
get_average_balanced_accuracy.__name__ = 'average_balanced_accuracy'


def get_eval(true_labels, pred_labels, unique_labels=None, is_weight=False):
    '''
    根据真实标记和预测标记，构建混淆矩阵，然后计算总体accuracy、每个类别以及所有类别平均的precision、recall、F1
    :param true_labels: 真实标记，序列，例如['A','B','C','A']
    :param pred_labels: 预测标记，序列，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    if unique_labels is None:  # 如果unique为空
        unique_labels = set(true_labels) | set(pred_labels)
    # 排序的标记集合，索引号即为混淆矩阵中对应的索引号
    sort_unique_labels = sorted(unique_labels)
    # 将true_labels,pred_labels中的标记转化为索引
    true_classes = [sort_unique_labels.index(true_label) for true_label in true_labels]
    pred_classes = [sort_unique_labels.index(pred_label) for pred_label in pred_labels]
    class_counts_arr1 = np.array(
        [true_classes.count(sort_unique_labels.index(label)) for label in sort_unique_labels])  # 每种类别的数量
    class_ratio_arr1 = class_counts_arr1 / len(true_classes)
    # MIN_VAL = 10e-8
    cn = len(sort_unique_labels)  # 类别数量 class number
    # np.set_printoptions(2)
    cm = np.zeros((cn, cn))  # 定义混淆矩阵 confusion matrix
    for true_class, pred_class in zip(true_classes, pred_classes):
        cm[true_class, pred_class] += 1
    true_label_arr1 = np.sum(cm, axis=1)
    pred_label_arr1 = np.sum(cm, axis=0)
    precision_arr1 = np.array([cm[i, i] / (pred_label_arr1[i] + MIN_VAL) for i in range(0, cn)])
    recall_arr1 = np.array([cm[i, i] / (true_label_arr1[i] + MIN_VAL) for i in range(0, cn)])
    F1_score_arr1 = 2.0 * precision_arr1 * recall_arr1 / (precision_arr1 + recall_arr1 + MIN_VAL)
    # 行为各类别，列为precision，recall，F1
    p_r_F1_arr2 = np.transpose(np.array([precision_arr1, recall_arr1, F1_score_arr1]))
    if is_weight == True:
        average_precision = float(np.sum(precision_arr1 * class_ratio_arr1))
        average_recall = float(np.sum(recall_arr1 * class_ratio_arr1))
        average_F1_score = float(np.sum(F1_score_arr1 * class_ratio_arr1))
    else:
        average_precision = float(np.mean(precision_arr1))
        average_recall = float(np.mean(recall_arr1))
        average_F1_score = float(np.mean(F1_score_arr1))
    average_p_r_F1_arr1 = np.array([average_precision, average_recall, average_F1_score])  # 所有类别平均的precision，recall，F1

    # 行为各类别，最后一行为平均值，列为precision，recall，F1
    res_p_r_F1_arr2 = np.zeros((cn + 1, 3))
    res_p_r_F1_arr2[:-1, :] = p_r_F1_arr2
    res_p_r_F1_arr2[-1, :] = average_p_r_F1_arr1

    # 总体accuracy
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    # 构建dataframe
    if is_weight == True:
        index = sort_unique_labels + ['Weighted_Average']
    else:
        index = sort_unique_labels + ['Average']
    columns = ['Precision(%)', 'Recall(%)', 'F1(%)']
    res_p_r_F1_a_df = pd.DataFrame(res_p_r_F1_arr2 * 100, index=index, columns=columns)

    res_p_r_F1_a_df.loc['Accuracy(%)', :] = accuracy * 100
    res_p_r_F1_a_df.loc['Accuracy(%)', 1:] = None

    res_p_r_F1_a_df.loc[:, 'Class_Count'] = None
    res_p_r_F1_a_df.loc[:-2, 'Class_Count'] = class_counts_arr1
    res_p_r_F1_a_df.loc[:, 'Class_Ratio'] = None
    res_p_r_F1_a_df.loc[:-2, 'Class_Ratio'] = class_ratio_arr1

    return res_p_r_F1_a_df


if __name__ == '__main__':
    # print(get_confusion_matrix_df.metric)
    true_labels = ['A', 'B', 'C', 'C', 'A', 'A', 'B', 'B', 'A', 'C']
    pred_labels = ['A', 'B', 'C', 'C', 'B', 'A', 'C', 'B', 'A', 'B']
    unique_labels = set(true_labels)

    cm_df = get_confusion_matrix_df(true_labels, pred_labels, unique_labels)
    print('cm_df:\n', cm_df)

    accuracy = get_overall_accuracy(true_labels, pred_labels)
    print('accuracy:\n', accuracy)

    precision_series = get_precision_series(true_labels, pred_labels)
    print('precision_series:\n', precision_series)
    average_precision = get_average_precision(true_labels, pred_labels)
    print('average_precision:\n', average_precision)

    recall_series = get_recall_series(true_labels, pred_labels)
    print('recall_series:\n', recall_series)
    average_recall = get_average_recall(true_labels, pred_labels)
    print('average_recall:\n', average_recall)

    F_score_series = get_F_score_series(true_labels, pred_labels, alpha=0.5)
    print('F_score_series:\n', F_score_series)
    average_F_score = get_average_F_score(true_labels, pred_labels, is_weight=True, alpha=0.5)
    print('average_F_score:\n', average_F_score)

    F1_score_series = get_F1_score_series(true_labels, pred_labels)
    print('F1_score_series:\n', F1_score_series)
    average_F1_score = get_average_F1_score(true_labels, pred_labels)
    print('average_F1_score:\n', average_F1_score)

    # res_p_r_F1_a_df=get_eval(true_labels, pred_labels, is_weight=True)
    # save_to_excel(res_p_r_F1_a_df,'result_test')

    sensitivity_series = get_sensitivity_series(true_labels, pred_labels)
    print('sensitivity_series:\n', sensitivity_series)
    average_sensitivity = get_average_sensitivity(true_labels, pred_labels)
    print('average_sensitivity:\n', average_sensitivity)

    specificity_series = get_specificity_series(true_labels, pred_labels)
    print('specificity_series:\n', specificity_series)
    average_specificity = get_average_specificity(true_labels, pred_labels)
    print('average_specificity:\n', average_specificity)

    balanced_accuracy_series = get_balanced_accuracy_series(true_labels, pred_labels)
    print('balanced_accuracy_series:\n', balanced_accuracy_series)
    average_balanced_accuracy = get_average_balanced_accuracy(true_labels, pred_labels)
    print('average_balanced_accuracy:\n', average_balanced_accuracy)

    # get_sensitivity_series.__name__='1'
    # print(get_sensitivity_series.__name__)