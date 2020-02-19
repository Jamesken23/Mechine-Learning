import os   #系统模块
from keras.preprocessing.sequence import pad_sequences  #序列补全函数
import numpy as np
from keras.utils import to_categorical
from keras.layers import Input, Embedding, Dense, GRU, regularizers
from keras.models import Model, load_model  #模型
from keras.utils.vis_utils import plot_model    #模型图片存储函数
from datetime import datetime   #时间模块
from random import choice   #随机选择函数
import pickle   #数据存储和导出模块
import random   #随机模块
import sys


# os.environ['TF_CPP_MIN_LOG_LEBEL']='2'


def build_model(input_dim,output_dim):
    '''
    构建模型
    :param max_len: 模型序列的长度
    :param char_sum: 字符的总数量（不含0）
    :return:
    '''
    start_time = datetime.now()  # 程序开始时间
    #打印时间
    print('Start building model! Start Time:%s' % str(start_time))
    # output_dim = output_dim
    #输入
    inputs = Input(shape=(input_dim,))

    #relu激活函数
    dense1 = Dense(units=200, #所有字符数量（不包含0）
                   activation='relu',
                   kernel_regularizer=regularizers.l2(0.0001))(inputs) #激活函数选取relu
    # dense2 = Dense(units=20,
    #                activation='relu',
    #                kernel_regularizer_regularizer=regularizers.l2(0.0001))(dense1)

    outputs = Dense(units=output_dim, #所有字符数量（不包含0）
                    activation='softmax',
                    kernel_regularizer=regularizers.l2(0.0001))(dense1)  # 激活函数选取softmax,outputs
    model = Model(inputs=inputs,    #放入输入层
                  outputs=outputs)  #放入输出层,构建模型
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=10, nesterov=True)
    model.compile(loss='categorical_crossentropy',  #损失函数
                  optimizer='adam',  # rmsprop,adam，优化器
                  metrics=['categorical_accuracy']) #评价标准
    print('--The model parameters:')
    print(model.summary())  #打印模型基本形式
    #打印时间
    print('Finish building model! Used Time:%s' % str(datetime.now() - start_time))
    return model

def save_model(model, model_dir,model_name):
    '''
    存储模型到本地
    :param model: 训练好的模型
    :param model_dir: 模型存放目录
    :param model_name: 模型名称
    :return:
    '''
    #如果没有模型目录，构建目录
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    #拼接模型存储路径
    model_path = os.path.join(model_dir, model_name)+'.h5'
    print('Save model to %s' % model_path)
    #存储模型
    model.save(model_path)

def save_model_figure(model,model_dir,model_name):
    '''
    存储模型图片到本地
    :param model: 训练好的模型
    :param model_dir: 模型存放目录
    :param model_name: 模型名称
    :return:
    '''
    # 如果没有模型目录，构建目录
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # 拼接模型图片存储路径
    model_figure_path = os.path.join(model_dir, model_name)+'.png'
    print('Save model figure to %s' % model_figure_path)
    #存储模型图片
    plot_model(model,
               to_file=model_figure_path,#模型图片存储路径
               show_shapes=True,    #是否保存形状
               show_layer_names=True)   #每层命名

def load_my_model(model_dir,model_name):
    '''
    从本地加载模型
    :param model_dir: 模型目录
    :param model_name: 模型名称
    :return:
    '''
    #模型路径拼接
    model_path = os.path.join(model_dir, model_name) + '.h5'
    print('Load model from %s' % model_path)
    #加载模型
    model = load_model(model_path)
    return model

# def train_model_generator(model,train_data_size,train_data_generator, batch_size,epochs, verbose=2):
#     '''
#     train model
#     :param train_data: like [[1,4,2,3423,12,...],[23,523,323,...],...]
#     a number represents a word index
#     :param train_labels:  like train_data, but is labels respond to word
#     :return:
#     '''
#     start_time = datetime.now()  # 程序开始时间
#     print('Start training model! Start Time:%s' % str(start_time))
#     # print(train_data_arr2)
#     real_epochs=math.ceil(train_data_size/batch_size*epochs)
#     # print(real_epochs)
#     model.fit_generator(generator=train_data_generator,
#                         steps_per_epoch=batch_size,
#                         epochs=real_epochs,
#                         verbose=verbose,
#                         shuffle=True)
#     print('Finish training model! Used Time:%s' % str(datetime.now() - start_time))
#     return model

def train_model(model,input_data_arr2,output_data_arr2, batch_size,epochs, verbose=2,shuffle=True,validation_split=0.):
    '''
    训练模型
    :param model: 编译好的模型
    :param input_data_arr2: 可以直接投入训练的输入数据集
    :param output_data_arr2: 可以直接投入训练的输出
    :param batch_size: 分块训练的数据数量
    :param epochs: 训练迭代次数
    :param verbose: 输出版本信息选择
    :param shuffle: 是否打乱
    :param validation_split:    验证集的分割率
    :return:
    '''
    start_time = datetime.now()  # 程序开始时间
    #打印时间
    print('Start training model! Start Time:%s' % str(start_time))
    # print(train_data_arr2)
    # real_epochs=math.ceil(train_data_size/batch_size*epochs)
    # print(real_epochs)
    model.fit(x=input_data_arr2,
              y=output_data_arr2,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=validation_split,
              verbose=verbose,
              shuffle=shuffle)
    print('Finish training model! Used Time:%s' % str(datetime.now() - start_time))
    return model

def exists_model(model_dir,model_name):
    '''
    判断模型是否存在
    :param model_dir: 模型目录
    :param model_name: 模型名称
    :return:
    '''
    #模型路径拼接
    model_path = os.path.join(model_dir, model_name) + '.h5'
    # print('Load model from %s' % model_path)
    #判断文件是否存在
    if os.path.exists(model_path):
        return True
    return False


if __name__=='__main__':
    from MAHAKIL import MAHAKIL

    # 读取本地的非平衡数据集
    imbalanced_train_data_path = 'imbalanced_train_data_arr2.npy'
    imbalanced_train_data_arr2 = np.load(imbalanced_train_data_path)
    # 测试MAHAKIL方法
    balanced_data_arr2 = MAHAKIL(imbalanced_train_data_arr2)
    feature_arr2=balanced_data_arr2[:,:-1]
    label_arr1=balanced_data_arr2[:,-1]
    input_dim=feature_arr2.shape[1]
    output_dim=2
    label_arr2=to_categorical(label_arr1,num_classes=output_dim)
    model=build_model(input_dim,output_dim)
    model=train_model(model,
                      feature_arr2,
                      label_arr2,
                      batch_size=32,
                      epochs=30)