from keras.utils import to_categorical
from keras.layers import Input, Dense, regularizers
from keras.models import Model
import numpy as np

a = np.array([0, 1, 1, 0])
b = to_categorical(a, num_classes=2)
inputs = Input(shape=(10,))
dense1 = Dense(units=200, #所有字符数量（不包含0）
                   activation='relu',
                   kernel_regularizer=regularizers.l2(0.0001))(inputs) #激活函数选取relu
outputs = Dense(units=2, #所有字符数量（不包含0）
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