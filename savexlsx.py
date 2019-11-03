'''
@Descripttion: 
@Version: 
@Author: zh
@Date: 2019-08-10 10:02:43
@LastEditors: zh
@LastEditTime: 2019-08-13 18:22:40
'''
# -*- coding: utf-8 -*-

import csv
import os
import time

import keras
import numpy as np
import openpyxl
from keras.layers import (LSTM, Conv1D, Dense, Dropout, Flatten, Input,
                          MaxPool1D, Reshape, concatenate)
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model, load_model
from keras.utils import to_categorical
from scipy import signal

from pydecodingMdata import DecodingMdata

# sampling rate
Fs = 100
# train_step
n_inter = 20
dic = {0: 'w', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}

#TODO:需要添加数据采样频率不一致时，数据转换接口
def convert_160hz_100hz(input_array):
    """
    睡眠分期，均是以30s为一帧数据
    该函数的目标是将采样频率为160hz的信号转换为100hz
    由此可以得到：
    输入：150X160X１ 1D
    输出：150X100X1  1D

    计算方式：每个８个数据拟合一条直线，然后等分取点
    """
    input_array_reshape= input_array[:24000].reshape(3000,8)
    output_array =[]
    for input_array_split in input_array_reshape:
        x = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
        y = input_array_split
        # print(np.array(x).shape)
        # print(np.array(y).shape)
        z1 = np.polyfit(x,y,3)#采用三项式拟合
        p1 = np.poly1d(z1)#拟合公式
        xvals = [1.6,3.2,4.8,6.4,8.0]#等分该段内的所有数据
        # print(xvals)
        yvals = p1(xvals)
        # print(np.array(yvals).shape)
        output_array.append(yvals)
    return np.array(output_array).reshape(15000,1)

def makeConvLayers(inputLayer):
    # two conv-nets in parallel for feature learning,
    # one with fine resolution another with coarse resolution
    # network to learn fine features
    convFine = Conv1D(filters=64, kernel_size=int(Fs / 2), strides=int(Fs / 16), padding='same', activation='relu',
                      name='fConv1')(inputLayer)
    convFine = MaxPool1D(pool_size=8, strides=8, name='fMaxP1')(convFine)
    convFine = Dropout(rate=0.5, name='fDrop1')(convFine)
    convFine = BatchNormalization()(convFine)
    convFine = Conv1D(filters=128, kernel_size=8, padding='same', activation='relu', name='fConv2')(convFine)
    convFine = BatchNormalization()(convFine)
    convFine = Conv1D(filters=128, kernel_size=8, padding='same', activation='relu', name='fConv3')(convFine)
    convFine = BatchNormalization()(convFine)
    convFine = Conv1D(filters=128, kernel_size=8, padding='same', activation='relu', name='fConv4')(convFine)
    convFine = MaxPool1D(pool_size=4, strides=4, name='fMaxP2')(convFine)
    convFine = BatchNormalization()(convFine)
    fineShape = convFine.get_shape()
    convFine = Flatten(name='fFlat1')(convFine)

    # network to learn coarse features
    convCoarse = Conv1D(filters=32, kernel_size=Fs * 4, strides=int(Fs / 2), padding='same', activation='relu',
                        name='cConv1')(inputLayer)
    convCoarse = MaxPool1D(pool_size=4, strides=4, name='cMaxP1')(convCoarse)
    convCoarse = Dropout(rate=0.5, name='cDrop1')(convCoarse)
    convFine = BatchNormalization()(convFine)
    convCoarse = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu', name='cConv2')(convCoarse)
    convFine = BatchNormalization()(convFine)
    convCoarse = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu', name='cConv3')(convCoarse)
    convFine = BatchNormalization()(convFine)
    convCoarse = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu', name='cConv4')(convCoarse)
    convCoarse = MaxPool1D(pool_size=2, strides=2, name='cMaxP2')(convCoarse)
    convFine = BatchNormalization()(convFine)
    coarseShape = convCoarse.get_shape()
    convCoarse = Flatten(name='cFlat1')(convCoarse)

    # concatenate coarse and fine cnns
    mergeLayer = concatenate([convFine, convCoarse], name='merge')

    outLayer = Dropout(rate=0.5, name='mDrop1')(mergeLayer)
    return mergeLayer, (coarseShape, fineShape)


def preTrainingNet():
    inLayer = Input(shape=(3000, 1), name='inLayer')
    mLayer, (_, _) = makeConvLayers(inLayer)
    outLayer = Dense(5, activation='softmax', name='outLayer')(mLayer)
    network = Model(inLayer, outLayer)
    network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return network

def norm(data):
    return((data-data.min())/(data.max()-data.min()))

def load_data(data_path):
    p= DecodingMdata()
    data = []
    senGroup=[0,0,0,0]
    Group =  [list() for i in range (4)]
    with open(data_path) as f:
        # print(data_path)
        for line in f.readlines()[:-1]:
            dic_eeg = {}
            p.setDataString(line.strip())
            eegData = p.getEEGList()
            #TODO:获得其他测量数据
            Group[0].append(p.getHr_data())
            Group[1].append(p.getSpo_data())
            Group[2].append(p.getTemperature_data())
            Group[3].append(p.getBody_state_data())
            dic_eeg['Hr_data'] = p.getHr_data()
            dic_eeg['Temperature_data'] = p.getTemperature_data()
            dic_eeg['Spo_data'] = p.getSpo_data()
            dic_eeg['Body_state_data'] = p.getBody_state_data()
            dic_eeg['eegData'] = eegData
            dic_eeg['channel_state_data'] = p.getEEG_channel_state_data()
            dic_eeg['Aacx'] = p.getAacxResult()
            dic_eeg['Aacy'] = p.getAacyResult()
            dic_eeg['Aacz'] = p.getAaczResult()
            dic_eeg['Gyrox'] = p.getGyroxResult()
            dic_eeg['Gyroy'] = p.getGyroyResult()
            dic_eeg['Gyroz'] = p.getGyrozResult()
            # jsondata.append(dic_eeg)
            data.extend(eegData)
        # 心率血氧额温体位均取一个文件中所有数据的均值显示
        Group = np.array(Group)
        senGroup[0] = round(non_zero_mean(Group[0]))        #心率
        senGroup[1] = round(non_zero_mean(Group[1]))        #血氧
        senGroup[2] = non_zero_mean(Group[2])               #额温
        g3 = list(map(int, Group[3]))
        senGroup[3] = np.argmax(np.bincount(g3))      #体位
    # file = open(data_path).readlines()
    # 需要凑成4800条数据
    # 因此，80*6
    b, a = signal.butter(5, (0.5 * 2 / 160.0, 45 * 2 / 160.0), 'bandpass')
    eeg = signal.filtfilt(b, a, data)
    eeg = convert_160hz_100hz(eeg)
    eegNorm = norm(eeg)
    # eeg[np.abs(eeg)>200] = 0.19       
    return eegNorm,senGroup,eeg

def file_name(file_dir):
    '''
    @ Description: 读取所有文件名
    @ Param : 根目录
    @ Return: 数据目录
    '''
    filepath  = []
    path_list = os.listdir(file_dir)
    path_list.sort()
    for file in path_list:  
        if os.path.splitext(file)[1] == '.txt':
            filepath.append(os.path.join(file_dir,file))             
    return filepath

def non_zero_mean(data):
    '''
    @ Description: 求非0值的平均值
    @ Param : 非0列向量
    @ Return: 均值
    '''
    data_ = 0
    exist = (data != 0)
    num = data.sum(axis=0)
    den = exist.sum(axis=0)
    if den != 0:
        data_ = float("%.2f" % (num/den))
    return data_
    
def main(data_path, w_path):
    keras.backend.clear_session()
    #TODO:重新修改数据解析方法
    x_,senGroup,_= load_data(data_path)
    split_x = x_.reshape(5,3000)
    list_pred = []
    sum_pred = 0
    for x in split_x:
        x = x[:3000].reshape(-1, 3000).astype(np.float32).T
        x = x[np.newaxis, :, :]
        model = preTrainingNet()
        model.load_weights(w_path)
        pred = np.argmax(model.predict(x), axis=1)
        print(pred)
        print('============')
        pred_value = pred[0]
        # sum_pred = sum_pred+pred_value
    # true_pred = int(sum_pred/5.0)
        list_pred.append(pred_value)
    true_pred = np.argmax(np.bincount(list_pred))
    # label =[]
    label = dic[true_pred]
    print("预测结果: %s" % label)
    return label,senGroup

w_path = '1.h5'
file_dir = '/home/zh/dataset/20190813'
file_path = file_name(file_dir)    
path ='/home/zh/dataset/20190813/0746.xlsx'
savepath = '/home/zh/dataset/20190813/111.xlsx'
wb = openpyxl.load_workbook(path)
ws = wb.worksheets[0]
ws.delete_cols(1,3)
ws.insert_cols(2)
ws.insert_cols(6)
for i,file in enumerate(file_path):
    print('==========第'+str(i+1)+'次预测==========')
    label,senGroup = main(file, w_path)
    for j ,row in enumerate(ws.rows):
        if j==0:
            row[1].value ='新心率'
            row[5].value ='新分期'
        elif j==i+1:
            row[1].value = senGroup[0]
            row[5].value = label
    wb.save(savepath)