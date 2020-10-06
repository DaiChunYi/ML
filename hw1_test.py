import pandas as pd
import sys

data = pd.read_csv('train_datas_0.csv',encoding = 'big5')
data2 = pd.read_csv('train_datas_1.csv',encoding = 'big5')
import numpy as np

raw_data = data.to_numpy()
raw_data2 = data2.to_numpy()

raw_data = np.concatenate([raw_data,raw_data2])
print(np.shape(raw_data)) 
print('raw_data to training data...')

for i in range(len(raw_data)):
    for j in range(15):
        if raw_data[i][j] == '-':
            raw_data[i][j] = raw_data[i-1][j]

trainX = np.empty([len(raw_data)-9, 15*9*2],dtype=float) #weight:15 features * 9 hours
trainY = np.empty([len(raw_data)-9,1],dtype=float)

for i in range(len(trainX)):
    trainY[i] = raw_data[i+9][10]   #col.10 = pm2.5
    for j in range(9):
        for k in range(15):
            trainX[i][j * 15 + k] = raw_data[i + j][k]
            trainX[i][15*9+j * 15 + k] = raw_data[i + j][k]
            trainX[i][15*9+j * 15 + k] = trainX[i][15*9+j * 15 + k]**2

#print(Mean)
#for i,s in enumerate(trainX):
 #   print(s)

print('feature_scale...')


mean = np.mean(trainX,axis=0,dtype=float)
#print(mean)
std = np.std(trainX,axis=0)
for i in range(len(trainX)):
    for j in range(15*9*2):
        if std[j] != 0:
            trainX[i][j] = (trainX[i][j] - mean[j] ) / std[j]
#print(trainX)
print('feature_scale finish')

#def Lossfunction(X,w,y):
#    return np.sqrt(np.sum(np.power(np.dot(X,w) - y, 2))/(15*9))
#def gradient(W,Y):
#    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)
'''
import math
trainX ,trainX,trainX,trainX = trainX[:math.floor(len(trainX)*0.8),:]
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
'''
#initialized
dim = 15*9*2 +1
w = np.zeros([dim,1])
trainX = np.concatenate((np.ones([len(trainX), 1]), trainX), axis = 1).astype(float)
#b = np.zeros([dim,1])
learning_rate = 10
#reg = 10
time = 100000
adagrad = np.zeros([dim,1])
eps = 0.0000000001

for i in range(time):
    if(i%10000==0):
        loss = np.sqrt(np.sum(np.power(np.dot(trainX,w) - trainY, 2))/len(trainX))
        print(loss)
    gradient = 2 * np.dot(trainX.transpose(), np.dot(trainX, w) - trainY) 
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight2',w)

test_data = pd.read_csv('test_datas.csv', header = None, encoding = 'big5')
test_data = test_data.iloc[1:, :]
test_data = test_data.to_numpy()
test_x = np.empty([500, 15*9*2], dtype = float)
for i in range(500):
    test_x[i, :15*9] = test_data[9 * i: 9* (i + 1), :].reshape(1, -1)
    test_x[i, 15*9:] = test_data[9 * i: 9* (i + 1), :].reshape(1, -1)
    test_x[i, 15*9:] = test_x[i, 15*9:]**2
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean[j]) / std[j]
test_x = np.concatenate((np.ones([500, 1]), test_x), axis = 1).astype(float)

w = np.load('weight2.npy')
ans_y = np.dot(test_x, w)
# negative
for i in range(len(ans_y)):
    if ans_y[i] < 0:
        ans_y[i] = mean[10]
ans_y

import csv
with open('submit3.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(500):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)
