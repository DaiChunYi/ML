#測試中
#

import numpy as np #mainly useful for its N-dimensional array objects 
import pandas as pd  #Python data analysis library, including structures such as dataframes
import matplotlib as mpl #繪製資料圖表
import scipy as sp #基於numpy的科學計算工具,包括統計、線性代數等工具
from google.colab import drive #only Unix

#!gdown --id '1wNKAxQ29G15kgpBy_asjTcZRRgmsCZRm'
#!unzip data.zip
##data =pd.read_csv('gdrive/My Drive/hw1-regression/train.csv,header=None,encoding = 'big5')
data = pd.read_csv('.\\file.csv',encoding = 'big5')

#Preprocessing 前置處理 處理 ,不確認可不可以 先假設可以 需回頭確認
data = data.iloc[:, 3:] #先列後行 從第三行開始看到NR換成0
data[data == 'NR'] = 0
raw_data = data.to_numpy()

#Extract Features
month_data = []
for month in range(12): #12month
	sample = np.empty([18,480]) #row:18features,col:前20天的24個小時
	for day in range(20):
		sample[:day * 24:(day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month +day + 1),:]
		#sample[起始向量:終點向量:間隔] = raw_data[row,col] 相當於 raw_data[day,all of features of this day]
		#[從頭開始:]
	month_data[month] = sample	