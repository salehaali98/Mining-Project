from keras import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import scipy.stats as stats

# pd.read_csv('WISDM_ar_v1.1_raw.txt')
file = open('WISDM_ar_v1.1_raw.txt')
lines = file.readlines()
list = []
for i, line in enumerate(lines):
    try:
        line = line.split(',')
        last = line[5].split(';')[0]
        if last == ' ':
            break;
        temp = [line[0], line[1], line[2], line[3], line[4] , last]
        list.append(temp)
    except:
        print('error at line number', i)

#print(list)
#array = np.array(list)
# print(array)

columns = ['userID', 'act', 'timestamp', 'x', 'y', 'z']
data = pd.DataFrame(data= list, columns=['userID', 'act', 'timestamp', 'x', 'y', 'z'])
print(data.head())
data.shape
data.info()
#data.isnull.sum()
activities =data['act'].index.value_counts()
#balanced data
data['x'] = data['x'].replace('[^\d.]', '', regex = True).replace('', 0).astype(float)
data['y'] = data['x'].replace('[^\d.]', '', regex = True).replace('', 0).astype(float)
data['z'] = data['x'].replace('[^\d.]', '', regex = True).replace('', 0).astype(float)

plottime= 20
data.info()
print("after conversion")
#Plotting


df = data.drop(['userID', 'timestamp'], axis = 1).copy()
df['act'].value_counts()
walking = df[df['act']== 'Walking'].head(3555).copy()
Jogging = df[df['act']== 'Jogging'].head(3555).copy()
Downstairs = df[df['act']== 'Downstairs'].head(3555).copy()
Sitting= df[df['act']== 'Sitting'].head(3555).copy()
Standing = df[df['act']== 'Standing'].head(3555).copy()
Upstairs = df[df['act']== 'Upstairs'].head(3555).copy()
bdf = pd.DataFrame()
balanced_data = bdf.append([walking,Jogging,Downstairs,Sitting,Standing,Upstairs])
balanced_data.shape

label =preprocessing.labelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['act'])
label.classes_
#standardize
x = balanced_data[['x, y, z']]
y = balanced_data['label']

scaler = StandardScaler()
x= scaler.fit_transform(x)
scaled_x = pd.DataFrame(data = x, columns =['x, y, z'])
scaled_X['Label']= y.values
scaled_x

framesize = plottime*4
jump_size = plottime*2
def Frames(df, framesize, jump_size):
    features = 3
    frames = []
    labels= []
    for i in range(0, len(df)- framesize, jumpsize):
        x= df['x'].values[i: i+framesize]
        y = df['x'].values[i: i + framesize]
        z = df['x'].values[i: i + framesize]

        label = stats.mode(df['label'][i: i+framesize])[0][0]
        frames.append([x,y,z])
        labels.append(label)

        frames = np.asarray(frames).reshape(-1, framesize, features)
        labels = np.asarray(labels)
        return frames, labels
    X, y = frames(scaled_x, framesize, jump_size)
    x.shape, y.shape