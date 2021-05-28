# -*- coding: utf-8 -*-
# FURKAN HAYTA
#15070022
# SIGNAL Voice Classification

from scipy.io import wavfile
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fft import fft

celal = pd.DataFrame(os.listdir('0'))
ilber = pd.DataFrame( os.listdir('1'))
test = pd.DataFrame( os.listdir('test'))

celal = celal.rename(columns={0:'file'})
ilber = ilber.rename(columns={0:'file'})   
test = test.rename(columns={0:'file'}) 


data_celal = np.zeros((941,7),dtype=float)
data_ilber = np.zeros((1265,7),dtype=float)
data_test = np.zeros((359,7),dtype=float)

for index, row in celal.iterrows(): 
    y, sr = librosa.load("./0/"+row['file'], mono=True, duration=4)
    fft_=fft(y)
    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    data_celal[index][0]=np.mean(np.real(fft_))
    data_celal[index][1]=np.mean(rmse)
    data_celal[index][2]=np.mean(chroma_stft)
    data_celal[index][3]=np.mean(spec_cent)
    data_celal[index][4]=np.mean(spec_bw)
    data_celal[index][5]=np.mean(rolloff)
    data_celal[index][6]=np.mean(zcr)
    
    
for index, row in ilber.iterrows(): 
    y, sr = librosa.load("./1/"+row['file'], mono=True, duration=4)
    fft_=fft(y)
    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    data_ilber[index][0]=np.mean(np.real(fft_))
    data_ilber[index][1]=np.mean(rmse)
    data_ilber[index][2]=np.mean(chroma_stft)
    data_ilber[index][3]=np.mean(spec_cent)
    data_ilber[index][4]=np.mean(spec_bw)
    data_ilber[index][5]=np.mean(rolloff)
    data_ilber[index][6]=np.mean(zcr)

for index, row in test.iterrows(): 
    y, sr = librosa.load("./test/"+row['file'], mono=True, duration=4)
    fft_=fft(y)
    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    data_test[index][0]=np.mean(np.real(fft_))
    data_test[index][1]=np.mean(rmse)
    data_test[index][2]=np.mean(chroma_stft)
    data_test[index][3]=np.mean(spec_cent)
    data_test[index][4]=np.mean(spec_bw)
    data_test[index][5]=np.mean(rolloff)
    data_test[index][6]=np.mean(zcr)

df_celal = pd.DataFrame(data_celal)
df_test = pd.DataFrame(data_test)
df_ilber = pd.DataFrame(data_ilber)
df_celal['person']='0'
df_ilber['person']='1'

df = pd.concat([df_celal, df_ilber], ignore_index=True)

x_train = df.iloc[:,0:7]
y_train = df.iloc[:,7]
x_test = df_test.iloc[:,0:7]
ss = StandardScaler()
X_train = ss.fit_transform(x_train)
X_test = ss.transform(x_test)

from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)

import csv

with open('predicts2.csv', 'w', newline='') as file:
    fieldnames = ['FileName', 'Class']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for index, row in test.iterrows():    
        writer.writerow({'FileName': row['file'], 'Class':y_pred[index] })

