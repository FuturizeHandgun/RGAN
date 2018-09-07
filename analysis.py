import numpy as np
import tensorflow
import csv
import data_utils
import pandas as pd



def scale_linear_bycolumn(rawpoints, high=1.0, low=-1.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

#load data frame in
df = pd.read_csv('~/Downloads/sandp500/all_stocks_5yr.csv')
samples = []
#Load the stock data by each row
#Shape is num_samples * sequence_length(each stocks time series) * number of signal channels (high, open, close, low, volume at each time step)

#Everything needs to be an np.ndarray
for stock, df_stock in df.groupby('Name'):
    sequence = []
    for index,row in df_stock.iterrows():
        #print(list(row[1:6]))
        sequence.append(np.array(list(row[1:6])))
    sequence = np.array(sequence)
    if (len(sequence) == 1259):
        samples.append(sequence)
samples = np.array(samples)
#split into train,vali,test split. Normalize/scale don't work correctly
train, vali, test = data_utils.split(samples,[0.6,0.2,0.2],normalise=False)
train_labels,vali_labels,test_labels = None, None, None
#Check that values scale correctly
print("Before scaling")
print(train)
for i in range(len(train)):
    train[i] = scale_linear_bycolumn(train[i])
for i in range(len(vali)):
    vali[i] = scale_linear_bycolumn(vali[i])
for i in range(len(test)):
    test[i] = scale_linear_bycolumn(test[i])
print("After scaling")
print(train)
labels = dict()
labels['train'], labels['vali'], labels['test'] = train_labels, vali_labels, test_labels

samples = dict()
samples['train'], samples['vali'], samples['test'] = train, vali, test


pdf = None

data_path =  './experiments/data/' + "snp500"  + '.data.npy'
np.save(data_path,{'samples':samples,'pdf': pdf,'labels':labels})

