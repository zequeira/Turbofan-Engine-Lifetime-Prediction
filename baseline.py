import wget
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Setting a seed for reproducibility
np.random.seed(1234)

# Data ingestion - reading the datasets from Azure blob
wget.download('http://azuremlsamples.azureml.net/templatedata/PM_train.txt', out='data/')
wget.download('http://azuremlsamples.azureml.net/templatedata/PM_test.txt', out='data/')
wget.download('http://azuremlsamples.azureml.net/templatedata/PM_truth.txt', out='data/')

# read training data
train_df = pd.read_csv('data/PM_train_FD001.txt', sep=" ", header=None)
# remove the last two columns that contains just NaN
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
sensors = [f's{j}' for j in range(1, 22)]
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + sensors

# read test data
test_df = pd.read_csv('data/PM_test_FD001.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + sensors

# read ground truth data
truth_df = pd.read_csv('data/PM_truth_FD001.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

train_df = train_df.sort_values(['id', 'cycle'])
train_df.head()

# Data Labeling - generate column RUL
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_df = train_df.merge(rul, on=['id'], how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)

# generate label columns for training data
# "w1" is used for the binary classification problem: Is this engine going to fail within w1 cycles? ("label1")
# "w0" is used for Multi-class classification: Is this engine going to fail within the window [1, w0] cycles
# or to fail within the window [w0+1, w1] cycles, or it will not fail within w1 cycles?
w1 = 30
w0 = 15
train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0)
train_df['label2'] = train_df['label1']
train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2

# MinMax normalization of train data
cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL', 'label1', 'label2'])
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                             columns=cols_normalize,
                             index=train_df.index)
join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
train_df = join_df.reindex(columns=train_df.columns)

# MinMax normalization of test data
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                            columns=cols_normalize,
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns=test_df.columns)
test_df = test_df.reset_index(drop=True)

# generate column max for test data
rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
truth_df.columns = ['more']
truth_df['id'] = truth_df.index + 1
truth_df['max'] = rul['max'] + truth_df['more']
truth_df.drop('more', axis=1, inplace=True)

# generate RUL for test data
test_df = test_df.merge(truth_df, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max', axis=1, inplace=True)

# generate label columns w0 and w1 for test data
test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )
test_df['label2'] = test_df['label1']
test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2

# Remove sensor columns not carrying any information:
# i.e., sensors 1, 5, 10, 16, 18 and 19
sensors_drop = ['s1', 's5', 's10', 's16', 's18', 's19']
train_df.drop(sensors_drop, axis=1, inplace=True)
test_df.drop(sensors_drop, axis=1, inplace=True)
