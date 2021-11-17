import h5py
import numpy as np
import pandas as pd
from sklearn import preprocessing
from pickle import dump

# Dataset
filename = 'data/N-CMAPSS/N-CMAPSS_DS03-012.h5'

# Load data
with h5py.File(filename, 'r') as hdf:
    # Development set
    W_dev = np.array(hdf.get('W_dev'))  # W
    X_s_dev = np.array(hdf.get('X_s_dev'))  # X_s
    X_v_dev = np.array(hdf.get('X_v_dev'))  # X_v
    Y_dev = np.array(hdf.get('Y_dev'))  # RUL
    A_dev = np.array(hdf.get('A_dev'))  # Auxiliary

    # Test set
    W_test = np.array(hdf.get('W_test'))  # W
    X_s_test = np.array(hdf.get('X_s_test'))  # X_s
    X_v_test = np.array(hdf.get('X_v_test'))  # X_v
    Y_test = np.array(hdf.get('Y_test'))  # RUL
    A_test = np.array(hdf.get('A_test'))  # Auxiliary

    # Variables names
    W_var = np.array(hdf.get('W_var'))
    X_s_var = np.array(hdf.get('X_s_var'))
    X_v_var = np.array(hdf.get('X_v_var'))
    A_var = np.array(hdf.get('A_var'))

    # from np.array to list dtype U4/U5
    W_var = list(np.array(W_var, dtype='U20'))
    X_s_var = list(np.array(X_s_var, dtype='U20'))
    X_v_var = list(np.array(X_v_var, dtype='U20'))
    A_var = list(np.array(A_var, dtype='U20'))

W = np.concatenate((W_dev, W_test), axis=0)
X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
Y = np.concatenate((Y_dev, Y_test), axis=0)
A = np.concatenate((A_dev, A_test), axis=0)

print("W_dev shape: " + str(W_dev.shape))
print("W_test shape: " + str(W_test.shape))
print("W shape: " + str(W.shape))
print("X_s shape: " + str(X_s.shape))
print("X_v shape: " + str(X_v.shape))
print("A shape: " + str(A.shape))

df_A = pd.DataFrame(data=A, columns=A_var)
print('Engine units in df: ', np.unique(df_A['unit']))

# for i in np.unique(df_A['unit']):
#     print('Unit: ' + str(i) + ' - Number of flight cyles (t_{EOF}): ',
#           len(np.unique(df_A.loc[df_A['unit'] == i, 'cycle'])))

# Create new DataFrame to combine all data
data_AWXY = pd.DataFrame(data=np.hstack((df_A['unit'].values.reshape(-1, 1), W, X_s, X_v, Y)),
                         columns=['unit'] + W_var + X_s_var + X_v_var + ['RUL'])
print('All Data Shape: ', data_AWXY.shape)

# MinMax normalization of features
cols_normalize = data_AWXY.columns.difference(['unit', 'RUL'])
min_max_scaler = preprocessing.MinMaxScaler()
norm_data_AWXY = pd.DataFrame(min_max_scaler.fit_transform(data_AWXY[cols_normalize]),
                              columns=cols_normalize,
                              index=data_AWXY.index)
join_df = data_AWXY[data_AWXY.columns.difference(cols_normalize)].join(norm_data_AWXY)
data_AWXY = join_df.reindex(columns=data_AWXY.columns)

# MinMax normalization of target variable
target_scaler = preprocessing.MinMaxScaler()
data_AWXY['RUL'] = target_scaler.fit_transform(data_AWXY['RUL'].values.reshape(-1, 1))

# Split data into train (units 1 to 9), validation (10 to 12), and test (units 13 to 15) dataset.
train_df = data_AWXY[(data_AWXY.unit <= 9)]
val_df = data_AWXY[(data_AWXY.unit >= 10) & (data_AWXY.unit <= 12)]
test_df = data_AWXY[(data_AWXY.unit >= 13)]

# Save scaler object for later use
dump(target_scaler, open('data/N-CMAPSS/target_scaler_DS03.pkl', 'wb'))

train_df.to_csv('data/N-CMAPSS/train_DS03.csv', sep=' ', float_format='%.3f')
val_df.to_csv('data/N-CMAPSS/val_DS03.csv', sep=' ', float_format='%.3f')
test_df.to_csv('data/N-CMAPSS/test_DS03.csv', sep=' ', float_format='%.3f')
