#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import json

import tensorflow as tf
from tensorflow import keras
from keras import layers

from tools.datetime_formatter import DateTimeFormatter
from tools.display_time import display_time

#%% Functions

def load_multi_datasets(path, set_range):
    # set_range = set of setNo, i.e. which data should be loaded
    fname_all = []
    for setNo in set_range:
        # Get the file names for the current dataset
        curr_fname = '{}/setNo{}.npy'.format(path, setNo)
        # Append
        fname_all.append(curr_fname)
        
    print('fname is set')
    # Load data
    if 'input' in path:
        print('Loading inputs!')
        dataset = np.array([np.load(fname)[3:6, :] for fname in fname_all])
    else:
        print('Loading outputs!')
        dataset = np.array([np.load(fname) for fname in fname_all])
    
    return dataset


def load_dataset_into_tfdata(set_range, paths, batch_size, shuffle = False, buffer_size = None):
    # set_range = set of setNo, i.e. which data should be loaded
    # Load data
    path_inputs, path_outputs = paths
    inputs = load_multi_datasets(path_inputs, set_range)
    outputs = load_multi_datasets(path_outputs, set_range)
    print('inputs shape = {}'.format(inputs.shape))
    print('outputs shape = {}'.format(outputs.shape))
    
    # Register in tfdata
    tfdataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
    # Random shufle
    if shuffle == True:
        tfdataset = tfdataset.shuffle(buffer_size = buffer_size) # buffer size = how many samples are included to shuffle
    # Batching
    tfdataset = tfdataset.batch(batch_size)
    
    del inputs, outputs
    
    return tfdataset


def compile_and_fit(model, train_ds, val_ds, epochs):
    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = 'mean_squared_error'
    )
    
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = epochs
    )
    return history


# Save history
def save_history(history, fname):
    history_dict = history.history
    json.dump(history_dict, open(fname, 'w'))
    

# Plot history_dict
def plot_loss(history_dict):
    plt.plot(history_dict['loss'], label = 'loss')
    plt.plot(history_dict['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error (MSE)')
    plt.legend()
    plt.grid(True)
    
    
# Plot randomly selected prediction
def plot_ytrue_vs_ypred(y_true, y_pred, size = 10):
    # Lag calculation for x-axis
    from spatial_subsampling import get_all_grid_points
    from frequency_variogram import FrequencyVariogramRaw
    N_batch = 10
    dx = 0.5* 10**-3
    s_full = np.around(dx* get_all_grid_points(N_batch, N_batch), 10)
    maxlag = np.round(0.5* np.sqrt(2)* N_batch* dx, 10) # [m]
    cfv = FrequencyVariogramRaw(grid_spacing = dx, maxlag = maxlag)
    cfv.set_positions(s_full)
    lags = cfv.get_lags()
    # Add zero
    lags = np.concatenate((np.zeros(1), lags))
    
    # Random selection
    rng = np.random.default_rng()
    sets = rng.choice(np.arange(y_pred.shape[0]), size, replace = False)
    
    for setNo in sets:
        plt.figure()
        plt.title('y_true vs y_pred: setNo = {}'.format(setNo))
        plt.plot(lags, y_true[setNo, :], label = 'y_true')
        plt.plot(lags, y_pred[setNo, :], label = 'y_pred')
        plt.legend()
        plt.xlabel('lag')
        plt.ylabel('Normalized FV')
        


#%% Datasets: parameters & loading
# Path to load the data  
path_train_inputs = 'npy_data/ML/training/input_featuremapped'
path_train_outputs = 'npy_data/ML/training/outputs/fv_norm'
path_val_inputs = 'npy_data/ML/vali/input_featuremapped'
path_val_outputs = 'npy_data/ML/vali/outputs/fv_norm'
path_test_inputs = 'npy_data/ML/test/input_featuremapped'
path_test_outputs = 'npy_data/ML/test/outputs/fv_norm'

# Dimensions of single input & output
N_input = 81
N_output = 25

# tensorflow parameters
N_cddwindow = 3 # CNN window length = 10 neighboring bins are fed to estimate the FV of a single FV
batch_size = 512
reg_factor = 0.01
#buffer_size = n_output -> entire dataset
n_train = int(720000)
n_val = int(180000)
n_test = int(180000)

# Datasets for training & validation
start = time.time()

train_ds = load_dataset_into_tfdata(np.arange(n_train), [path_train_inputs, path_train_outputs], 
                                    batch_size, shuffle = True, buffer_size = n_train)
val_ds = load_dataset_into_tfdata(np.arange(n_val), 
                                  [path_val_inputs, path_val_outputs], batch_size, shuffle = True, buffer_size = n_val)

print('*** Data loading ***')
display_time(round(time.time() - start))
print(' ')
 
#import sys
#sys.exit()
#%% Model building & training

### Model Building ###
inputs = keras.Input(shape = (N_cddwindow, N_input))
# (1) Mask to ignore the missing inputs
x = layers.Masking(mask_value = -1.0, input_shape = (N_cddwindow, N_input))(inputs)
# (2) CNN + dropout
x = layers.Conv1D(32, 3, activation = "relu", kernel_regularizer = keras.regularizers.L1(reg_factor))(x)
x = layers.Dropout(rate = 0.2)(x)
## (3) CNN + dropout
#x = layers.Conv1D(32, 3, activation = "relu", kernel_regularizer = keras.regularizers.L1(reg_factor))(x)
#x = layers.Dropout(rate = 0.2)(x)
## (4) CNN + dropout
#x = layers.Conv1D(32, 3, activation = "relu", kernel_regularizer = keras.regularizers.L1(reg_factor))(x)
#x = layers.Dropout(rate = 0.2)(x)
# Flatten
x = layers.Flatten()(x)
x = layers.Dense(32, activation = "relu", kernel_regularizer = keras.regularizers.L1(reg_factor))(x)
outputs = layers.Dense(N_output)(x) # No activation
# Model architecture
model = keras.Model(inputs = inputs, outputs = outputs, name = 'fv_estimation')
model.summary()

#import sys
#sys.exit()

# Incorporate early stop
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 3)

# Setup for file name
dtf = DateTimeFormatter()
today = dtf.get_date_str()
del dtf

# Training
start = time.time()
epochs = 100
model.compile(optimizer = keras.optimizers.Adam(), loss = 'mean_squared_error', metrics = ['mse'])
history = model.fit(train_ds, validation_data = val_ds, epochs = epochs, callbacks = [early_stop])
    
print(' ')
print('#======================================================#')
print('End of training')
display_time(round(time.time() - start, 3))
print('#======================================================#')

# Plot
plot_loss(history.history)

# Save
path = 'tf_models/conv1d_{}/model'.format(today)
model.save(path)

# Free up the memory
#del train_ds, val_ds

#%% Test
# Load the existing model
#path2load = 'tf_models/conv1d_{}/model'.format(today)
#model = keras.models.load_model(path2load)

# Dataset
test_inputs = load_multi_datasets(path_test_inputs, np.arange(n_test))
test_outputs = load_multi_datasets(path_test_outputs, np.arange(n_test))

test_results = model.evaluate(test_inputs, test_outputs, return_dict = True)
# Prediction
y_pred = model.predict(test_inputs)
# Plots
plot_ytrue_vs_ypred(test_outputs, y_pred)

