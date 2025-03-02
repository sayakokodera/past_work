#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
import json

import tensorflow as tf
from tensorflow import keras
from keras import layers

from tools.datetime_formatter import DateTimeFormatter
from tools.display_time import display_time
from tools.npy_file_writer import save_data

plt.close('all')

#%% Functions

def load_multi_datasets(path, set_range, dtype, N_convwindow = None):
    # set_range = set of setNo, i.e. which data should be loaded
    fname_all = []
    for setNo in set_range:
        # Get the file names for the current dataset
        curr_fname = '{}/{}/setNo{}.npy'.format(path, dtype, setNo)
        # Append
        fname_all.append(curr_fname)
        
    print('fname is set')
    # Load data
    if dtype == 'fv_norm':
        print('Loading fv_norm!')
        dataset = np.array([np.load(fname) for fname in fname_all])
    elif dtype == 'hist': 
        print('Loading histogram!')
        dataset = np.array([np.tile(np.load(fname), (N_convwindow, 1)) for fname in fname_all])
    elif dtype == 'fvmax':
        print('Loading fvmax!')
        if 'input' in path:
            dataset = np.array([np.reshape(np.load(fname), (N_convwindow, 1)) for fname in fname_all])
        else:
            dataset = np.array([np.load(fname) for fname in fname_all]) 
            dataset = np.reshape(dataset, (dataset.shape[0], 1))
        print('fvmax dataset shape = {}'.format(dataset.shape))
    else:
        raise AttributeError('dtype {} is not supported!'.format(dtype))
    
    # For input data: check the shape of the dataset
    if 'input' in path:  
        if dataset.shape[1] != N_convwindow:
            raise ValueError('Input data: dataset.shape[1] != N_convwindow')
    
    return dataset


def load_and_save_npy(path, datasize, N_convwindow):
    path_inputs, path_outputs = path
    # Inputs
    in_fvnorm = load_multi_datasets(path_inputs, np.arange(datasize), 'fv_norm', N_convwindow)
    in_hist = load_multi_datasets(path_inputs, np.arange(datasize), 'hist', N_convwindow)
    in_fvmax = load_multi_datasets(path_inputs, np.arange(datasize), 'fvmax', N_convwindow)
    # Save
    save_data(in_fvnorm, path_inputs, 'fv_norm_all.npy')
    save_data(in_hist, path_inputs, 'hist_all.npy')
    save_data(in_fvmax, path_inputs, 'fvmax_all.npy')
    
    # Inputs
    out_fvnorm = load_multi_datasets(path_outputs, np.arange(datasize), 'fv_norm')
    out_fvmax = load_multi_datasets(path_outputs, np.arange(datasize), 'fvmax')
    # Save
    save_data(out_fvnorm, path_outputs, 'fv_norm_all.npy')
    save_data(out_fvmax, path_outputs, 'fvmax_all.npy')
    
    # Store in a dictionary
    ds_dict = {
        'in_fv_norm' : in_fvnorm,
        'in_hist' : in_hist,
        'in_fvmax' : in_fvmax,
        'out_fv_norm' : out_fvnorm,
        'out_fvmax' : out_fvmax
        }
    
    return ds_dict
    

# TF data conversion: normalized FV
def convert2tfdata_fv_norm(inp_fv, inp_hist, out_fv, out_fvmax, batch_size, buffer_size, shuffle = False):
    # Register in tfdata
    tfdataset = tf.data.Dataset.from_tensor_slices((
            {'in_fv_norm': inp_fv, 'hist' : inp_hist}, 
            {'pred_fv_norm' : out_fv}
            ))
    
    # Random shufle
    if shuffle == True:
        tfdataset = tfdataset.shuffle(buffer_size = buffer_size) # buffer size = how many samples are included to shuffle
    # Batching
    tfdataset = tfdataset.batch(batch_size)
    
    return tfdataset

# TF data conversion: FV max
def convert2tfdata_fvmax(inp_fv_pred, inp_hist, inp_fvmax, out_fv, out_fvmax, fvmax_refval, batch_size, buffer_size, 
                        shuffle = False):
    # Scale the fvmax values
    inp_fvmax = inp_fvmax / fvmax_refval
    out_fvmax = out_fvmax / fvmax_refval
    
    # Register in tfdata
    tfdataset = tf.data.Dataset.from_tensor_slices((
            {'in_fv_norm': inp_fv, 'hist' : inp_hist, 'in_fvmax': inp_fvmax}, 
            {'pred_fv_norm' : out_fv, 'pred_fvmax': out_fvmax}
            ))
    
    # Random shufle
    if shuffle == True:
        tfdataset = tfdataset.shuffle(buffer_size = buffer_size) # buffer size = how many samples are included to shuffle
    # Batching
    tfdataset = tfdataset.batch(batch_size)
    
    return tfdataset


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
def plot_ytrue_vs_ypred(y_true, y_pred, x_input, size = 10, rnd_sets = None):
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
    #lags = np.concatenate((np.zeros(1), lags))
    
    # Random selection
    if rnd_sets is None:
        rng = np.random.default_rng()
        sets = rng.choice(np.arange(y_pred.shape[0]), size, replace = False)
    else:
        sets = np.copy(rnd_sets)
    
    for setNo in sets:
        plt.figure()
        plt.title('y_true vs y_pred: setNo = {}'.format(setNo))
        plt.plot(lags, y_true[setNo, :], label = 'y_true')
        plt.plot(lags, y_pred[setNo, :], label = 'y_pred')
        plt.plot(lags, x_input[setNo, 1, :], label = 'x_input')
        plt.legend()
        plt.xlabel('lag')
        plt.ylabel('Normalized FV')
        
        
# Save results
def save_results(setNo, y_true, y_pred, x_input):
    save_data(y_true[setNo, :], 'npy_data/ML/results/setNo_{}'.format(setNo), 'y_true.npy')
    save_data(y_pred[setNo, :], 'npy_data/ML/results/setNo_{}'.format(setNo), 'y_pred.npy')
    save_data(x_input[setNo, 1, :], 'npy_data/ML/results/setNo_{}'.format(setNo), 'x_input.npy')
        


#%% Datasets: parameters & loading
# Path to load the data  
path_train_inputs = 'npy_data/ML/train/input_scalarlags_smooth'
path_train_outputs = 'npy_data/ML/train/outputs'
path_vali_inputs = 'npy_data/ML/vali/input_scalarlags_smooth'
path_vali_outputs = 'npy_data/ML/vali/outputs'
path_test_inputs = 'npy_data/ML/test/input_scalarlags_smooth'
path_test_outputs = 'npy_data/ML/test/outputs'

# tensorflow parameters
N_lags = 25
N_convwindow = 3 # CNN window length = 10 neighboring bins are fed to estimate the FV of a single FV
batch_size = 128
reg_factor = 10**-8
reg_factor2 = 10**-12
#buffer_size = n_output -> entire dataset
n_train = int(720000)
n_vali = int(180000)
n_test = int(180000)

# Datasets for training & validation
fvmax_refval = round(10520.000, 5) # For normalizing the fvmax value
scaling = 20.0#15.0 / fvmax_refval
scaling_fvmax = 10**4
ds_date = '210511_smooth_normed'
load_existing_tfds = True
load_npy = True
save_tfds = True


if load_npy == True:
    # Training data
    train_in_fvnorm = np.load('{}/fv_norm_all.npy'.format(path_train_inputs))
    train_in_hist = np.load('{}/hist_all.npy'.format(path_train_inputs))
    #train_in_fvmax = np.load('{}/fvmax_all.npy'.format(path_train_inputs))[:, 1, 0][:, np.newaxis]
    train_out_fvnorm = np.load('{}/fv_norm_all.npy'.format(path_train_outputs))
    #train_out_fvmax = np.load('{}/fvmax_all.npy'.format(path_train_outputs))
    
    # Convert into tf.data.Dataset
    train_ds = convert2tfdata_fv_norm(train_in_fvnorm, train_in_hist, train_in_fvmax, train_out_fvnorm, 
                                   train_out_fvmax, fvmax_refval, batch_size, n_train, shuffle = True)
    
    #del train_in_fvnorm, train_in_hist, train_in_fvmax, train_out_fvnorm, train_out_fvmax
    
    # validation data
    vali_in_fvnorm = np.load('{}/fv_norm_all.npy'.format(path_vali_inputs))
    vali_in_hist = np.load('{}/hist_all.npy'.format(path_vali_inputs))
    #vali_in_fvmax = np.load('{}/fvmax_all.npy'.format(path_vali_inputs))[:, 1, 0][:, np.newaxis]
    vali_out_fvnorm = np.load('{}/fv_norm_all.npy'.format(path_vali_outputs))
    #vali_out_fvmax = np.load('{}/fvmax_all.npy'.format(path_vali_outputs))
    
    # Convert into tf.data.Dataset
    vali_ds = convert2tfdata_fv_norm(vali_in_fvnorm, vali_in_hist, vali_in_fvmax, vali_out_fvnorm, 
                                  vali_out_fvmax, fvmax_refval, batch_size, n_vali, shuffle = True)
    
    #del val_in_fvnorm, val_in_hist, val_out_fvnorm
    

elif load_existing_tfds == False:
    start = time.time()
    
    # Train dataset
    train_ds_dict = load_and_save_npy([path_train_inputs, path_train_outputs], n_train, N_convwindow)
    train_ds = convert2tfdata_fv_norm(train_ds_dict['in_fv_norm'], train_ds_dict['in_hist'], train_ds_dict['in_fvmax'], 
                                   train_ds_dict['out_fv_norm'], train_ds_dict['out_fvmax'], 
                                   batch_size, n_train, shuffle = True)
    
    # Validation dataset
    vali_ds_dict = load_and_save_npy([path_vali_inputs, path_vali_outputs], n_vali, N_convwindow)
    vali_ds = convert2tfdata_fv_norm(vali_ds_dict['in_fv_norm'], vali_ds_dict['in_hist'], vali_ds_dict['in_fvmax'], 
                                  vali_ds_dict['out_fv_norm'], vali_ds_dict['out_fvmax'], 
                                  batch_size, n_vali, shuffle = True)
    
    print('*** Data loading ***')
    display_time(round(time.time() - start))
    print(' ')
    
    if save_tfds == True:
        print('Save tf.dataset!')
        tf.data.experimental.save(train_ds, 'tf_models/datasets/{}/train_ds'.format(ds_date))
        tf.data.experimental.save(vali_ds, 'tf_models/datasets/{}/vali_ds'.format(ds_date))
    
 
else:
    print('Load the existing tf.dataset fomr {}!'.format(ds_date))
    
    element_spec = (
        {'in_fv': tf.TensorSpec(shape=(None, 3, 25), dtype=tf.float64, name=None), # inp_fv
         'hist': tf.TensorSpec(shape=(None, 3, 25), dtype=tf.float64, name=None), #hist
         'in_fvmax': tf.TensorSpec(shape=(None, 1), dtype=tf.float64, name=None)}, # in_fvmax
        {'pred_fv_norm': tf.TensorSpec(shape=(None, 25), dtype=tf.float64, name=None), # out_fv
         'pred_fvmax' : tf.TensorSpec(shape=(None, 1), dtype=tf.float64, name=None)} # out_fvmax
             )
    
    train_ds = tf.data.experimental.load(
            'tf_models/datasets/{}/train_ds'.format(ds_date), 
            element_spec
            )
    vali_ds = tf.data.experimental.load(
                'tf_models/datasets/{}/vali_ds'.format(ds_date), 
                element_spec
                )


#%% DNN for normalized FV

### Model Building ###
# (1) Register inputs
inp_fv = keras.Input(shape = (N_convwindow, N_lags), name = 'in_fv_norm') # normalized FV, input_1
inp_hist = keras.Input(shape = (N_convwindow, N_lags),  name = 'hist') # histogram, input_2
# (2) Preprocessing: rescaling
x_hist = layers.experimental.preprocessing.Rescaling(scaling)(inp_hist)
######### Prediction of normalized FV #########
# (3) Concatenate inputs = normalized_fv, histogram
x = layers.concatenate([inp_fv, x_hist])
# (4) 1-layer CNN
x = layers.Conv1D(32, 3, activation = "relu", kernel_regularizer = keras.regularizers.L1(reg_factor))(x)
x = layers.Dropout(rate = 0.2)(x)
# Flatten
x = layers.Flatten()(x)
# (5) Dense layers* 4 
x = layers.Dense(32, activation = "relu", kernel_regularizer = keras.regularizers.L1(reg_factor))(x)
x = layers.Dense(32, activation = "relu", kernel_regularizer = keras.regularizers.L1(reg_factor))(x)
x = layers.Dense(32, activation = "relu", kernel_regularizer = keras.regularizers.L1(reg_factor))(x)
x = layers.Dense(32, activation = "relu", kernel_regularizer = keras.regularizers.L1(reg_factor))(x)
# Prediction
out_fv = layers.Dense(N_lags, name = 'pred_fv_norm')(x) # normalized FV


# Build model
model = keras.Model(
    inputs = [inp_fv, inp_hist], 
    outputs = [out_fv], 
    name = 'fv_estimate'
    )
model.summary()

# Setup for file name
dtf = DateTimeFormatter()
today = dtf.get_date_str()
del dtf

#import sys
#sys.exit()

# Training
start = time.time()
epochs = 100
# Compile
model.compile(
        optimizer = keras.optimizers.Adam(), 
        loss = {
            'pred_fv_norm' : 'mean_squared_error'        
            }, 
        metrics = ['mse']
        )
# Incorporate early stop
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 3, 
                                           restore_best_weights = True)

# Fit
history = model.fit(
        train_ds, 
        validation_data = vali_ds, 
        epochs = epochs, 
        callbacks = [early_stop]
        )
    
print(' ')
print('#======================================================#')
print('End of training')
display_time(round(time.time() - start, 3))
print('#======================================================#')

# Plot
plot_loss(history.history)

# Save
#path = 'tf_models/conv1d_{}/model'.format(today)
#model.save(path)


#%% Prediction of scaling

######## Prediction of scaling, fv_max #########
# (1) Register inputs
inp_fvmax = keras.Input(shape = (1,),  name = 'in_fvmax') # fv_max, input_3
# () Rescaling
x_fvmax = layers.experimental.preprocessing.Rescaling(scaling_fvmax)(inp_fvmax)
# (6) Concatenate inputs = scaled fvmax, predicted normalized fv, scaled histogram
x2 = layers.concatenate([x_fvmax, out_fv, x_hist[:, 0, :]])
# (7) Dense nets: non-negative constraints + dropout
x2 = layers.Dense(32, activation = "relu", kernel_regularizer = keras.regularizers.L1(reg_factor2))(x2)
x2 = layers.Dropout(rate = 0.2)(x2)

x2 = layers.Dense(32, activation = "relu", kernel_regularizer = keras.regularizers.L1(reg_factor2))(tf.abs(x2))
x2 = layers.Dropout(rate = 0.2)(x2)

x2 = layers.Dense(32, activation = "relu", kernel_regularizer = keras.regularizers.L1(reg_factor2))(tf.abs(x2))
x2 = layers.Dropout(rate = 0.2)(x2)

x2 = layers.Dense(32, activation = "relu", kernel_regularizer = keras.regularizers.L1(reg_factor2))(tf.abs(x2))
# Prediction
out_fvmax = layers.Dense(1)(tf.abs(x2)) # max. FV value, i.e. scaling
out_fvmax = layers.multiply([out_fvmax, tf.math.sign(out_fvmax)], name = 'pred_fvmax')


#%% Test
# Load the existing model
#path2load = 'tf_models/conv1d_{}/model'.format(today)
#model = keras.models.load_model(path2load)
if load_npy == True:
    # Load
    test_in_fv_norm = np.load('{}/fv_norm_all.npy'.format(path_test_inputs))
    test_in_hist = np.load('{}/hist_all.npy'.format(path_test_inputs))
    test_in_fvmax = np.load('{}/fvmax_all.npy'.format(path_test_inputs))[:, 1, 0][:, np.newaxis]
    test_out_fv_norm = np.load('{}/fv_norm_all.npy'.format(path_test_outputs))
    test_out_fvmax = np.load('{}/fvmax_all.npy'.format(path_test_outputs))
    
    # Scaling the fvmax values
    test_in_fvmax = test_in_fvmax / fvmax_refval
    test_out_fvmax = test_out_fvmax / fvmax_refval
    
    test_inputs = [test_in_fv_norm, test_in_hist, test_in_fvmax]
    test_outputs = [test_out_fv_norm, test_out_fvmax]
    
    
elif load_existing_tfds == False:
    
    # Load and save
    test_ds_dict = load_and_save_npy([path_test_inputs, path_test_outputs], n_test, N_convwindow)
    
    test_inputs = [test_ds_dict['in_fv_norm'], test_ds_dict['in_hist'], test_ds_dict['in_fvmax']]
    test_outputs = [test_ds_dict['out_fv_norm'], test_ds_dict['out_fvmax']]
    

else:
    print('Load the existing tf.dataset fomr {}!'.format(ds_date))
    test_ds = tf.data.experimental.load(
            'tf_models/datasets/{}/test_ds'.format(ds_date), 
            element_spec
            )

test_results = model.evaluate(test_inputs, test_outputs, return_dict = True)
# Prediction
y_pred = model.predict(test_inputs)    


# Plots
rnd_sets = np.random.randint(0, n_test, 5)

# Normalized FV
plot_ytrue_vs_ypred(test_outputs[0], y_pred[0], rnd_sets = rnd_sets)

# Scale
fv_true = test_out_fv_norm* test_out_fvmax* fvmax_refval
fv_pred = y_pred[0]* y_pred[1]* fvmax_refval
plot_ytrue_vs_ypred(fv_true, fv_pred, rnd_sets = rnd_sets)


