"""Staged training loop + CLI entry point for the differential-neural distinguisher."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

import speck32.model as model_module
import utils.PolytopicQadrupleGenerator as pqg
logging.basicConfig(level=logging.FATAL)

import speck32.cipher as speck
ABORT_TRAINING_BELOW_ACC = 0.505
EPOCHS = 120
NUM_SAMPLES = 10 ** 7
NUM_VAL_SAMPLES = 10 ** 5
BATCH_SIZE = 10000  # Reasonable batch size for GPU
POS_DELTAS = [(16384, 0), (0, 128), (32, 0)]
NEG_DELTAS = [(32, 0), (0, 1056), (0, 1026)]

def cyclic_lr(num_epochs, high_lr, low_lr):
    return lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)

def get_strategy():
    gpus = tf.config.list_physical_devices('GPU')
    return tf.distribute.OneDeviceStrategy(device='/gpu:0') if gpus else tf.distribute.get_strategy()

def train_one_round(model, X, Y, X_val, Y_val, round_number, epochs=40, model_name='model', load_weight_file=False, log_prefix='', lr_scheduler=None):
    if load_weight_file and os.path.exists(os.path.join(log_prefix, f'{model_name}_round{round_number - 1}.h5')):
        model.load_weights(os.path.join(log_prefix, f'{model_name}_round{round_number - 1}.h5'))

    checkpoint = ModelCheckpoint(os.path.join(log_prefix, f'{model_name}_round{round_number}.h5'), monitor='val_loss', save_best_only=True)
    callbacks = [checkpoint]
    if lr_scheduler is not None: callbacks.append(LearningRateScheduler(lr_scheduler))

    history = model.fit(X, Y, epochs=epochs, batch_size=BATCH_SIZE, validation_data=(X_val, Y_val), callbacks=callbacks, verbose=1)
    pd.to_pickle(history.history, os.path.join(log_prefix, f'{model_name}_training_history_round{round_number}.pkl'))
    return np.max(history.history['val_acc'])

def train_neural_distinguisher(starting_round, data_generator, model_name, input_size, log_prefix='./', _epochs=None, _num_samples=None):
    lr = cyclic_lr(10, 0.001, 0.0002)
    strategy = get_strategy()
    current_round = starting_round
    load_weight_file = False
    best_val_acc = 0
    best_round = starting_round
    epochs = _epochs if _epochs is not None else EPOCHS
    num_samples = _num_samples if _num_samples is not None else NUM_SAMPLES

    while True:
        with strategy.scope():
            model = model_module.make_model(input_size)
            model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True), loss='mse', metrics=['acc'])

        print(f'--- Training Round {current_round} ---')
        X, Y = data_generator(num_samples, current_round)
        X_val, Y_val = data_generator(NUM_VAL_SAMPLES, current_round)

        val_acc = train_one_round(model, X, Y, X_val, Y_val, current_round, epochs=epochs, load_weight_file=load_weight_file, log_prefix=log_prefix, model_name=model_name, lr_scheduler=lr)
        
        if val_acc <= ABORT_TRAINING_BELOW_ACC: break
        best_round, best_val_acc = current_round, val_acc
        current_round += 1
        load_weight_file = True
        tf.keras.backend.clear_session()

    return best_round, best_val_acc

def train_neural_distinguishers(output_dir='results', starting_round=1, epochs=None, num_samples=None, feature_mode='full'):
    os.makedirs(output_dir, exist_ok=True)
    plain_bits = 32
    input_size = {'raw': 4*plain_bits, 'diff': 3*plain_bits, 'full': 7*plain_bits}[feature_mode]

    def generator(n, nr):
        gen = pqg.PolytopicQuadrupleGenerator(
            encryption_function=speck.encrypt_wrapper,
            pos_diffs=POS_DELTAS, neg_diffs=NEG_DELTAS, 
            plain_bits=32, key_bits=64, nr=nr, 
            n_samples=n, batch_size=100000, # Data generation batch size
            feature_mode=feature_mode, use_gpu=True, to_float32=True
        )
        return gen[0]

    return train_neural_distinguisher(starting_round, generator, 'model', input_size, output_dir, epochs, num_samples)