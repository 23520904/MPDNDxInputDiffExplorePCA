"""Staged training loop + CLI entry point for the differential-neural distinguisher.

Run as a script (`python train.py -o results/`) to train from round 5. Importing
this module (e.g. `import train`) does NOT start training -- only `python
train.py` or an explicit call to `main()` does.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # filter TF C++ logs; must be set before tf import
import warnings
warnings.filterwarnings("ignore")

import argparse
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

import data
import model as model_module
import utils.PolyhedralMultiPairGenerator as pmpg
logging.basicConfig(level=logging.FATAL)

import speck32.cipher as speck
ABORT_TRAINING_BELOW_ACC = 0.505  # stop staged training once val accuracy drops to/below this
EPOCHS = 120
NUM_SAMPLES = 10 ** 7
NUM_VAL_SAMPLES = 10 ** 6
BATCH_SIZE = 10000
POS_DELTAS = [(16384, 0), (0, 128), (32, 0)]
NEG_DELTAS = [(32, 0), (0, 1056), (0, 1026)]


def cyclic_lr(num_epochs, high_lr, low_lr):
    return lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)


def get_strategy():
    """Pick a tf.distribute strategy that matches the available hardware,
    instead of assuming exactly 2 GPUs are present (e.g. a single-GPU Colab
    runtime, which the original hardcoded devices=["/gpu:0","/gpu:1"] would
    crash on)."""
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) >= 2:
        return tf.distribute.MirroredStrategy()
    if len(gpus) == 1:
        return tf.distribute.OneDeviceStrategy(device='/gpu:0')
    return tf.distribute.get_strategy()


def train_one_round(model, X, Y, X_val, Y_val, round_number, epochs=40,
                     model_name='model', load_weight_file=False,
                     log_prefix='', lr_scheduler=None):
    """Train `model` on (X, Y) for one round, checkpointing on best val_loss.

    :param load_weight_file: if True, load weights saved from round_number - 1 first.
    :return: best validation accuracy reached this round.
    """
    if load_weight_file:
        logging.info("loading weights from previous round...")
        model.load_weights(f'{log_prefix}_{model_name}_round{round_number - 1}.h5')

    checkpoint = ModelCheckpoint(
        f'{log_prefix}_{model_name}_round{round_number}.h5',
        monitor='val_loss', save_best_only=True,
    )
    callbacks = [checkpoint]
    if lr_scheduler is not None:
        callbacks.append(LearningRateScheduler(lr_scheduler))

    history = model.fit(
        X, Y, epochs=epochs, batch_size=BATCH_SIZE,
        validation_data=(X_val, Y_val), callbacks=callbacks, verbose=1,
    )

    pd.to_pickle(history.history, f'{log_prefix}_{model_name}_training_history_round{round_number}.pkl')
    return np.max(history.history['val_acc'])


def train_neural_distinguisher(starting_round, data_generator, model_name, input_size,
                                word_size, log_prefix='./', _epochs=EPOCHS, _num_samples=None):
    """Staged training: train one round at a time, moving to round+1 only while
    validation accuracy stays above ABORT_TRAINING_BELOW_ACC.

    :param data_generator: data_generator(num_samples, round) -> (X, Y)
    :return: (best_round, best_val_acc)
    """
    logging.info(f'CREATE NEURAL NETWORK MODEL {model_name}')
    lr = cyclic_lr(10, 0.001, 0.0002)
    strategy = get_strategy()

    with strategy.scope():
        if model_name == 'model':
            model = model_module.make_model(input_size)
            optimizer = tf.keras.optimizers.Adam(amsgrad=True)
            model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])

    current_round = starting_round
    load_weight_file = False
    best_val_acc = None
    best_round = None
    epochs = _epochs if _epochs is not None else EPOCHS
    num_samples = _num_samples if _num_samples is not None else NUM_SAMPLES

    print(f'Training on {epochs} epochs ...')
    while True:
        logging.info(
            f"CREATE CIPHER DATA for round {current_round} "
            f"(training samples={num_samples:.0e}, validation samples={NUM_VAL_SAMPLES:.0e})..."
        )
        X, Y = data_generator(num_samples, current_round)
        X_val, Y_val = data_generator(NUM_VAL_SAMPLES, current_round)

        logging.info(f"TRAIN neural network for round {current_round}...")
        val_acc = train_one_round(
            model, X, Y, X_val, Y_val, current_round,
            epochs=epochs, load_weight_file=load_weight_file,
            log_prefix=log_prefix, model_name=model_name, lr_scheduler=lr,
        )
        print(f'{model_name}, round {current_round}. Best validation accuracy: {val_acc}', flush=True)

        load_weight_file = True

        if val_acc <= ABORT_TRAINING_BELOW_ACC:
            logging.info(f"ABORT TRAINING (best validation accuracy {val_acc}<={ABORT_TRAINING_BELOW_ACC}).")
            break

        best_round = current_round
        best_val_acc = val_acc
        current_round += 1

        del X, Y, X_val, Y_val
        tf.keras.backend.clear_session()

    return best_round, best_val_acc


def train_neural_distinguishers(output_dir, starting_round, epochs=None, nets=('model',), num_samples=None):
    """Train each net in `nets` starting from `starting_round`; append results
    to `output_dir/results.txt`."""
    plain_bits = 32
    input_size= 2 * 3 * plain_bits
    word_size = 16
    results = {}
    generator = lambda n, nr: pmpg.PolyhedralMultiPairGenerator(
                encryption_function=speck.encrypt_wrapper,
                pos_deltas=POS_DELTAS,
                neg_deltas=NEG_DELTAS,
                plain_bits=plain_bits,
                key_bits=64,
                nr=nr,
                n_samples=n,
                batch_size= max(1, n // 100),
                start_idx=0,
                use_gpu=True
            )[0]
    
    for net in nets:
        print(f'Training {net} starting from round {starting_round}...')
        best_round, best_val_acc = train_neural_distinguisher(
            starting_round=starting_round,
            data_generator=generator,
            model_name=net,
            input_size=input_size,
            word_size=word_size,
            log_prefix=output_dir,
            _epochs=epochs,
            _num_samples=num_samples,
        )
        results[net] = {'Best round': best_round, 'Validation accuracy': best_val_acc}

    results_file_path = os.path.join(output_dir, 'results.txt')
    with open(results_file_path, 'a') as f:
        for net in nets:
            f.write(f'{net} : {results[net]["Best round"]}, {results[net]["Validation accuracy"]}\n')
    print(results)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Obtain good input differences for neural cryptanalysis.')
    parser.add_argument('-o', '--output', type=str, nargs='?', default='results',
                         help='the folder where to store the experiment results')
    args, _unknown = parser.parse_known_args()
    os.makedirs(args.output, exist_ok=True)
    return args.output


def main():
    output_dir = parse_args()
    train_neural_distinguishers(output_dir, 5, nets=['model'])


if __name__ == '__main__':
    main()