"""Worker: train exactly one round of the neural distinguisher, then exit.

This script is spawned as a **subprocess** by the orchestrator
(``train.py``).  Each invocation gets a completely fresh Python /
TensorFlow process — no stale XLA cache, no VRAM fragmentation, no
background-thread leaks.

Communication
─────────────
  Input :  ``--config-json``  (path to ``train_config.json``)
  Output:  ``round_N_result.json``  +  ``round_N_heartbeat.json``

The worker imports all training logic from ``train.py`` (single source
of truth) — this file is intentionally thin.
"""

# ──────────────────────────────────────────────────────────────────────
# Imports  (TF-related imports are deferred until after env setup)
# ──────────────────────────────────────────────────────────────────────
import os
import sys
import json
import argparse
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import warnings
warnings.filterwarnings("ignore")

# Ensure  src/  is importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
from tensorflow.keras import mixed_precision

# Single source of truth — shared functions from train.py
from speck32.train import (          # noqa: E402
    cyclic_lr,
    create_and_compile_model,
    get_startup_diagnostics,
    get_strategy,
    get_system_info,
    HeartbeatThread,
    HEARTBEAT_INTERVAL_S,
    make_dataset,
    train_one_round,
)


# ──────────────────────────────────────────────────────────────────────
# Worker logger
# ──────────────────────────────────────────────────────────────────────


def _setup_worker_logger(log_dir, round_number):
    """Create a per-round logger with both file and console handlers."""
    logger = logging.getLogger(f'worker_r{round_number}')
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fh = logging.FileHandler(
        os.path.join(log_dir, f'round_{round_number}.log'),
        encoding='utf-8',
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    )
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        logging.Formatter(f'[Worker R{round_number}] %(message)s')
    )
    logger.addHandler(ch)

    return logger


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Train one round (subprocess worker)'
    )
    parser.add_argument('--config-json', required=True,
                        help='Path to train_config.json')
    parser.add_argument('--round', type=int, required=True,
                        help='Round number to train')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for checkpoints and results')
    parser.add_argument('--log-dir', default=None,
                        help='Directory for log files (default: output-dir/logs)')
    parser.add_argument('--load-weights', action='store_true',
                        help='Load weights from previous round')
    args = parser.parse_args()

    # -- directories ---------------------------------------------------
    log_dir = args.log_dir or os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    logger = _setup_worker_logger(log_dir, args.round)
    result_path = os.path.join(
        args.output_dir, f'round_{args.round}_result.json'
    )
    heartbeat_path = os.path.join(
        args.output_dir, f'round_{args.round}_heartbeat.json'
    )

    # -- start heartbeat IMMEDIATELY (revision #1) ---------------------
    heartbeat = HeartbeatThread(heartbeat_path, interval=HEARTBEAT_INTERVAL_S)
    heartbeat.start()

    try:
        # ① Load config
        logger.info(f'Worker started for round {args.round}')
        with open(args.config_json) as f:
            config = json.load(f)
        config['pos_deltas'] = [tuple(d) for d in config['pos_deltas']]
        config['neg_deltas'] = [tuple(d) for d in config['neg_deltas']]
        logger.info(f'Config loaded: {args.config_json}')

        # ② TensorFlow setup + startup diagnostics (revision #4)
        mixed_precision.set_global_policy('mixed_float16')

        logger.info('=' * 50)
        logger.info('  Startup Diagnostics')
        logger.info('=' * 50)
        diag = get_startup_diagnostics()
        for key, val in diag.items():
            logger.info(f'  {key}: {val}')
        logger.info('=' * 50)

        # ③ Create model (shared function)
        strategy = get_strategy()
        xla_active = not isinstance(strategy, tf.distribute.MirroredStrategy)
        logger.info(f'Strategy: {type(strategy).__name__}')
        logger.info(f'XLA (jit_compile): {xla_active}')
        logger.info(f'Mixed precision: {mixed_precision.global_policy()}')

        model = create_and_compile_model(strategy, config['input_size'])
        logger.info(
            f'Model compiled. Parameters: {model.count_params():,}'
        )
        heartbeat.update(current_epoch=0)

        # ④ Create datasets (revision #5: detailed logging)
        logger.info('-' * 40)
        logger.info('  Dataset Initialization')
        logger.info('-' * 40)
        logger.info(f'  Feature mode       : {config["feature_mode"]}')
        logger.info(f'  Input size         : {config["input_size"]}')
        logger.info(f'  Batch size         : {config["batch_size"]}')
        logger.info(f'  Training samples   : {config["num_samples"]:,}')
        logger.info(f'  Validation samples : {config["num_val_samples"]:,}')

        logger.info('Creating training dataset …')
        train_ds, train_steps = make_dataset(
            config, args.round, config['num_samples'], logger
        )
        logger.info(f'Training dataset ready: {train_steps} steps/epoch')

        logger.info('Creating validation dataset …')
        val_ds, val_steps = make_dataset(
            config, args.round, config['num_val_samples'], logger
        )
        logger.info(f'Validation dataset ready: {val_steps} steps/epoch')
        logger.info('-' * 40)

        # ⑤ Train (shared function)
        lr = cyclic_lr(10, 0.001, 0.0002)
        logger.info('Calling train_one_round …')

        val_acc = train_one_round(
            model,
            train_ds,
            val_ds,
            args.round,
            config,
            load_weight_file=args.load_weights,
            output_dir=args.output_dir,
            model_name=config.get('model_name', 'model'),
            lr_scheduler=lr,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            logger=logger,
            heartbeat_thread=heartbeat,
        )

        # ⑥ Success result
        logger.info(f'Training completed. val_acc = {val_acc:.6f}')
        result = {
            'round': args.round,
            'val_acc': val_acc,
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            **get_system_info(),
        }

    except Exception as exc:
        logger.error(f'FATAL: {type(exc).__name__}: {exc}')
        logger.error(traceback.format_exc())
        heartbeat.update(abort_reason=str(exc))
        result = {
            'round': args.round,
            'val_acc': 0.0,
            'status': 'error',
            'error': str(exc),
            'error_type': type(exc).__name__,
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat(),
        }
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        heartbeat.stop()
        logger.info('Worker exiting with error')
        sys.exit(1)

    # -- write success result ------------------------------------------
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    heartbeat.stop()
    logger.info('Worker exiting normally')
    # Process exits → OS reclaims all TF / GPU resources automatically.


if __name__ == '__main__':
    main()
