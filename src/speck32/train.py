"""Staged training loop for the differential-neural distinguisher.

Architecture: Subprocess Isolation
──────────────────────────────────
Each training round runs in a **fresh Python process** via ``train_round.py``.
Process exit handles all resource cleanup — no ``clear_session()``,
no ``gc.collect()``, no ``del model``.

This file contains:
  1. **Shared functions** (imported by the worker ``train_round.py``):
     model creation, dataset creation, training loop, callbacks, logging.
  2. **Orchestrator** (``train_neural_distinguisher``):
     spawns one subprocess per round, monitors health, handles retries.
  3. **Public API** (``train_neural_distinguishers``):
     backward-compatible entry point called by ``config.py``.
"""

# ──────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import json
import logging
import math
import signal
import subprocess
import sys
import threading
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

import speck32.cipher as speck
import speck32.model as model_module
import data_utils.PolytopicQadrupleGenerator as pqg

# ──────────────────────────────────────────────────────────────────────
# Module-level constants  (overridable by config.py via  train.X = ...)
# ──────────────────────────────────────────────────────────────────────
ABORT_TRAINING_BELOW_ACC = 0.505
EPOCHS = 120
NUM_SAMPLES = 10 ** 7
NUM_VAL_SAMPLES = 10 ** 5
BATCH_SIZE = 10_000
POS_DELTAS = [(16384, 0), (0, 128), (32, 0)]
NEG_DELTAS = [(32, 0), (0, 1056), (0, 1026)]

# Orchestrator settings
MAX_RETRIES = 3
HEARTBEAT_INTERVAL_S = 30          # background heartbeat every 30 s
HEARTBEAT_STALE_S = 300            # 5 min before considering stale
TIMEOUT_SAFETY_FACTOR = 3.0       # adaptive timeout multiplier
RETRY_BACKOFF_S = 10
PROGRESS_REPORT_INTERVAL_S = 60   # orchestrator status print interval


# ======================================================================
#  SHARED FUNCTIONS  (imported by train_round.py — single source of truth)
# ======================================================================


def cyclic_lr(num_epochs, high_lr, low_lr):
    """Cyclic learning-rate schedule."""
    return lambda i: (
        low_lr
        + ((num_epochs - 1) - i % num_epochs)
        / (num_epochs - 1)
        * (high_lr - low_lr)
    )


def get_strategy():
    """Auto-detect GPU distribution strategy."""
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 1:
        return tf.distribute.MirroredStrategy()
    if gpus:
        return tf.distribute.OneDeviceStrategy(device='/gpu:0')
    return tf.distribute.get_strategy()


def get_system_info():
    """Collect CPU / GPU resource usage.  Graceful fallback when optional
    dependencies (``psutil``) are missing."""
    info = {}
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['cpu_ram_used_gb'] = round(mem.used / 1e9, 2)
        info['cpu_ram_total_gb'] = round(mem.total / 1e9, 2)
        info['cpu_ram_percent'] = mem.percent
    except ImportError:
        pass
    try:
        gpu_mem = tf.config.experimental.get_memory_info('GPU:0')
        info['gpu_vram_current_mb'] = round(gpu_mem['current'] / 1e6, 1)
        info['gpu_vram_peak_mb'] = round(gpu_mem['peak'] / 1e6, 1)
    except Exception:
        pass
    return info


def get_startup_diagnostics():
    """Comprehensive environment snapshot (logged once at worker start)."""
    diag = {
        'tensorflow_version': tf.__version__,
        'python_version': sys.version.split()[0],
        'gpus': [g.name for g in tf.config.list_physical_devices('GPU')],
        'mixed_precision_policy': str(mixed_precision.global_policy()),
    }
    try:
        build = tf.sysconfig.get_build_info()
        diag['cuda_version'] = build.get('cuda_version', 'n/a')
        diag['cudnn_version'] = build.get('cudnn_version', 'n/a')
    except Exception:
        pass
    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            details = tf.config.experimental.get_device_details(gpu)
            diag['gpu_model'] = details.get('device_name', 'n/a')
            break  # first GPU
    except Exception:
        pass
    diag.update(get_system_info())
    return diag


def make_dataset(config, round_number, n_samples, logger=None):
    """Create a streaming ``tf.data.Dataset`` for one round.

    Extracted from the former nested ``generator()`` closure so that
    both orchestrator tests and the worker can call it.
    """
    if logger:
        logger.info(
            f'Initializing generator: n_samples={n_samples:,}, '
            f'batch_size={config["batch_size"]}, '
            f'feature_mode={config["feature_mode"]}, '
            f'input_size={config["input_size"]}'
        )

    gen = pqg.PolytopicQuadrupleGenerator(
        encryption_function=speck.encrypt_wrapper,
        pos_diffs=config['pos_deltas'],
        neg_diffs=config['neg_deltas'],
        plain_bits=32,
        key_bits=64,
        nr=round_number,
        n_samples=n_samples,
        batch_size=config['batch_size'],
        feature_mode=config['feature_mode'],
        use_gpu=False,
        encrypt_backend='numpy',
        to_float32=True,
    )

    if logger:
        logger.info(f'Generator initialized: {len(gen)} steps')

    input_size = config['input_size']

    def gen_func():
        for i in range(len(gen)):
            X, Y = gen[i]
            yield X, Y.astype(np.float32)

    dataset = tf.data.Dataset.from_generator(
        gen_func,
        output_signature=(
            tf.TensorSpec(shape=(None, input_size), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    )

    if logger:
        logger.info(
            f'Dataset created successfully: {n_samples:,} samples, '
            f'{len(gen)} steps/epoch'
        )

    return dataset.repeat().prefetch(tf.data.AUTOTUNE), len(gen)


def create_and_compile_model(strategy, input_size):
    """Create and compile the distinguisher model within *strategy* scope."""
    with strategy.scope():
        model = model_module.make_model(input_size)
        use_xla = not isinstance(strategy, tf.distribute.MirroredStrategy)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(amsgrad=True),
            loss='mse',
            metrics=['acc'],
            jit_compile=use_xla,
        )
    return model


# ──────────────────────────────────────────────────────────────────────
#  HeartbeatThread  (revision #1 — independent of epoch completion)
# ──────────────────────────────────────────────────────────────────────


class HeartbeatThread(threading.Thread):
    """Background daemon thread that writes a heartbeat JSON file every
    *interval* seconds, **independently** of epoch completion.

    The ``TrainingHealthCallback`` calls ``update()`` with fresh metrics
    whenever an epoch finishes, but the heartbeat keeps ticking regardless.
    """

    def __init__(self, heartbeat_path, interval=HEARTBEAT_INTERVAL_S):
        super().__init__(daemon=True)
        self.heartbeat_path = heartbeat_path
        self.interval = interval
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.start_time = time.time()
        self._data = {
            'worker_alive': True,
            'current_epoch': 0,
            'latest_train_loss': None,
            'latest_val_loss': None,
            'latest_train_acc': None,
            'latest_val_acc': None,
            'best_val_acc': None,
            'steps_per_sec': None,
            'samples_per_sec': None,
            'avg_batch_time_s': None,
            'avg_epoch_time_s': None,
            'abort_reason': None,
        }

    # -- public API (called from any thread) ---------------------------

    def update(self, **kwargs):
        """Merge *kwargs* into the heartbeat payload."""
        with self._lock:
            self._data.update(kwargs)

    def stop(self):
        """Signal the thread to stop and wait for it to finish."""
        self.update(worker_alive=False)
        self._stop_event.set()
        self.join(timeout=5)

    # -- thread body ---------------------------------------------------

    def run(self):
        while not self._stop_event.wait(self.interval):
            self._write()
        self._write()  # final heartbeat on exit

    def _write(self):
        with self._lock:
            data = dict(self._data)
        data['elapsed_s'] = round(time.time() - self.start_time, 1)
        data['timestamp'] = datetime.now().isoformat()
        data.update(get_system_info())
        try:
            tmp = self.heartbeat_path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.heartbeat_path)  # atomic on POSIX + Win
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────
#  TrainingHealthCallback
# ──────────────────────────────────────────────────────────────────────


class TrainingHealthCallback(tf.keras.callbacks.Callback):
    """Keras callback that monitors training health from *inside* the
    worker process.

    Responsibilities
    ~~~~~~~~~~~~~~~~
    * Detect NaN / exploding loss → ``model.stop_training = True``.
    * Verify checkpoint integrity after each save  (revision #7).
    * Compute throughput metrics: steps/s, samples/s  (revision #3).
    * Push epoch metrics into the ``HeartbeatThread``.
    * Log epoch progress to the per-round log file.
    """

    def __init__(
        self,
        round_number,
        output_dir,
        heartbeat_thread=None,
        logger=None,
        model_name='model',
        loss_explosion_factor=100.0,
    ):
        super().__init__()
        self.round_number = round_number
        self.output_dir = output_dir
        self.heartbeat_thread = heartbeat_thread
        self.logger = logger
        self.model_name = model_name
        self.loss_explosion_factor = loss_explosion_factor

        self.train_start_time = None
        self.epoch_start_time = None
        self.initial_loss = None
        self.abort_reason = None
        self.best_val_acc = 0.0
        self.epoch_times = []
        self.batch_times = []
        self._batch_start = None

    # -- lifecycle hooks -----------------------------------------------

    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()
        if self.logger:
            self.logger.info('Training started')

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.batch_times = []

    def on_train_batch_begin(self, batch, logs=None):
        self._batch_start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if self._batch_start is not None:
            self.batch_times.append(time.time() - self._batch_start)

    def on_epoch_end(self, epoch, logs=None):
        epoch_dur = time.time() - self.epoch_start_time
        total_elapsed = time.time() - self.train_start_time
        self.epoch_times.append(epoch_dur)

        loss = logs.get('loss', 0.0)
        val_loss = logs.get('val_loss', 0.0)
        train_acc = logs.get('acc', 0.0)
        val_acc = logs.get('val_acc', 0.0)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

        # -- throughput (revision #3) ----------------------------------
        avg_batch = float(np.mean(self.batch_times)) if self.batch_times else 0.0
        steps_in_epoch = len(self.batch_times)
        steps_per_sec = steps_in_epoch / epoch_dur if epoch_dur > 0 else 0.0
        samples_per_sec = steps_per_sec * BATCH_SIZE

        # -- NaN detection ---------------------------------------------
        if math.isnan(loss) or math.isnan(val_loss):
            self.abort_reason = 'NaN loss detected'
            self.model.stop_training = True

        # -- loss explosion --------------------------------------------
        if self.initial_loss is None and not math.isnan(loss):
            self.initial_loss = loss
        elif (
            self.initial_loss
            and self.initial_loss > 0
            and loss > self.initial_loss * self.loss_explosion_factor
        ):
            self.abort_reason = (
                f'Loss exploded: {loss:.4f} > '
                f'{self.initial_loss:.4f} × {self.loss_explosion_factor}'
            )
            self.model.stop_training = True

        # -- checkpoint integrity (revision #7) ------------------------
        ckpt_path = os.path.join(
            self.output_dir,
            f'{self.model_name}_round{self.round_number}.h5',
        )
        if os.path.exists(ckpt_path) and not _verify_checkpoint(
            ckpt_path, self.logger
        ):
            self.abort_reason = f'Checkpoint integrity failed: {ckpt_path}'
            self.model.stop_training = True

        # -- push to heartbeat thread ----------------------------------
        if self.heartbeat_thread:
            self.heartbeat_thread.update(
                current_epoch=epoch + 1,
                latest_train_loss=(
                    float(loss) if not math.isnan(loss) else 'NaN'
                ),
                latest_val_loss=(
                    float(val_loss) if not math.isnan(val_loss) else 'NaN'
                ),
                latest_train_acc=round(float(train_acc), 6),
                latest_val_acc=round(float(val_acc), 6),
                best_val_acc=round(float(self.best_val_acc), 6),
                steps_per_sec=round(steps_per_sec, 2),
                samples_per_sec=round(samples_per_sec, 1),
                avg_batch_time_s=round(avg_batch, 4),
                avg_epoch_time_s=round(float(np.mean(self.epoch_times)), 1),
                abort_reason=self.abort_reason,
            )

        # -- log to file -----------------------------------------------
        if self.logger:
            total_epochs = (
                self.params.get('epochs', '?') if self.params else '?'
            )
            remaining = (
                (total_epochs - (epoch + 1))
                if isinstance(total_epochs, int)
                else 0
            )
            eta = remaining * float(np.mean(self.epoch_times))
            self.logger.info(
                f'Epoch {epoch + 1}/{total_epochs} | '
                f'loss={loss:.4f} val_loss={val_loss:.4f} | '
                f'acc={train_acc:.4f} val_acc={val_acc:.4f} | '
                f'best={self.best_val_acc:.4f} | '
                f'{epoch_dur:.1f}s/ep | '
                f'{steps_per_sec:.1f} steps/s '
                f'({samples_per_sec:,.0f} samp/s) | '
                f'ETA {_fmt_duration(eta)}'
            )
            if self.abort_reason:
                self.logger.error(f'ABORT: {self.abort_reason}')

    def on_train_end(self, logs=None):
        total = time.time() - self.train_start_time if self.train_start_time else 0
        if self.logger:
            self.logger.info(
                f'Training ended. Total: {_fmt_duration(total)}, '
                f'Best val_acc: {self.best_val_acc:.6f}'
            )


# ──────────────────────────────────────────────────────────────────────
#  train_one_round  (shared — called by the worker)
# ──────────────────────────────────────────────────────────────────────


def train_one_round(
    model,
    train_ds,
    val_ds,
    round_number,
    config,
    load_weight_file=False,
    output_dir='./',
    model_name='model',
    lr_scheduler=None,
    steps_per_epoch=None,
    validation_steps=None,
    logger=None,
    heartbeat_thread=None,
):
    """Train *model* for one round.  Returns the best validation accuracy."""
    # -- load previous weights -----------------------------------------
    if load_weight_file and round_number > 1:
        prev = os.path.join(output_dir, f'{model_name}_round{round_number - 1}.h5')
        if os.path.exists(prev):
            model.load_weights(prev)
            if logger:
                logger.info(f'Weights loaded: {os.path.abspath(prev)}')
        else:
            if logger:
                logger.warning(
                    'No previous checkpoint found. Training from scratch.'
                )

    # -- callbacks -----------------------------------------------------
    ckpt_path = os.path.join(
        output_dir, f'{model_name}_round{round_number}.h5'
    )
    checkpoint = ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True)
    callbacks = [checkpoint]
    if lr_scheduler is not None:
        callbacks.append(LearningRateScheduler(lr_scheduler))

    health_cb = TrainingHealthCallback(
        round_number=round_number,
        output_dir=output_dir,
        heartbeat_thread=heartbeat_thread,
        logger=logger,
        model_name=model_name,
    )
    callbacks.append(health_cb)

    if logger:
        logger.info(
            f'model.fit: epochs={config["epochs"]}, '
            f'steps_per_epoch={steps_per_epoch}, '
            f'validation_steps={validation_steps}'
        )

    # -- train ---------------------------------------------------------
    history = model.fit(
        train_ds,
        epochs=config['epochs'],
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

    pd.to_pickle(
        history.history,
        os.path.join(
            output_dir,
            f'{model_name}_training_history_round{round_number}.pkl',
        ),
    )

    best_val_acc = float(np.max(history.history['val_acc']))

    # -- final checkpoint integrity (revision #7) ----------------------
    if os.path.exists(ckpt_path):
        if not _verify_checkpoint(ckpt_path, logger):
            raise RuntimeError(
                f'Final checkpoint integrity check failed: {ckpt_path}'
            )
        if logger:
            size_mb = os.path.getsize(ckpt_path) / 1e6
            logger.info(f'Checkpoint verified: {ckpt_path} ({size_mb:.1f} MB)')

    # -- propagate abort from health callback --------------------------
    if health_cb.abort_reason:
        raise RuntimeError(f'Training aborted: {health_cb.abort_reason}')

    return best_val_acc


# ======================================================================
#  HELPER FUNCTIONS
# ======================================================================


def _verify_checkpoint(path, logger=None):
    """Return True if checkpoint file exists, is non-empty, and readable."""
    try:
        if not os.path.exists(path):
            return False
        size = os.path.getsize(path)
        if size == 0:
            if logger:
                logger.error(f'Checkpoint empty: {path}')
            return False
        with open(path, 'rb') as f:
            header = f.read(8)
            if len(header) < 8:
                if logger:
                    logger.error(f'Checkpoint too small ({size} bytes): {path}')
                return False
        return True
    except Exception as exc:
        if logger:
            logger.error(f'Checkpoint integrity error ({path}): {exc}')
        return False


def _fmt_duration(seconds):
    """Human-readable duration string."""
    seconds = max(0, float(seconds))
    if seconds < 60:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        return f'{seconds / 60:.1f}m'
    return f'{seconds / 3600:.1f}h'


def _resolve_feature_mode(input_size):
    """Reverse-map *input_size* → feature mode string."""
    plain_bits = 32
    return {
        4 * plain_bits: 'raw',
        3 * plain_bits: 'diff',
        7 * plain_bits: 'full',
    }.get(input_size, 'full')


def _setup_logger(name, log_file, level=logging.DEBUG):
    """Create a logger with both file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    )
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
    logger.addHandler(ch)
    return logger


# ======================================================================
#  ORCHESTRATOR  (subprocess management)
# ======================================================================


def _spawn_and_monitor_worker(
    round_number,
    config_path,
    output_dir,
    logs_dir,
    load_weights,
    config,
    logger,
    adaptive_epoch_time=None,
):
    """Spawn one worker subprocess, stream its output, and monitor its
    heartbeat.  Returns ``val_acc`` on success or ``None`` on failure."""

    worker_script = str(Path(__file__).parent / 'train_round.py')
    cmd = [
        sys.executable, worker_script,
        '--config-json', config_path,
        '--round', str(round_number),
        '--output-dir', output_dir,
        '--log-dir', logs_dir,
    ]
    if load_weights:
        cmd.append('--load-weights')

    round_log = os.path.join(logs_dir, f'round_{round_number}.log')
    heartbeat_path = os.path.join(
        output_dir, f'round_{round_number}_heartbeat.json'
    )
    result_path = os.path.join(
        output_dir, f'round_{round_number}_result.json'
    )
    stop_file = os.path.join(output_dir, 'STOP')

    # Clean stale artefacts from previous attempts
    for p in (heartbeat_path, result_path):
        if os.path.exists(p):
            os.remove(p)

    # -- adaptive timeout (revision #9) --------------------------------
    if adaptive_epoch_time and adaptive_epoch_time > 0:
        timeout = (
            config['epochs']
            * adaptive_epoch_time
            * config.get('timeout_safety_factor', TIMEOUT_SAFETY_FACTOR)
        )
    else:
        timeout = 7200  # conservative default: 2 hours

    logger.info(f'Worker command: {" ".join(cmd)}')
    logger.info(f'Worker log: {round_log}')
    logger.info(f'Timeout: {_fmt_duration(timeout)}')

    # -- launch (revision #6: Popen) -----------------------------------
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).parent.parent),
        text=True,
        bufsize=1,  # line-buffered
    )
    logger.info(f'Worker PID: {proc.pid}')

    start_time = time.time()
    last_heartbeat_mtime = start_time
    last_stdout_time = [start_time]  # mutable for thread access
    last_progress_time = start_time

    # -- stream stdout in a background thread --------------------------
    log_fh = open(round_log, 'a', encoding='utf-8')

    def _stream():
        try:
            for line in proc.stdout:
                if line:
                    print(line, end='', flush=True)
                    log_fh.write(line)
                    log_fh.flush()
                    last_stdout_time[0] = time.time()
        except ValueError:
            pass  # stdout closed

    output_thread = threading.Thread(target=_stream, daemon=True)
    output_thread.start()

    # -- main monitoring loop ------------------------------------------
    terminate_reason = None
    try:
        while proc.poll() is None:
            time.sleep(5)
            elapsed = time.time() - start_time

            # ① STOP file (revision #8)
            if os.path.exists(stop_file):
                terminate_reason = 'STOP file detected'
                break

            # ② heartbeat checks
            if os.path.exists(heartbeat_path):
                cur_mtime = os.path.getmtime(heartbeat_path)
                if cur_mtime > last_heartbeat_mtime:
                    last_heartbeat_mtime = cur_mtime

                try:
                    with open(heartbeat_path) as hf:
                        hb = json.load(hf)
                    if hb.get('abort_reason'):
                        terminate_reason = (
                            f'Worker self-abort: {hb["abort_reason"]}'
                        )
                        break
                except (json.JSONDecodeError, IOError):
                    pass  # file being written

            # ③ stale heartbeat — multi-indicator (revision #2)
            stale = time.time() - last_heartbeat_mtime
            if stale > config.get('heartbeat_stale_s', HEARTBEAT_STALE_S):
                stdout_active = (
                    time.time() - last_stdout_time[0]
                ) < config.get('heartbeat_stale_s', HEARTBEAT_STALE_S)
                if stdout_active:
                    logger.warning(
                        f'Heartbeat stale ({stale:.0f}s) but stdout active. '
                        f'Continuing.'
                    )
                else:
                    terminate_reason = (
                        f'Worker unresponsive: heartbeat stale {stale:.0f}s, '
                        f'no stdout for '
                        f'{time.time() - last_stdout_time[0]:.0f}s'
                    )
                    break

            # ④ timeout (revision #9 adaptive)
            if elapsed > timeout:
                terminate_reason = (
                    f'Timeout ({_fmt_duration(elapsed)} > '
                    f'{_fmt_duration(timeout)})'
                )
                break

            # ⑤ progress report (revision #6)
            if time.time() - last_progress_time > PROGRESS_REPORT_INTERVAL_S:
                last_progress_time = time.time()
                _print_orchestrator_progress(
                    round_number, proc.pid, elapsed, timeout,
                    heartbeat_path, config, logger,
                )

    except KeyboardInterrupt:
        terminate_reason = 'KeyboardInterrupt'

    # -- terminate if needed -------------------------------------------
    if terminate_reason:
        logger.error(f'Terminating worker: {terminate_reason}')
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    # -- wait for output thread to drain -------------------------------
    output_thread.join(timeout=10)
    log_fh.close()

    # -- propagate KeyboardInterrupt -----------------------------------
    if terminate_reason == 'KeyboardInterrupt':
        raise KeyboardInterrupt

    # -- evaluate result -----------------------------------------------
    if proc.returncode != 0:
        logger.error(f'Worker exited with code {proc.returncode}')
        return None

    if not os.path.exists(result_path):
        logger.error(f'No result file: {result_path}')
        return None

    try:
        with open(result_path) as f:
            result = json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        logger.error(f'Cannot read result file: {exc}')
        return None

    if result.get('status') == 'error':
        logger.error(f'Worker error: {result.get("error", "unknown")}')
        return None

    return result['val_acc']


def _print_orchestrator_progress(
    round_number, pid, elapsed, timeout, heartbeat_path, config, logger
):
    """Print a one-line progress summary to the terminal / log."""
    hb = None
    if os.path.exists(heartbeat_path):
        try:
            with open(heartbeat_path) as f:
                hb = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    parts = [
        f'Round {round_number}',
        f'PID {pid}',
        f'Elapsed {_fmt_duration(elapsed)}',
    ]

    if hb:
        epoch = hb.get('current_epoch', '?')
        total_epochs = config.get('epochs', '?')
        val_acc = hb.get('latest_val_acc', '?')
        parts.append(f'Epoch {epoch}/{total_epochs}')
        if isinstance(val_acc, (int, float)):
            parts.append(f'val_acc={val_acc:.4f}')
        avg_ep = hb.get('avg_epoch_time_s')
        if avg_ep and isinstance(epoch, int) and isinstance(total_epochs, int):
            remaining = (total_epochs - epoch) * avg_ep
            parts.append(f'ETA {_fmt_duration(remaining)}')
        sps = hb.get('steps_per_sec')
        if sps:
            parts.append(f'{sps:.1f} steps/s')
    else:
        parts.append('Heartbeat: waiting…')

    parts.append(f'Timeout in {_fmt_duration(max(0, timeout - elapsed))}')
    logger.info('[Progress] ' + ' | '.join(parts))


def _run_round_with_retry(
    round_number,
    config_path,
    output_dir,
    logs_dir,
    load_weights,
    config,
    logger,
    adaptive_epoch_time=None,
):
    """Run one round with automatic retry on failure."""
    max_retries = config.get('max_retries', MAX_RETRIES)

    for attempt in range(1, max_retries + 1):
        logger.info(
            f'Round {round_number}, attempt {attempt}/{max_retries}'
        )
        print(
            f'[Orchestrator] Round {round_number} — '
            f'attempt {attempt}/{max_retries}'
        )

        val_acc = _spawn_and_monitor_worker(
            round_number, config_path, output_dir, logs_dir,
            load_weights, config, logger, adaptive_epoch_time,
        )

        if val_acc is not None:
            return True, val_acc

        if attempt < max_retries:
            logger.warning(f'Retrying in {RETRY_BACKOFF_S}s …')
            print(
                f'[Orchestrator] Failed (attempt {attempt}/{max_retries}). '
                f'Retrying in {RETRY_BACKOFF_S}s …'
            )
            time.sleep(RETRY_BACKOFF_S)

    return False, 0.0


# ======================================================================
#  train_neural_distinguisher  (orchestrator entry)
# ======================================================================


def train_neural_distinguisher(
    starting_round,
    model_name,
    input_size,
    log_prefix='./',
    _epochs=None,
    _num_samples=None,
    feature_mode='full',
):
    """Staged training via subprocess isolation.

    Each round executes in a fresh Python process.  The orchestrator
    serialises config → spawns worker → monitors heartbeat → reads result
    → decides continue / retry / abort.
    """
    epochs = _epochs if _epochs is not None else EPOCHS
    num_samples = _num_samples if _num_samples is not None else NUM_SAMPLES

    # -- serialise runtime config (includes overridden module-level vars) --
    config = {
        'epochs': epochs,
        'num_samples': num_samples,
        'num_val_samples': NUM_VAL_SAMPLES,
        'batch_size': BATCH_SIZE,
        'input_size': input_size,
        'feature_mode': feature_mode,
        'pos_deltas': [list(d) for d in POS_DELTAS],
        'neg_deltas': [list(d) for d in NEG_DELTAS],
        'abort_below_acc': ABORT_TRAINING_BELOW_ACC,
        'model_name': model_name,
        'max_retries': MAX_RETRIES,
        'heartbeat_stale_s': HEARTBEAT_STALE_S,
        'timeout_safety_factor': TIMEOUT_SAFETY_FACTOR,
    }

    os.makedirs(log_prefix, exist_ok=True)
    logs_dir = os.path.join(log_prefix, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    config_path = os.path.join(log_prefix, 'train_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    orch_logger = _setup_logger(
        'orchestrator', os.path.join(logs_dir, 'orchestrator.log')
    )
    orch_logger.info(f'Config: {config_path}')

    current_round = starting_round
    load_weights = os.path.exists(
        os.path.join(
            log_prefix, f'{model_name}_round{current_round - 1}.h5'
        )
    )
    best_round, best_val_acc = starting_round, 0.0
    adaptive_epoch_time = None

    # -- signal handling -----------------------------------------------
    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint(signum, frame):
        orch_logger.warning('Ctrl+C — terminating …')
        print('\n[Orchestrator] Ctrl+C received. Cleaning up …')
        signal.signal(signal.SIGINT, original_sigint)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint)

    try:
        while True:
            # STOP file (revision #8)
            stop_file = os.path.join(log_prefix, 'STOP')
            if os.path.exists(stop_file):
                orch_logger.warning('STOP file detected. Exiting.')
                print('[Orchestrator] STOP file detected. Exiting.')
                break

            orch_logger.info(f'=== Round {current_round} ===')
            print(f'\n{"=" * 60}')
            print(f'  Spawning subprocess for Round {current_round}')
            if adaptive_epoch_time:
                est = config['epochs'] * adaptive_epoch_time
                print(f'  Estimated round time: {_fmt_duration(est)}')
            print(f'{"=" * 60}')

            success, val_acc = _run_round_with_retry(
                current_round, config_path, log_prefix, logs_dir,
                load_weights, config, orch_logger, adaptive_epoch_time,
            )

            if not success:
                orch_logger.error(
                    f'Round {current_round} failed after all retries.'
                )
                print(f'[Orchestrator] Round {current_round} failed.')
                break

            # update adaptive timeout from heartbeat (revision #9)
            hb_path = os.path.join(
                log_prefix, f'round_{current_round}_heartbeat.json'
            )
            if os.path.exists(hb_path):
                try:
                    with open(hb_path) as f:
                        hb = json.load(f)
                    aet = hb.get('avg_epoch_time_s')
                    if aet and isinstance(aet, (int, float)) and aet > 0:
                        adaptive_epoch_time = aet
                        orch_logger.info(
                            f'Adaptive epoch time: {adaptive_epoch_time:.1f}s'
                        )
                except Exception:
                    pass

            orch_logger.info(
                f'Round {current_round}: val_acc = {val_acc:.6f}'
            )
            print(
                f'[Orchestrator] Round {current_round}: '
                f'val_acc = {val_acc:.6f}'
            )

            if val_acc <= ABORT_TRAINING_BELOW_ACC:
                orch_logger.info(
                    f'val_acc {val_acc:.6f} ≤ {ABORT_TRAINING_BELOW_ACC}. '
                    f'Stopping.'
                )
                print('[Orchestrator] Accuracy below threshold. Stopping.')
                break

            best_round, best_val_acc = current_round, val_acc
            current_round += 1
            load_weights = True

    except KeyboardInterrupt:
        orch_logger.info('Training interrupted by user.')
        print('[Orchestrator] Training interrupted.')

    finally:
        signal.signal(signal.SIGINT, original_sigint)

    orch_logger.info(
        f'Done. Best: round {best_round}, val_acc {best_val_acc:.6f}'
    )
    print(
        f'\n[Orchestrator] Done. '
        f'Best: round {best_round}, val_acc {best_val_acc:.6f}'
    )
    return best_round, best_val_acc


# ======================================================================
#  PUBLIC API  (backward-compatible — called by config.py)
# ======================================================================


def train_neural_distinguishers(
    output_dir='results',
    starting_round=1,
    epochs=None,
    num_samples=None,
    feature_mode='full',
):
    """Train neural distinguishers with staged subprocess isolation.

    This is the public entry point.  Existing ``config.py`` files that
    override module-level constants (``POS_DELTAS``, ``EPOCHS``, etc.)
    and call this function continue to work without modification.
    """
    os.makedirs(output_dir, exist_ok=True)
    plain_bits = 32
    input_size = {
        'raw': 4 * plain_bits,
        'diff': 3 * plain_bits,
        'full': 7 * plain_bits,
    }[feature_mode]

    print('=' * 60)
    print('  Train Configuration (Subprocess Isolation)')
    print('=' * 60)
    print(f'  Output directory   : {os.path.abspath(output_dir)}')
    print(f'  Starting round     : {starting_round}')
    print(f'  Epochs per round   : {epochs if epochs is not None else EPOCHS}')
    print(f'  Training samples   : {(num_samples if num_samples is not None else NUM_SAMPLES):,}')
    print(f'  Validation samples : {NUM_VAL_SAMPLES:,}')
    print(f'  Batch size         : {BATCH_SIZE:,}')
    print(f'  Feature mode       : {feature_mode}')
    print(f'  Input size         : {input_size}')
    print(f'  Pos deltas         : {POS_DELTAS}')
    print(f'  Neg deltas         : {NEG_DELTAS}')
    print(f'  Abort below acc    : {ABORT_TRAINING_BELOW_ACC}')
    print(f'  Max retries        : {MAX_RETRIES}')
    print(f'  Heartbeat interval : {HEARTBEAT_INTERVAL_S}s')
    print('=' * 60)

    return train_neural_distinguisher(
        starting_round,
        'model',
        input_size,
        output_dir,
        epochs,
        num_samples,
        feature_mode,
    )