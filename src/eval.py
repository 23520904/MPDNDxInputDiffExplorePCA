import sys
from pathlib import Path

# Ensure src directory is in sys.path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import os
import argparse
import numpy as np
import tensorflow as tf
import gc

import speck32.cipher as speck
import speck32.model as model_module
import utils.PolytopicQadrupleGenerator as pqg
POS_DELTAS = [(16384, 0), (0, 128), (32, 0)]
NEG_DELTAS = [(32, 0), (0, 1056), (0, 1026)]
def evaluate_models(model_dir='train_output', start_round=1, end_round=13, feature_mode='full', num_samples=10**6,pos_deltas=POS_DELTAS, neg_deltas=NEG_DELTAS):
    plain_bits = 32
    if feature_mode == 'raw':
        input_size = 4 * plain_bits
    elif feature_mode == 'diff':
        input_size = 3 * plain_bits
    else:  # 'full'
        input_size = 7 * plain_bits

    

    for round_number in range(start_round, end_round + 1):
        model_path = os.path.join(model_dir, f'model_round{round_number}.h5')
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}. Skipping round {round_number}.")
            continue

        print(f"\n--- Evaluating Round {round_number} ---")
        try:
            # Recreate the model architecture manually to avoid Keras TrueDivide deserialization bug
            model = model_module.make_model(input_size)
            model.compile(optimizer='adam', loss='mse', metrics=['acc'])
            model.load_weights(model_path)
            
            # Generate test data specifically for this round
            print(f"Generating {num_samples} test samples...")
            generator = pqg.PolytopicQuadrupleGenerator(
                encryption_function=speck.encrypt_wrapper,
                pos_diffs=pos_deltas,
                neg_diffs=neg_deltas,
                plain_bits=plain_bits,
                key_bits=64,
                nr=round_number,
                n_samples=num_samples,
                batch_size=max(1, num_samples // 100),
                feature_mode=feature_mode,
                use_gpu=True,
                to_float32=True
            )
            
            X_test, Y_test = generator[0]
            print("Test data generated successfully.")

            # Verify the shape of the test data
            print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
            
            print("Evaluating model...")
            results = model.evaluate(X_test, Y_test, verbose=0)
            
            # Predict to get confusion matrix metrics
            Y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
            
            tp = np.sum((Y_test == 1) & (Y_pred == 1))
            tn = np.sum((Y_test == 0) & (Y_pred == 0))
            fp = np.sum((Y_test == 0) & (Y_pred == 1))
            fn = np.sum((Y_test == 1) & (Y_pred == 0))
            
            print(f"Test Loss: {results[0]:.4f}")
            print(f"Test Accuracy: {results[1]:.4f}")
            print(f"True Positives: {tp}")
            print(f"True Negatives: {tn}")
            print(f"False Positives: {fp}")
            print(f"False Negatives: {fn}")
            
            del X_test, Y_test, Y_pred
            del model
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            print(f"Error evaluating round {round_number}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained neural distinguishers.')
    parser.add_argument('-d', '--model_dir', type=str, default='train_output', help='Directory containing the .h5 models')
    parser.add_argument('-s', '--start_round', type=int, default=1, help='Start round number')
    parser.add_argument('-e', '--end_round', type=int, default=13, help='End round number')
    parser.add_argument('-n', '--num_samples', type=int, default=10**6, help='Number of test samples')
    parser.add_argument('-f', '--feature_mode', type=str, choices=['full', 'diff', 'raw'], default='full')
    args = parser.parse_args()
    
    evaluate_models(args.model_dir, args.start_round, args.end_round, args.feature_mode, args.num_samples)
