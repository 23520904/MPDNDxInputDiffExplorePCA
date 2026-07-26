import speck32.train as train
train.POS_DELTAS = [(0x0004, 0x0040), (0x0040, 0x0000), (0x0100, 0x0000)]
train.NEG_DELTAS = [(0x0800, 0x0010), (0x0040, 0x0000), (0x4000, 0x0004)]

train.ABORT_TRAINING_BELOW_ACC = 0.505
train.EPOCHS = 120
train.NUM_SAMPLES = 10 ** 7
train.NUM_VAL_SAMPLES = 10 ** 5
train.BATCH_SIZE = 10000  # Reasonable batch size for GPU
print("ok")



# 1. Set the parameters for your run
START_ROUND = 5
OUTPUT_PATH = 'train_speck3264_6round_raw/id_340_726'  # Folder where checkpoints are saved

# 2. Start the staged training
# If it crashes at round 7, just change START_ROUND to 7 and run this cell again.
results = train.train_neural_distinguishers(
    output_dir=OUTPUT_PATH,
    starting_round=START_ROUND,
    feature_mode='raw',
    
)

print(f"Training finished. Best round: {results[0]} with Accuracy: {results[1]}")
