import sys
import os

# Đảm bảo Python có thể import được các module trong thư mục src
# Thư mục hiện tại là `configs/`, nên src nằm ở `../src/`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import speck32.train as train

# ==============================================================================
# 1. GHI ĐÈ CÁC HẰNG SỐ CẤU HÌNH CỦA ORCHESTRATOR
# ==============================================================================
train.EPOCHS = 120
train.BATCH_SIZE = 10000
train.POS_DELTAS = [(0x0004, 0x0040), (0x0040, 0x0000), (0x0100, 0x0000)]
train.NEG_DELTAS = [(0x0800, 0x0010), (0x0040, 0x0000), (0x4000, 0x0004)]

# ==============================================================================
# 2. KHỞI CHẠY TRAINING (TỰ ĐỘNG SINH THƯ MỤC LƯU TRỮ)
# ==============================================================================
if __name__ == "__main__":
    # KHÔNG CẦN CHỈ ĐỊNH output_dir!
    # Hệ thống sẽ tự động tạo: results/train/YYYYMMDD_HHMMSS_speck32_6r_raw
    results = train.train_neural_distinguishers(
        starting_round=5,
        feature_mode='raw'
    )

    print(f"✅ Hoàn thành! Best round: {results[0]} - Accuracy: {results[1]}")

