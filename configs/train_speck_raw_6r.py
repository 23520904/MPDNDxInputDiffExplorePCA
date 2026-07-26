import sys
import os
from datetime import datetime

# Đảm bảo Python có thể import được các module trong thư mục src
# Thư mục hiện tại là `configs/`, nên src nằm ở `../src/`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import speck32.train as train

# ==============================================================================
# 1. HƯỚNG DẪN ĐẶT TÊN THAM SỐ output_dir (EXPERIMENT TRACKING)
# ==============================================================================
# Nên đặt tên theo format: [Ngày tháng]_[Loại model]_[Số round]_[Chế độ feature]
# Ví dụ: 20260726_143000_speck32_6r_raw
# 
# Cách làm tốt nhất là dùng thư viện datetime để tự động sinh timestamp.
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
feature_mode = 'raw'
target_round = 6

experiment_name = f"{timestamp}_speck32_{target_round}r_{feature_mode}"

# Output dir sẽ trỏ ra thư mục `results/` ngoài root
OUTPUT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'results', experiment_name
))

# ==============================================================================
# 2. GHI ĐÈ CÁC HẰNG SỐ CẤU HÌNH CỦA ORCHESTRATOR
# ==============================================================================
train.EPOCHS = 120
train.BATCH_SIZE = 10000
train.POS_DELTAS = [(0x0004, 0x0040), (0x0040, 0x0000), (0x0100, 0x0000)]
train.NEG_DELTAS = [(0x0800, 0x0010), (0x0040, 0x0000), (0x4000, 0x0004)]

# ==============================================================================
# 3. KHỞI CHẠY TRAINING
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(f"🚀 BẮT ĐẦU EXPERIMENT: {experiment_name}")
    print(f"📂 KẾT QUẢ SẼ LƯU TẠI: {OUTPUT_DIR}")
    print("=" * 80)

    # Bắt đầu train từ round 5
    results = train.train_neural_distinguishers(
        output_dir=OUTPUT_DIR,
        starting_round=5,
        feature_mode=feature_mode
    )

    print(f"✅ Hoàn thành! Best round: {results[0]} - Accuracy: {results[1]}")
