import os
from datetime import datetime

def generate_experiment_path(category="train", prefix="", suffix=""):
    """
    Tạo đường dẫn thư mục lưu trữ tự động dựa trên thời gian.
    Mặc định lưu vào `../results/<category>/<timestamp>_<suffix>`
    (Giả định working directory đang là `src/` hoặc root project).
    
    Args:
        category (str): 'train' hoặc 'explore'. Xác định thư mục cha trong `results/`.
        prefix (str): Chuỗi đặt trước timestamp (tuỳ chọn).
        suffix (str): Chuỗi đặt sau timestamp (tuỳ chọn, ví dụ: 'speck32_6r_raw').
        
    Returns:
        str: Đường dẫn tuyệt đối đến thư mục experiment.
    """
    # Lấy timestamp hiện tại
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Tạo tên thư mục
    folder_parts = []
    if prefix:
        folder_parts.append(prefix)
    
    folder_parts.append(timestamp)
    
    if suffix:
        folder_parts.append(suffix)
        
    folder_name = "_".join(folder_parts)
    
    # Xác định đường dẫn gốc của project
    # File này nằm ở src/data_utils/name_generator.py
    # Nên thư mục gốc là 2 cấp đi lên (..)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Kết hợp lại thành path hoàn chỉnh
    results_dir = os.path.join(project_root, 'results', category, folder_name)
    
    return results_dir
