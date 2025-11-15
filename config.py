# config.py
from pathlib import Path

# ====== 数据根目录 ======
DATA_ROOT = Path("/Users/xrz/Documents/project1/data")

# ====== 数据集选择 ======
DATASET_NAME = "FAZ"   # 使用 FAZ

# ====== 数据格式 ======
DATA_FORMAT = "h5"     # 使用 h5 格式

def get_domain_path(domain_id: int):
    dom = f"domain{domain_id}"
    if DATA_FORMAT == "h5":
        return DATA_ROOT / f"{DATASET_NAME}_h5" / dom
    else:
        return DATA_ROOT / f"{DATASET_NAME}_png" / dom
