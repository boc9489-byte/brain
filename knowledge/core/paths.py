from pathlib import Path
import os

KNOWEDGE_ROOT = Path(__file__).resolve().parent.parent

print(f"{KNOWEDGE_ROOT}")
LOCAL_BASE_DIR = os.path.join(KNOWEDGE_ROOT,"temp_data")

FRONT_PAGE_DIR = os.path.join(KNOWEDGE_ROOT, "front")

def get_local_base_dir()->str:
    return LOCAL_BASE_DIR

def get_front_page_dir() ->str:
    return FRONT_PAGE_DIR