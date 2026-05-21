"""
logger.py — UTF-8 safe dual logging (file + console).
Suppresses TF C++ noise and third-party library spam.
"""
import io
import logging
import os
import sys
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTHONIOENCODING",       "utf-8")

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(
    LOGS_DIR, datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + ".log"
)

_FMT = logging.Formatter(
    "[%(asctime)s]  %(levelname)-8s  %(name)-28s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(_FMT)

try:
    _sh = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                         errors="replace", line_buffering=True)
    )
except AttributeError:
    _sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_FMT)

logging.basicConfig(level=logging.INFO, handlers=[_fh, _sh])

for _lib in ("urllib3", "PIL", "matplotlib", "absl",
             "h5py", "tensorflow", "ultralytics", "roboflow"):
    logging.getLogger(_lib).setLevel(logging.ERROR)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
