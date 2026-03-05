# Copyright 2026 tistyse
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import cv2
import numpy as np
import mss
import pytesseract
import hashlib
import requests
import ctypes
import logging
from logging.handlers import RotatingFileHandler
from collections import OrderedDict
from difflib import SequenceMatcher
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QColor

if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

log_file_path = os.path.join(base_path, "debug.log")

try:
    if os.path.exists(log_file_path):
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.truncate(0)
except Exception:
    pass

logger = logging.getLogger(__name__)

if logger.hasHandlers():
    logger.handlers.clear()

rotating_handler = RotatingFileHandler(
    log_file_path, 
    maxBytes=5*1024*1024, 
    backupCount=3, 
    encoding='utf-8',
    mode='w'
)
stream_handler = logging.StreamHandler(sys.stdout)
log_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

rotating_handler.setFormatter(log_format)
stream_handler.setFormatter(log_format)

logger.setLevel(logging.INFO)
logger.addHandler(rotating_handler)
logger.addHandler(stream_handler)
logger.propagate = False

logger.info("==========================================")
logger.info("Application Starting.")
logger.info("==========================================")

tess_root = os.path.abspath(os.path.join(base_path, 'Tesseract-OCR'))
TESSERACT_PATH = os.path.join(tess_root, 'tesseract.exe')
tessdata_dir = os.path.join(tess_root, 'tessdata')

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
TESS_CONFIG = f'--tessdata-dir {tessdata_dir} --psm 7'

logger.info(f"Tesseract Path: {TESSERACT_PATH}")
logger.info(f"Data File Exists: {os.path.exists(os.path.join(tessdata_dir, 'eng.traineddata'))}")
logger.info(f"Config: {TESS_CONFIG}")

GITHUB_TXT_URL = "https://raw.githubusercontent.com/tistyse/TeamCheckTown/refs/heads/main/friendly.txt"
LOCAL_CACHE_FILE = os.path.join(base_path, "ally_cache.txt")
SIMILARITY_THRESHOLD = 0.7

class LimitedCache(OrderedDict):
    def __init__(self, maxsize=1000):
        super().__init__()
        self.maxsize = maxsize
    def __setitem__(self, key, value):
        if len(self) >= self.maxsize:
            self.popitem(last=False)
        super().__setitem__(key, value)

line_cache = LimitedCache()

def load_ally_list():
    logger.info("Loading ally list...")
    try:
        response = requests.get(GITHUB_TXT_URL, timeout=5)
        if response.status_code == 200:
            names = [l.strip().upper() for l in response.text.splitlines() if l.strip()]
            with open(LOCAL_CACHE_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(names))
            logger.info(f"Synced {len(names)} entries from GitHub.")
            return names
    except Exception as e:
        logger.warning(f"Network sync failed: {e}. Falling back to local cache.")
    
    if os.path.exists(LOCAL_CACHE_FILE):
        with open(LOCAL_CACHE_FILE, "r", encoding="utf-8") as f:
            names = [l.strip().upper() for l in f.read().splitlines() if l.strip()]
            logger.info(f"Loaded {len(names)} entries from local cache.")
            return names
    return []

ALLY_LIST = load_ally_list()

class OCRWorker(QThread):
    result_signal = pyqtSignal(str, str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.running = True
        self.prev_frame_hash = None

    def normalize(self, text):
        if not text: return ""
        t = text.upper().strip()
        repl = {'1':'I', 'L':'I', '|':'I', '0':'O', '5':'S', '8':'B'}
        return "".join(repl.get(c, c) for c in t if c.isalnum())

    def run(self):
        logger.info("OCR Worker Thread started.")
        grab_area = {
            "top": self.cfg["top"], "left": self.cfg["left"], 
            "width": self.cfg["width"], "height": self.cfg["height"]
        }
        
        with mss.mss() as sct:
            while self.running:
                try:
                    sct_img = sct.grab(grab_area)
                    img = np.array(sct_img)[:, :, :3]
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    thresh = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 30, 255]))
                    
                    curr_hash = hashlib.md5(thresh.tobytes()).hexdigest()
                    if curr_hash == self.prev_frame_hash:
                        self.msleep(150)
                        continue
                    self.prev_frame_hash = curr_hash

                    line_h = thresh.shape[0] // 3
                    detected_status = "NONE"
                    detected_name = ""

                    for i in range(3):
                        line_img = thresh[i*line_h : (i+1)*line_h, :]
                        if np.std(line_img) < 5: continue 
                        
                        line_hash = hashlib.md5(line_img.tobytes()).hexdigest()
                        text = line_cache.get(line_hash)
                        
                        if text is None:
                            text = pytesseract.image_to_string(line_img, config=TESS_CONFIG).strip()
                            line_cache[line_hash] = text
                        
                        if text:
                            norm_name = self.normalize(text)
                            is_ally = any(
                                ally in norm_name or 
                                (len(norm_name) > 2 and SequenceMatcher(None, norm_name, ally).ratio() >= SIMILARITY_THRESHOLD) 
                                for ally in ALLY_LIST
                            )
                            
                            detected_name = text.replace('\n', ' ').strip()
                            if is_ally:
                                detected_status = "ALLY"
                                break
                            else:
                                detected_status = "ENEMY"
                    
                    self.result_signal.emit(detected_status, detected_name)
                    self.msleep(100)

                except Exception as e:
                    logger.error(f"Worker Error: {e}")
                    self.msleep(1000)

    def stop(self):
        self.running = False
        self.wait()

class Overlay(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_status = "NONE"
        self.init_monitor()
        self.init_ui()
        self.worker = OCRWorker(self.cfg)
        self.worker.result_signal.connect(self.update_status)
        self.worker.start()

    def init_monitor(self):
        with mss.mss() as sct:
            monitors = sct.monitors
            target_idx = 1
            max_area = 0
            
            for i in range(1, len(monitors)):
                m = monitors[i]
                area = m["width"] * m["height"]
                if area > max_area:
                    max_area = area
                    target_idx = i
            
            target = monitors[target_idx]
            full = monitors[0] 
            w, h = 640, 72
            self.cfg = {
                "top": target["top"] + target["height"] - h,
                "left": target["left"] + (target["width"] // 2) - (w // 2),
                "width": w, 
                "height": h,
                "full": full, 
                "target": target
            }
            logger.info(f"selected monitor: {target_idx} resolution: {target['width']}x{target['height']})")

    def init_ui(self):
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        f = self.cfg["full"]
        self.setGeometry(f["left"], f["top"], f["width"], f["height"])

    def update_status(self, status, name):
        if self.current_status != status:
            if status != "NONE":
                logger.info(f"Status: {status} | Name: [{name}]")
            else:
                logger.info(f"Status: {status}")
            
            self.current_status = status
            self.update()

    def paintEvent(self, event):
        if self.current_status == "NONE": return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        t, f = self.cfg["target"], self.cfg["full"]
        
        color = QColor(0, 255, 0, 180) if self.current_status == "ALLY" else QColor(255, 0, 0, 180)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        
        aim_x = (t["left"] - f["left"]) + (t["width"] // 2)
        aim_y = (t["top"] - f["top"]) + (t["height"] // 2)
        painter.drawEllipse(aim_x - 8, aim_y - 8, 16, 16)

    def closeEvent(self, event):
        self.worker.stop()
        logger.info("==========================================")
        logger.info("Application Shutdown.")
        logger.info("==========================================")
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if not os.path.exists(TESSERACT_PATH):
        logger.critical(f"Missing Tesseract: {TESSERACT_PATH}")
        sys.exit(1)
    overlay = Overlay()
    overlay.show()
    sys.exit(app.exec_())
