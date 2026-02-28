import os
import sys
import cv2
import numpy as np
import mss
import easyocr
import hashlib
import torch
import requests
from difflib import SequenceMatcher

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

import PyQt5
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QPainter, QColor

SELECTED_MONITOR_INDEX = 2
SIMILARITY_THRESHOLD = 0.7
GITHUB_TXT_URL = "https://raw.githubusercontent.com/tistyse/TeamCheckTown/refs/heads/main/friendly.txt"

def get_friends_from_github(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            lines = response.text.splitlines()
            return [line.strip().upper() for line in lines if line.strip()]
    except:
        pass
    return ["dev_morn","mdtk_IO"]

FRIENDS = get_friends_from_github(GITHUB_TXT_URL)

def get_monitor_config(index):
    with mss.mss() as sct:
        monitors = sct.monitors
        if index >= len(monitors):
            index = 1
        full_virtual = monitors[0]
        target = monitors[index]
        sw, sh = target["width"], target["height"]
        ox, oy = target["left"], target["top"]
        width, height = 640, 72
        left = ox + (sw // 2) - (width // 2)
        top = oy + sh - height
        return {
            "top": top, "left": left, "width": width, "height": height,
            "full": full_virtual, "target": target
        }

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
line_cache = {}
prev_frame = None
current_status = "NONE"

def normalize(text):
    if not text: return ""
    t = text.upper().strip()
    repl = {'1':'I', 'L':'I', '|':'I', '!':'I', '0':'O', '5':'S', '8':'B'}
    return "".join(repl.get(c, c) for c in t if not c.isspace())

class Overlay(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.WindowTransparentForInput | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        f = self.cfg["full"]
        self.setGeometry(f["left"], f["top"], f["width"], f["height"])
        self.sct = mss.mss()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ocr)
        self.timer.start(200)

    def update_ocr(self):
        global current_status, prev_frame
        try:
            grab_area = {
                "top": self.cfg["top"], "left": self.cfg["left"],
                "width": self.cfg["width"], "height": self.cfg["height"]
            }
            img = np.array(self.sct.grab(grab_area))
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            if prev_frame is not None and gray.shape == prev_frame.shape:
                if np.mean(cv2.absdiff(prev_frame, gray)) < 0.5: return
            prev_frame = gray.copy()
            _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
            line_h = thresh.shape[0] // 3
            new_status = "NONE" 
            for i in range(3):
                line_img = thresh[i*line_h : (i+1)*line_h, :]
                if np.std(line_img) < 15: continue
                line_hash = hashlib.md5(line_img.tobytes()).hexdigest()
                text = line_cache.get(line_hash)
                if text is None:
                    res = reader.readtext(line_img, detail=0, paragraph=False, mag_ratio=1.2)
                    text = res[0] if res else ""
                    line_cache[line_hash] = text
                if text.strip():
                    norm_name = normalize(text)
                    is_ally = any(f in norm_name or SequenceMatcher(None, norm_name, f).ratio() >= SIMILARITY_THRESHOLD for f in FRIENDS)
                    new_status = "ALLY" if is_ally else "ENEMY"
                    break
            if current_status != new_status:
                current_status = new_status
                self.update()
        except: pass

    def paintEvent(self, event):
        if current_status == "NONE": return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        f_left, f_top = self.cfg["full"]["left"], self.cfg["full"]["top"]
        t = self.cfg["target"]
        color = QColor(0, 255, 0, 255) if current_status == "ALLY" else QColor(255, 0, 0, 255)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        aim_x = t["left"] + (t["width"] // 2) - f_left
        aim_y = t["top"] + (t["height"] // 2) - f_top
        painter.drawEllipse(aim_x + 25, aim_y - 10, 12, 12)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    monitor_cfg = get_monitor_config(SELECTED_MONITOR_INDEX)
    overlay = Overlay(monitor_cfg)
    overlay.show()
    sys.exit(app.exec_())
