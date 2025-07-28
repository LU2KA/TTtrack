"""
Constants for video processing and OCR configuration.
"""

import numpy as np

PLAYER_NAME_CONFIG = r"--oem 3 --psm 6"
CUSTOM_CONFIG = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"


LEFT_TOP_COR_PLAYER = (37, 948)
RIGHT_BOTTOM_COR_PLAYER = (389, 1007)

LEFT_TOP_COR_SET = (498, 948)
RIGHT_BOTTOM_COR_SET = (568, 1007)

LEFT_TOP_COR_SCORE = (567, 948)
RIGHT_BOTTOM_COR_SCORE = (640, 1007)

PLAYER_SHIFT_Y = 63
SCORE_SHIFT_X = 75

FRAME_INTERVAL = 64
FRAME_INTERVAL_SECONDS = 30

THRESHOLD = 0.95

MAX_PIXEL_JUMP = 50

TABLE_POLYGON = np.array([
    [947, 506],
    [1251, 414],
    [1532, 452],
    [1277, 562],
])
