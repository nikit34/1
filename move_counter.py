import numpy as np
import cv2
import pandas as pd
import sys


class Data:
    def __init__(self, env):
        self.env = env

    def read_file(self) -> str:
        path = 'data/' + self.env + '/'
        sys.path.insert(0, path)
        file = open(r'buffer.txt', 'r')
        return file.read()

    def get_param_camera(self):
        cap = cv2.VideoCapture(self.read_file())
        frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return [frames_count, fps, width, height]

    def __str__(self):
        return self.env




df = pd.DataFrame()
df.index.name = "Frames"

frame_number = 0
obj_cross_up = 0
obj_cross_down = 0
obj_id = []
obj_crossed = []
obj_total = 0

back_bg = cv2.createBackgroundSubtractorMOG2()

ret, frame = cap.read()
ratio = .5
image = cv2.resize(frame, (0, 0), None, ratio, ratio)

if __name__ == 'main':
    Data('local')

