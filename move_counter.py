import numpy as np
import cv2
import pandas as pd
import sys


class Data:
    def __init__(self, env='local'):
        self.env = env

    def read_file(self) -> str:
        path = 'data/' + self.env + '/'
        sys.path.insert(0, path)
        file = open(r'buffer.txt', 'r')
        return file.read()


class Camera(Data):
    def __init__(self):
        self.cap = cv2.VideoCapture(self.read_file())
        super(Data).__init__('local')

    def get_param_camera(self):
        params = {
            'frames_count': self.cap.get(cv2.CAP_PROP_FRAME_COUNT),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'width': self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        }
        return params

    def list_read(self):
        return list(self.cap.read())


class Statistic:
    def __init__(self):
        self.df = pd.DataFrame()
        self.frame_number = 0
        self.obj_cross_up = 0
        self.obj_cross_down = 0
        self.obj_id = []
        self.obj_crossed = []
        self.obj_total = 0

    def set_index(self):
        self.df.index.name = "Frames"


class Console(Camera):
    def __init__(self):
        self.params = self.get_param_camera()
        super(Camera).__init__()

    def log_input(self):
        print('**********************************************************')
        print('****************** Параметры видео ***********************')
        print('* Количество кадров:                                ', self.params['frames_count'])
        print('* FPS:                                              ', self.params['fps'])
        print('* разрешение:                             ', self.params['width'], ' x, ', self.params['height'], ' px')
        print('* Продолжительность:                             ', self.params['frames_count'] / self.params['fps'], ' мин.')
        print('**********************************************************')


if __name__ == 'main':
    Console().log_input()
    Statistic().set_index()
    back_bg = cv2.createBackgroundSubtractorMOG2()
    ret, frame = Camera().list_read()
    ratio = .5
    image = cv2.resize(frame, (0, 0), None, ratio, ratio)

    while ret:
        pass



