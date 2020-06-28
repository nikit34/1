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

    def stop_record(self):
        self.cap.release()


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
        print('******************************************************')
        print('****************** Параметры видео *******************')
        print('* Количество кадров:                            ', self.params['frames_count'])
        print('* FPS:                                          ', self.params['fps'])
        print('* разрешение:                         ', self.params['width'], ' x, ', self.params['height'], ' px')
        print('* Продолжительность:                       ', self.params['frames_count'] / self.params['fps'], ' мин.')
        print('******************************************************')


class Write(Camera):
    def __init__(self, file, env='local'):
        self.file = file
        self.env = env
        self.params = self.get_param_camera()
        super(Camera, self).__init__()

    def get_video(self):
        path = 'data/' + self.env + '/'
        sys.path.insert(0, path)
        out_video = cv2.VideoWriter(
            path + self.file,
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            self.params['fps'],
            (self.params['height'], self.params['width']),
            True,
        )
        return out_video


if __name__ == 'main':
    Console().log_input()
    Statistic().set_index()
    camera = Camera()
    ret, frame = camera.list_read()
    ratio = .5

    video = Write('test0.avi').get_video()

    while ret:
        image = cv2.resize(frame, (0, 0), None, ratio, ratio)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foreground_mask = cv2.createBackgroundSubtractorMOG2().apply(gray)  # создание фона вычитания ч/б изображения

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # применение морфологического ядра
        closing = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)  # удаляем черный шум внутри белых частей
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)  # удаляем белый шум снаружи черных частей
        dilation = cv2.dilate(opening, kernel)  # выравниваем границы по внешнему контуру
        _, arr_bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # для разделения на 0 и max
        contours, hierarchy = cv2.findContours(arr_bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hull = [cv2.convexHull(c) for c in contours]
        cv2.drawContours(image, hull, -1, (0, 255, 0), 3)

        cv2.imshow("contours", image)
        cv2.moveWindow("contours", 0, 0)
        cv2.imshow("foreground mask", foreground_mask)
        cv2.moveWindow("foreground mask", 0, image.shape[0])

        video.write(image)

    camera.stop_record()
    cv2.destroyAllWindows()



