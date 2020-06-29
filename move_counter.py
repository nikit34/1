import numpy as np
import cv2
import pandas as pd
import os


def get_env():
    return '/local/'


def get_video():
    return 'test0.avi'


def get_record():
    return 'buffer.txt'


def get_statistic():
    return 'test0.csv'


def get_root_path():
    return os.path.dirname(os.path.abspath(__file__)) + '/data'


class Camera:
    def __init__(self):
        self.root_path = get_root_path()
        self.env = get_env()
        self.record = get_record()
        self.cap = cv2.VideoCapture(self.get_full_path_input())

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

    def get_full_path_input(self) -> str:
        path = self.root_path + self.env
        file = open(path + self.record, 'r')
        full_path = path + 'in/' + file.read()
        return full_path

    def stop_record(self):
        self.cap.release()


class FileStatistic:
    def __init__(self):
        self.df = pd.DataFrame()
        self.frame_number = 0
        self.obj_cross_up = 0
        self.obj_cross_down = 0
        self.obj_id = []
        self.obj_crossed = []
        self.obj_total = 0

    def set_index(self):
        self.df.index.name = 'Frames'


class Console(Camera):
    def __init__(self):
        super().__init__()
        self.params = self.get_param_camera()

    def log_input(self):
        print('****************************************************')
        print('**************** Параметры видео *******************')
        print('* Количество кадров:                          ', self.params['frames_count'])
        print('* FPS:                                        ', self.params['fps'])
        print('* разрешение:                       ',
              self.params['width'], ' x, ', self.params['height'], ' px')
        print('* Продолжительность:                     ',
              self.params['frames_count'] / (self.params['fps'] + 1), ' мин.')
        print('****************************************************')


class VideoStatistic(Camera):
    def __init__(self):
        self.root_path = get_root_path()
        self.env = get_env()
        self.name_video = get_video()
        super().__init__()
        self.params = self.get_param_camera()

    def set_record(self):
        path = self.root_path + self.env + 'out/' + self.name_video
        out_video = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            int(self.params['fps']),
            (int(self.params['height']), int(self.params['width'])),
            True,
        )
        return out_video


if __name__ == "__main__":
    camera = Camera()
    console = Console()
    file_statistic = FileStatistic()
    video_statistic = VideoStatistic()

    console.log_input()
    file_statistic.set_index()
    ret = camera.list_read()[0]
    ratio = .5
    video = video_statistic.set_record()

    while ret:
        ret, frame = camera.list_read()
        try:
            image = cv2.resize(frame, (0, 0), None, ratio, ratio)
        except Exception as e:
            continue
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
        cv2.moveWindow("foreground mask", image.shape[1] + 20, 0)

        if cv2.waitKey(int(1000 / camera.get_param_camera()['fps'])) & 0xff == 27:  # 0xff <-> 255
            break
        video.write(image)

    camera.stop_record()
    cv2.destroyAllWindows()

    file_statistic.df.to_csv(get_root_path() + get_env() + 'out/' + get_statistic(), mode='w+', sep=',')


