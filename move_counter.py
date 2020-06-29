import numpy as np
import cv2
import pandas as pd
import os


class Interface:
    def __init__(self):
        self.env = '/local/'
        self.video = 'test0.avi'
        self.record = 'buffer.txt'
        self.statistic = 'test0.csv'
        self.root_path = os.path.dirname(os.path.abspath(__file__)) + '/data'

        # resize
        self.ratio = 0.5

        # getStructuringElement
        self.shape = cv2.MORPH_ELLIPSE
        self.ksize = (20, 20)
        self.anchor = (-1, -1)

        # threshold
        self.thresh = 200
        self.maxval = 250
        self.type = cv2.THRESH_TRIANGLE

        # createBackgroundSubtractorMOG2
        self.history = 20
        self.varThreshold = 0
        self.detectShadows = False

        # LineBounds
        self.lines = [
            {
                'p1': (0, 5),
                'p2': (100, 5),
                'rgb': (255, 0, 0),
                'bond': 3
            },
            {
                'p1': (0, 30),
                'p2': (100, 30),
                'rgb': (0, 255, 0),
                'bond': 3
            },
        ]


class Camera(Interface):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(self.get_full_path_input())

    def get_param_camera(self):
        params = {
            'frames_count': self.cap.get(cv2.CAP_PROP_FRAME_COUNT),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'width': self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        }
        return params

    def read(self):
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


class VideoStatistic(Camera, Interface):
    def __init__(self):
        super().__init__()
        self.params = self.get_param_camera()

    def set_record(self):
        path = self.root_path + self.env + 'out/' + self.video
        out_video = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            int(self.params['fps']),
            (int(self.params['height']), int(self.params['width'])),
            True,
        )
        return out_video


class LineBounds(Camera, Interface):
    def __init__(self):
        super().__init__()
        self.param = self.get_param_camera()
        self.create_lines()

    def create_lines(self):
        for line in self.lines:
            coord_p1 = (
                int(self.param['width'] * line['p1'][0] / 100),
                int(self.param['height'] * line['p1'][1] / 100)
            )
            coord_p2 = (
                int(self.param['width'] * line['p2'][0] / 100),
                int(self.param['height'] * line['p2'][1] / 100)
            )
            rgb = line['rgb']
            bond = line['bond']
            cv2.line(image, coord_p1, coord_p2, rgb, bond)


if __name__ == "__main__":
    interface = Interface()
    camera = Camera()
    console = Console()
    file_statistic = FileStatistic()
    video_statistic = VideoStatistic()

    console.log_input()
    file_statistic.set_index()

    foreground_bg = cv2.createBackgroundSubtractorMOG2(
        history=interface.history,
        varThreshold=interface.varThreshold,
        detectShadows=interface.detectShadows
    )  # разделить на два слоя bg и fg

    ret = camera.read()[0]
    video = video_statistic.set_record()

    while ret:
        ret, frame = camera.read()
        try:
            image = cv2.resize(src=frame, dsize=(0, 0), dst=None, fx=interface.ratio, fy=interface.ratio)
        except Exception as e:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foreground_mask = foreground_bg.apply(gray)  # создание фона вычитания ч/б изображения

        kernel = cv2.getStructuringElement(
            shape=interface.shape,
            ksize=interface.ksize,
            anchor=interface.anchor
        )  # применение морфологического ядра
        closing = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)  # удаляем черный шум внутри белых частей
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)  # удаляем белый шум снаружи черных частей
        dilation = cv2.dilate(opening, kernel)  # выравниваем границы по внешнему контуру
        _, arr_bins = cv2.threshold(
            src=dilation,
            thresh=interface.thresh,
            maxval=interface.maxval,
            type=interface.type
        )  # разделение по thresh с присвоением 0 или max из всех значений
        contours, hierarchy = cv2.findContours(arr_bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hull = [cv2.convexHull(c) for c in contours]
        cv2.drawContours(image, hull, -1, (0, 255, 0), 1)

        linies = LineBounds()

        cv2.imshow("contours", image)
        cv2.moveWindow("contours", 0, 0)
        cv2.imshow("foreground mask", foreground_mask)
        cv2.moveWindow("foreground mask", image.shape[1], 0)

        if cv2.waitKey(int(1000 / camera.get_param_camera()['fps'])) & 0xff == 27:  # 0xff <-> 255
            break
        video.write(image)

    camera.stop_record()
    cv2.destroyAllWindows()

    file_statistic.df.to_csv(interface.root_path + interface.env + 'out/' + interface.statistic, mode='w+', sep=',')
