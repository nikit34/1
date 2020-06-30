import numpy as np
import cv2
import pandas as pd
import os
import math
import time


class Interface:
    def __init__(self, space):
        if space == 'main':
            # createBackgroundSubtractorMOG2
            self.history = 25
            self.varThreshold = 0
            self.detectShadows = False

            # resize
            self.ratio = 0.5

            # get_structuring_element
            self.shape = cv2.MORPH_ELLIPSE
            self.ksize = (20, 20)
            self.anchor = (-1, -1)

            # threshold
            self.thresh = 240
            self.maxval = 255
            self.type = cv2.THRESH_TRIANGLE

            # count_cross_line
            self.min_area = 5000
            self.max_area = 8000

        elif space == 'Camera':
            self.record = 'buffer.txt'

        elif space == 'VideoStatistic':
            self.name_video = 'test0.avi'

        elif space == 'FileStatistic':
            self.statistic = 'test0.csv'

        elif space == 'CountCrossLine':
            # self.max_rad = 12
            self.epsilon = 200
            self.timeout = 0.5

        if space in ['Camera', 'VideoStatistic', 'FileStatistic']:
            self.env = '/local/'
            self.root_path = os.path.dirname(os.path.abspath(__file__)) + '/data'

        if space in ['LineBounds', 'CountCrossLine']:
            self.lines = [
                {
                    'id_': 'top',
                    'p1': (0, 1),
                    'p2': (100, 1),
                    'rgb': (0, 0, 255),
                    'bond': 2,
                    'cross': 2,
                },
                {
                    'id_': 'bottom',
                    'p1': (0, 30),
                    'p2': (100, 30),
                    'rgb': (0, 0, 255),
                    'bond': 2,
                    'cross': 2,
                },
                {
                    'id_': 'left',
                    'p1': (10, 0),
                    'p2': (10, 100),
                    'rgb': (0, 0, 255),
                    'bond': 2,
                    'cross': 4,
                },
                {
                    'id_': 'right',
                    'p1': (28, 0),
                    'p2': (28, 100),
                    'rgb': (0, 0, 255),
                    'bond': 2,
                    'cross': 4,
                },
            ]


class Camera(Interface):
    def __init__(self):
        super(Camera, self).__init__('Camera')
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
        file = open(path + self.record)
        full_path = path + 'in/' + file.read()
        return full_path

    def stop_record(self):
        self.cap.release()


class Console(Camera):
    def __init__(self):
        super(Console, self).__init__()
        self.params = self.get_param_camera()

    def log_input(self):
        print('****************************************************')
        print('**************** Параметры видео *******************')
        print('* Количество кадров:                          ', self.params['frames_count'])
        print('* FPS:                                        ', self.params['fps'])
        print('* разрешение:                                 ',
              self.params['width'], ' x, ', self.params['height'], ' px')
        print('* Продолжительность:                     ',
              round((self.params['frames_count'] / (self.params['fps'] + 1)), 1), ' сек.')
        print('****************************************************')


class FileStatistic(Interface):
    def __init__(self):
        self.df = pd.DataFrame()
        self.frame_number = 0
        self.obj_cross_up = 0
        self.obj_cross_down = 0
        self.obj_id = []
        self.obj_crossed = []
        self.obj_total = 0
        super(FileStatistic, self).__init__('FileStatistic')

    def set_index(self):
        self.df.index.name = 'Frames'

    def save_data(self):
        self.df.to_csv(self.root_path + self.env + 'out/' + self.statistic, mode='w+')


class VideoStatistic(Camera, Interface):
    def __init__(self):
        Camera.__init__(self)
        Interface.__init__(self, 'VideoStatistic')
        self.params = self.get_param_camera()
        self.video = None

    def set_record(self):
        path = self.root_path + self.env + 'out/' + self.name_video
        self.video = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            int(self.params['fps']),
            (int(self.params['height']), int(self.params['width'])),
            True,
        )

    def write_record(self, img):
        self.video.write(img)


class LineBounds(Camera, Interface):
    def __init__(self):
        Camera.__init__(self)
        Interface.__init__(self, 'LineBounds')
        self.param = self.get_param_camera()
        self.count_lines = len(self.lines)
        self.coord_p1 = [(0, 0) for _ in range(self.count_lines)]
        self.coord_p2 = [(0, 0) for _ in range(self.count_lines)]
        self.rgb = [(0, 0, 0) for _ in range(self.count_lines)]
        self.bond = [0 for _ in range(self.count_lines)]

    def create_lines(self):
        for i, line in enumerate(self.lines):
            self.coord_p1[i] = (
                int(self.param['width'] * line['p1'][0] / 100),
                int(self.param['height'] * line['p1'][1] / 100)
            )
            self.coord_p2[i] = (
                int(self.param['width'] * line['p2'][0] / 100),
                int(self.param['height'] * line['p2'][1] / 100)
            )
            self.rgb[i] = line['rgb']
            self.bond[i] = line['bond']

    def update_lines(self, img):
        for i in range(self.count_lines):
            cv2.line(img, self.coord_p1[i], self.coord_p2[i], self.rgb[i], self.bond[i])


class CountCrossLine(Interface):
    def __init__(self):
        super(CountCrossLine, self).__init__('CountCrossLine')
        self.count_cross = [0 for _ in range(len(self.lines))]
        self.done_cross = [False for _ in range(len(self.lines))]
        self.total = 0
        self.last_time = [0. for _ in range(len(self.lines))]

    def filter_cross(self, current_contours, min_area, max_area):
        for cnt in self.filter_area(current_contours, min_area, max_area):
            moment = cv2.moments(array=cnt, binaryImage=False)
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])

            for i_line, line in enumerate(self.lines):
                if self.timeout < time.time() - self.last_time[i_line]:
                    if self.dist_point_line((cx, cy), line['p1'], line['p2']) < self.epsilon:
                        self.count_cross[i_line] += 1
                        self.last_time[i_line] = time.time()

                if self.count_cross[i_line] >= line['cross']:
                    self.done_cross[i_line] = True
                    if False not in self.done_cross:
                        print(11111111111111) 
                        self.update()

    def switch_color_line(self, obj_line_bounds):
        for i in range(len(self.lines)):
            if self.count_cross[i] == 1:
                obj_line_bounds.rgb[i] = (0, 100, 100)
            elif self.count_cross[i] == 2:
                obj_line_bounds.rgb[i] = (0, 200, 200)
            elif self.count_cross[i] == 3:
                obj_line_bounds.rgb[i] = (0, 255, 255)
            elif self.done_cross[i]:
                obj_line_bounds.rgb[i] = (0, 255, 0)
            else:
                obj_line_bounds.rgb[i] = (0, 0, 255)

    @staticmethod
    def dist_point_line(point, line1, line2):
        area_double_triangle = abs(
            (line2[1] - line1[1]) * point[0] -
            (line2[0] - line1[0]) * point[1] +
            line2[0] * line1[1] - line2[1] * line1[0]
        )
        dist_line = math.sqrt(pow((line2[1] - line1[1]), 2) + pow((line2[0] - line1[0]), 2))
        return int(area_double_triangle / dist_line)

    @staticmethod
    def filter_area(current_contours, min_area, max_area):
        len_contours = len(current_contours)
        cxx = np.zeros(len_contours)
        cyy = np.zeros(len_contours)

        for i_contour in range(len_contours):
            if hierarchy[0][i_contour][3] == -1:
                area = cv2.contourArea(current_contours[i_contour])
                if min_area < area < max_area:
                    yield current_contours[i_contour]

    def update(self):
        self.count_cross = [0 for _ in range(len(self.lines))]
        self.done_cross = [False for _ in range(len(self.lines))]
        self.total += 1


if __name__ == "__main__":
    interface = Interface('main')
    camera = Camera()
    console = Console()
    file_statistic = FileStatistic()
    video_statistic = VideoStatistic()
    line_bounds = LineBounds()
    count_cross_line = CountCrossLine()

    foreground_bg = cv2.createBackgroundSubtractorMOG2(
        history=interface.history,
        varThreshold=interface.varThreshold,
        detectShadows=interface.detectShadows
    )  # разделить на два слоя bg и fg

    ret = camera.read()[0]

    console.log_input()
    file_statistic.set_index()
    video_statistic.set_record()

    line_bounds.create_lines()

    while ret:
        ret, frame = camera.read()
        try:
            image = cv2.resize(src=frame, dsize=(0, 0), fx=interface.ratio, fy=interface.ratio)
        except cv2.error as e:
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

        hull = [cv2.convexHull(c) for c in count_cross_line.filter_area(
            contours, interface.min_area, interface.max_area
        )]
        cv2.drawContours(image, hull, -1, (255, 0, 0), 2)

        line_bounds.update_lines(image)
        count_cross_line.filter_cross(contours, interface.min_area, interface.max_area)
        count_cross_line.switch_color_line(line_bounds)

        cv2.imshow("contours", image)
        cv2.moveWindow("contours", 0, 0)
        cv2.imshow("foreground mask", foreground_mask)
        cv2.moveWindow("foreground mask", int(image.shape[1] * 1.2), 0)

        if cv2.waitKey(int(1000 / camera.get_param_camera()['fps'])) & 0xff == 27:  # 0xff <-> 255
            break

        video_statistic.write_record(image)

    camera.stop_record()
    cv2.destroyAllWindows()

    file_statistic.save_data()
