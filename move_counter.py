import numpy as np
import cv2
import pandas as pd
import os


class Interface:
    def __init__(self, space):
        if space == 'main':
            # createBackgroundSubtractorMOG2
            self.history = 20
            self.varThreshold = 0
            self.detectShadows = False

            # resize
            self.ratio = 0.5

            # get_structuring_element
            self.shape = cv2.MORPH_ELLIPSE
            self.ksize = (20, 20)
            self.anchor = (-1, -1)

            # threshold
            self.thresh = 200
            self.maxval = 250
            self.type = cv2.THRESH_TRIANGLE

        elif space == 'Camera':
            self.record = 'buffer.txt'

        elif space == 'VideoStatistic':
            self.name_video = 'test0.avi'

        elif space == 'FileStatistic':
            self.statistic = 'test0.csv'

        elif space == 'FilterArea':
            self.min_area = 500
            self.max_area = 1500
            self.max_rad = 12

        if space in ['Camera', 'VideoStatistic', 'FileStatistic']:
            self.env = '/local/'
            self.root_path = os.path.dirname(os.path.abspath(__file__)) + '/data'

        if space in ['LineBounds', 'FilterArea']:
            self.lines = [
                {
                    'p1': (0, 5),
                    'p2': (100, 5),
                    'rgb': (0, 0, 255),
                    'bond': 3
                },
                {
                    'p1': (0, 30),
                    'p2': (100, 30),
                    'rgb': (0, 0, 255),
                    'bond': 3
                },
                {
                    'p1': (30, 0),
                    'p2': (30, 100),
                    'rgb': (0, 0, 255),
                    'bond': 3
                },
                {
                    'p1': (5, 0),
                    'p2': (5, 100),
                    'rgb': (0, 0, 255),
                    'bond': 3
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
        self.create_lines()

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


class FilterArea(Interface):
    def __init__(self):
        super(FilterArea, self).__init__('FilterArea')

    def update(self, current_contours):
        len_contours = len(current_contours)
        cxx = np.zeros(len_contours)
        cyy = np.zeros(len_contours)

        for i in range(len_contours):
            if hierarchy[0][i][3] == -1:
                area = cv2.contourArea(current_contours[i])
                if self.min_area < area < self.max_area:
                    cnt = current_contours[i]
                    # считаем момент - центр масс
                    moment = cv2.moments(array=cnt, binaryImage=True)
                    # ищем координаты центра
                    cx = int(moment.m10 / moment.m00)
                    cy = int(moment.m01 / moment.m00)

                    # считаем пересечения. Цикл - количество переходов каждой линии свопадает с ее значением cross
                    for i, line in enumerate(self.lines):
                        if pass


if __name__ == "__main__":
    interface = Interface('main')
    camera = Camera()
    console = Console()
    file_statistic = FileStatistic()
    video_statistic = VideoStatistic()
    line_bounds = LineBounds()
    filter_area = FilterArea()

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

        hull = [cv2.convexHull(c) for c in contours]
        cv2.drawContours(image, hull, -1, (0, 255, 0), 2)

        line_bounds.update_lines(image)
        filter_area.update(contours)

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
