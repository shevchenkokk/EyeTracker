import cv2
import dlib
import os

from .eye import Eye

class EyeTracker:
    # отношение между шириной и высотой координат ключевых точек глаза в открытом состоянии
    EYE_ASPECT_RELATION = 0.23

    def __init__(self):
        # загрузка модели для обнаружения лица
        self.face_detector = dlib.get_frontal_face_detector()

        # загрузка модели предиктора формы лица
        absolute_path = os.path.dirname(os.path.abspath(__file__))

        self.face_predictor = dlib.shape_predictor(f'{absolute_path}/models/shape_predictor_68_face_landmarks.dat')
        
        # текущий кадр
        self.frame = None

        # левый и правый глаз
        self.left_eye = None
        self.right_eye = None

        # счётчик морганий
        self.blinks_counter = 0

        # счётчик миллисекунд для подсчёта времени, в течение которого
        # глаза находятся в закрытом состоянии
        self.eyes_closed_time_counter = 0

    def _analyze_frame(self):  
        # преобразование цветного кадра в оттенки серого
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # обнаружение лиц на кадре
        faces = self.face_detector(gray_frame)

        # обработка найденного лица
        try:
            face = faces[0]
            face_x, face_y = face.left(), face.top()
            face_width, face_height = face.right() - face_x, face.bottom() - face_y
            
            # обводка найденного лица в прямоугольник
            cv2.rectangle(
                self.frame,
                (face_x, face_y),
                (face_x + face_width, face_y + face_height),
                (255, 0, 0),
                2,
            )

            # определение формы лица
            face_shape = self.face_predictor(gray_frame, face)

            # инициализация объектов, соответствующих левому и правому глазу
            self.left_eye = Eye(self.frame, face_shape, True)
            self.right_eye = Eye(self.frame, face_shape, False)

            self._check_eyes_state()
            self._check_gaze_direction()
        except IndexError:
            pass

    def update_frame(self, frame):
        self.frame = frame
        self._analyze_frame()

    def _check_eyes_state(self):
        # алгоритм проверки состояния левого и правого глаза (открыт / закрыт)
        left_eye_aspect = self.left_eye._get_eye_aspect()
        right_eye_aspect = self.right_eye._get_eye_aspect()
        
        self.eyes_aspect_avg = (left_eye_aspect + right_eye_aspect) / 2

        # подсчёт времени, в течение которого глаза находятся в закрытом состоянии
        # если итоговое время больше 3 мс – увеличиваем счётчик морганий
        # и обнуляем счётчик миллисекунд 
        if self.eyes_aspect_avg < self.EYE_ASPECT_RELATION:
            self.eyes_closed_time_counter += 1
        else:
            if self.eyes_closed_time_counter >= 3:
                self.blinks_counter += 1
                self.eyes_closed_time_counter = 0

    @property
    def are_eyes_on_frame(self):
        if self.left_eye is not None and self.right_eye is not None:
            return True
        return False

    def is_left_eye_opened(self):
        return not self.is_left_eye_closed() if self.are_eyes_on_frame else False

    def is_left_eye_closed(self):
        return self.left_eye.eye_aspect < self.EYE_ASPECT_RELATION if self.are_eyes_on_frame else False
    
    def is_right_eye_opened(self):
        return not self.is_right_eye_closed() if self.are_eyes_on_frame else False
    
    def is_right_eye_closed(self):
        return self.right_eye.eye_aspect < self.EYE_ASPECT_RELATION if self.are_eyes_on_frame else False
    
    def get_blinks_counter(self):
        return self.blinks_counter
    
    def _check_gaze_direction(self):
        gaze_relation_left = self.left_eye._get_gaze_relation()
        gaze_relation_right = self.right_eye._get_gaze_relation()

        self.gaze_relation_avg = (gaze_relation_left + gaze_relation_right) / 2

    def is_gaze_left(self):
        return not self.is_gaze_center() and not self.is_gaze_right() if self.are_eyes_on_frame else False

    def is_gaze_center(self):
        return 1 < self.gaze_relation_avg < 1.7 if self.are_eyes_on_frame else False

    def is_gaze_right(self):
        return self.gaze_relation_avg <= 1 if self.are_eyes_on_frame else False