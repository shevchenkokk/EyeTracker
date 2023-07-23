import cv2
import numpy as np
import math

class Eye:
    # индексы ключевых точек для левого и правого глаза
    left_eye_indices = [36, 37, 38, 39, 40, 41]
    right_eye_indices = [42, 43, 44, 45, 46, 47]

    def __init__(self, frame, face_shape, is_left_eye):
        self.frame = frame
        self.face_shape = face_shape
        self.is_left_eye = is_left_eye
        self._analyze_frame()

    def _analyze_frame(self):
        # определение индексов ключевых точек глаза
        if self.is_left_eye:
            self.eye_indices = self.left_eye_indices
        else:
            self.eye_indices = self.right_eye_indices

        # вычисление координат ключевых точек глаза
        self.eye_points = []
        for index in self.eye_indices:
            self.eye_points.append(
                (self.face_shape.part(index).x, self.face_shape.part(index).y)
            )
    
        # отрисовка ключевых точек глаза
        for point_x, point_y in self.eye_points:
            cv2.circle(
                self.frame,
                (point_x, point_y),
                2,
                (0, 255, 0),
                1,
            )

    @staticmethod
    def _find_dist(vec1, vec2):
        return math.sqrt(
             math.pow(vec1[0] - vec2[0], 2) + math.pow(vec1[1] - vec2[1], 2)
        )

    def _get_eye_aspect(self):
        v_dist_first = self._find_dist(self.eye_points[1], self.eye_points[-1])
        v_dist_second = self._find_dist(self.eye_points[2], self.eye_points[-2])
        h_dist = self._find_dist(self.eye_points[0], self.eye_points[-3])

        self.eye_aspect = (v_dist_first + v_dist_second) / (2 * h_dist)
        
        return self.eye_aspect

    def _get_gaze_relation(self):
        mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
        cv2.polylines(mask, [np.array(self.eye_points)], True, 255, 2)
        cv2.fillPoly(mask, [np.array(self.eye_points)], 255)
        eye_frame = cv2.bitwise_and(self.frame, self.frame, mask=mask)


        # нахождение минимальных и максимальных координат глаза
        min_x = min(self.eye_points[i][0] for i in range(len(self.eye_points)))
        max_x = max(self.eye_points[i][0] for i in range(len(self.eye_points)))
        min_y = min(self.eye_points[i][1] for i in range(len(self.eye_points)))
        max_y = max(self.eye_points[i][1] for i in range(len(self.eye_points)))
        
        # вырезка области с глазом
        eye_frame = eye_frame[min_y: max_y, min_x: max_x]
        gray_eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        gray_eye_frame = cv2.GaussianBlur(gray_eye_frame, (5, 5), 0)
        gray_eye_frame = cv2.medianBlur(gray_eye_frame, 5)

        _, threshold = cv2.threshold(gray_eye_frame, 80, 255, cv2.THRESH_BINARY)
        threshold = cv2.resize(threshold, None, fx=5, fy=5)
        
        height, width = threshold.shape[:2]
        
        left_side_threshold = threshold[0: height, 0: int(width / 2)]
        right_side_threshold = threshold[0: height, int(width / 2): width]
        
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_white = cv2.countNonZero(right_side_threshold)

        match left_side_white, right_side_white:
            case 0, _:
                self.gaze_relation = 1
            case _, 0:
                self.gaze_relation = 5
            case _, _:
                self.gaze_relation = left_side_white / right_side_white

        return self.gaze_relation