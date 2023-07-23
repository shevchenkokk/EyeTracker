"""
Пример работы библиотеки eye_tracker
"""

import cv2
from eye_tracker import EyeTracker

def main():
    # создание экземпляра класса
    eye_tracker = EyeTracker()
    
    # загрузка изображения с камеры
    cap = cv2.VideoCapture(0)

    while True:
        # чтение кадра
        ret, frame = cap.read()

        # уменьшаем размер кадра в два раза для достижения
        # быстрой скорости распознавания
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        # обновление кадра в айтрекере
        eye_tracker.update_frame(frame)

        # отображение состояния левого и правого глаза (открыт / закрыт)
        if eye_tracker.is_left_eye_opened():
            cv2.putText(frame, 'Left eye is opened', (10, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        elif eye_tracker.is_left_eye_closed():
            cv2.putText(frame, 'Left eye is closed', (10, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
        if eye_tracker.is_right_eye_opened():
            cv2.putText(frame, 'Right eye is opened', (10, 60),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        elif eye_tracker.is_right_eye_closed():
             cv2.putText(frame, 'Right eye is closed', (10, 60),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # получение текущего числа морганий
        blinks_counter = eye_tracker.get_blinks_counter()
        
        # отображение количества морганий
        if eye_tracker.are_eyes_on_frame:
            cv2.putText(frame, f'Blinks: {blinks_counter}', (10, 90),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # определение направления взгляда
        if eye_tracker.is_gaze_left():
            cv2.putText(frame, 'Gaze direction: left', (10, 120),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        elif eye_tracker.is_gaze_center():
            cv2.putText(frame, 'Gaze direction: center', (10, 120),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        elif eye_tracker.is_gaze_right():
            cv2.putText(frame, 'Gaze direction: right', (10, 120),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # отображение текущего кадра в окне
        cv2.imshow('Eye tracking', frame)

        # установка паузы в 1 мс между видеокадрами
        pressed_key = cv2.waitKey(1)

        # отслеживание работы с клавиатурой
        if pressed_key == 27 or pressed_key == ord('q'):
            break

    # освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()