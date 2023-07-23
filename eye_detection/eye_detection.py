import cv2
import os

def main():
    # загрузка классификаторов для обнаружения лица и глаз
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    face_cascade = cv2.CascadeClassifier(f'{absolute_path}/models/haarcascade_frontalface_default.xml')
    left_eye_cascade = cv2.CascadeClassifier(f'{absolute_path}/models/haarcascade_lefteye_2splits.xml')
    right_eye_cascade = cv2.CascadeClassifier(f'{absolute_path}/models/haarcascade_righteye_2splits.xml')

    # загрузка изображения с камеры
    cap = cv2.VideoCapture(0)

    # захват видео с камеры и анализ закрытия глаз в реальном времени
    while True:
        # чтение кадра
        ret, frame = cap.read()

        # преобразование цветного кадра в оттенки серого
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # обнаружение лиц на кадре
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(20, 20)
        )

        # обводка каждого найденного лица в прямоугольник
        for (x, y, width, height) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (255, 0, 0),
                2,
            )
            
            # вырезка из кадра области с лицом, соответствующей левому глазу
            gray_left_frame = gray_frame[y:y + height, x + width // 2: x + width]
            left_frame = frame[y: y + height, x + width // 2: x + width]

            # обнаружение левого глаза в области
            left_eye = left_eye_cascade.detectMultiScale(
                gray_left_frame,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(5, 5)
            )

            # обводка найденного левого глаза в круг
            for (eye_x, eye_y, eye_width, eye_height) in left_eye:
                # определение координат центра левого глаза
                eye_center = (eye_x + eye_width // 2, eye_y + eye_height // 2)

                # определение радиуса левого глаза
                eye_radius = int((eye_width + eye_height) / 4)

                # обводка найденного левого глаза в круг
                cv2.circle(left_frame, eye_center, eye_radius, (0, 255, 0), 2)

            # вырезка из кадра области с лицом, соответствующей правому глазу
            gray_right_frame = gray_frame[y:y + height, x: x + width // 2]
            right_frame = frame[y: y + height, x: x + width // 2]

            # обнаружение правого глаза в области
            right_eye = right_eye_cascade.detectMultiScale(
                gray_right_frame,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(5, 5)
            )

            # обводка найденного правого глаза в круг
            for (eye_x, eye_y, eye_width, eye_height) in right_eye:
                # определение координат центра правого глаза
                eye_center = (eye_x + eye_width // 2, eye_y + eye_height // 2)

                # определение радиуса правого глаза
                eye_radius = int((eye_width + eye_height) / 4)

                # обводка найденного правого глаза в круг
                cv2.circle(right_frame, eye_center, eye_radius, (0, 255, 0), 2)

        # отображение текущего кадра в окне
        cv2.imshow('Eyetracking', frame)

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