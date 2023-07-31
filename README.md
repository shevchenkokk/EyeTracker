# Айтрекер для анализа состояния глаз, моргания и направления взгляда

Данный проект разработан для обнаружения лица и глаз на кадре с последующим 
анализом в режиме реального времени с использованием пакетов `cv2` и `dlib`.

## Начало работы

Перед началом работы с проектом необходимо создать виртуальное окружение и 
выполнить установку зависимостей при помощи менеджера пакетов `pip`.

Создание виртуального окружения:
```
conda create --name <имя_окружения> python=3.11.4
```

Активация окружения:
```
conda activate <имя_окружения>
```

Установка необходимых зависимостей:

```
pip install -r requirements.txt
```

Запуск тестового файла осуществляется при помощи команды

```
python3 main.py
```

## Пример тестового файла

```python
import cv2
from eye_tracker import EyeTracker

def main():
    eye_tracker = EyeTracker()
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        eye_tracker.update_frame(frame)

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
        
        blinks_counter = eye_tracker.get_blinks_counter()
        
        if eye_tracker.are_eyes_on_frame:
            cv2.putText(frame, f'Blinks: {blinks_counter}', (10, 90),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        if eye_tracker.is_gaze_left():
            cv2.putText(frame, 'Gaze direction: left', (10, 120),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        elif eye_tracker.is_gaze_center():
            cv2.putText(frame, 'Gaze direction: center', (10, 120),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        elif eye_tracker.is_gaze_right():
            cv2.putText(frame, 'Gaze direction: right', (10, 120),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Eye tracking', frame)

        pressed_key = cv2.waitKey(1)

        if pressed_key == 27 or pressed_key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

## Документация

В приведённом выше тестовом примере переменная `eye_tracker` ссылается на 
объект класса `EyeTracker`, который автоматически добавляется в текущее 
пространство имён при импорте библиотеки.

### Обновление и анализ кадра

```python
eye_tracker.update_frame(frame)
```

При работе с видеопотоком, необходимо осуществлять вызов данного метода в 
цикле, как показано в примере выше.

### Проверка на наличие глаз в кадре

```python
eye_tracker.are_eyes_on_frame
```

Возвращает `True`, если глаза находятся в текущем кадре, иначе – `False`.

### Определение состояния левого глаза

```python
eye_tracker.is_left_eye_opened()
```

Возвращает `True`, если глаза находятся в кадре и левый глаз находится в 
открытом состоянии, иначе – 
`False`.

```python
eye_tracker.is_left_eye_closed()
```

Возвращает `True`, если глаза находятся в кадре и левый глаз находится в 
закрытом состоянии, иначе – 
`False`.

### Определение состояния правого глаза

```python
eye_tracker.is_right_eye_opened()
```

Возвращает `True`, если глаза находятся в кадре и правый глаз находится в 
открытом состоянии, иначе – 
`False`.

```python
eye_tracker.is_right_eye_closed()
```

Возвращает `True`, если глаза находятся в кадре и правый глаз находится в 
закрытом состоянии, иначе – 
`False`.

### Получение текущего числа морганий

```python
blinks_counter = eye_tracker.get_blinks_counter()
```

Сохраняет в переменную текущее количество морганий с момента создания 
экземпляра класса `EyeTracker`.

### Определение направления взгляда

```python
eye_tracker.is_gaze_left()
```

Возвращает `True`, если глаза находятся в кадре и взгляд направлен влево, иначе 
– `False`.

```python
eye_tracker.is_gaze_center()
```

Возвращает `True`, если глаза находятся в кадре и взгляд направлен в центр, 
иначе – `False`.

```python
eye_tracker.is_gaze_right()
```

Возвращает `True`, если глаза находятся в кадре и взгляд направлен вправо, 
иначе – `False`.