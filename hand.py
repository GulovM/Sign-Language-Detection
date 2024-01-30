import cv2
import numpy as np 
import mediapipe as mp
import streamlit as st
from tensorflow.keras.models import load_model

# Путь к файлу модели H5
model_path = 'action.h5'

# Загрузка модели
model = load_model(model_path)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    # Конвертируем изображение из цветовой модели BGR в RGB,
    # так как Mediapipe работает с изображениями в формате RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Устанавливаем флаг, указывающий, что изображение неизменяемо,
    # чтобы предотвратить его модификацию во время обработки.
    image.flags.writeable = False

    # Обрабатываем изображение с помощью модели Mediapipe,
    # сохраняем результаты обработки в переменной results.
    results = model.process(image)

    # Возвращаем флаг изображения в исходное состояние,
    # разрешая его изменение после обработки.
    image.flags.writeable = True

    # После обработки возвращаем изображение в исходный формат BGR
    # для совместимости с OpenCV.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Возвращаем обработанное изображение и результаты.
    return image, results

def draw_landmarks(image, results):

    # Рисование точек и соединений
    if results.multi_hand_landmarks:  # Проверка на обнаружение хотя бы одной руки
        for handLms in results.multi_hand_landmarks:  # Перебор всех обнаруженных рук
            for id, lm in enumerate(handLms.landmark):  # Перебор всех точек (landmark) на руке
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # Рассчет координат точек в пикселях

                # Рисование точек и соединений на изображении
                mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
                
def extract_keypoints(results):
    left_hand_points = np.zeros(21 * 3)  # Массив для точек левой руки
    right_hand_points = np.zeros(21 * 3)  # Массив для точек правой руки

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Итерация по всем рукам в многорукой детекции
            hand_type = "Left Hand" if hand_landmarks.landmark[0].x < hand_landmarks.landmark[20].x else "Right Hand"
            hand_points = []

            for res in hand_landmarks.landmark:
                # Итерация по всем ключевым точкам текущей руки
                test = np.array([res.x, res.y, res.z])  # Создание массива с координатами и видимостью точки
                hand_points.append(test)

            # Сохранение координат в соответствующий массив
            if hand_type == "Left Hand":
                left_hand_points = np.array(hand_points).flatten() if hand_points else np.zeros(21 * 3)
            else:
                right_hand_points = np.array(hand_points).flatten() if hand_points else np.zeros(21 * 3)

    return np.concatenate([left_hand_points, right_hand_points])

colors = [(245,117,16), (117,245,16), (16,117,245)]

def prob_viz(res, actions, input_frame, colors):
    
    # Копируем входной кадр для создания выходного кадра, на котором будут отображаться вероятности действий.
    output_frame = input_frame.copy()
    
    # Итерируемся по каждому элементу res, который представляет вероятности для каждого действия.
    for num, prob in enumerate(res):
        # Создаем прямоугольник, который будет отображать вероятность действия на выходном кадре.
        # Прямоугольник начинается с (0,60+num*40) и заканчивается в (int(prob*100), 90+num*40).
        # Пропорции прямоугольника зависят от вероятности действия, умноженной на 100.
        # Цвет прямоугольника берется из списка colors.
        # Если вероятность действия равна 1, прямоугольник будет заполнен полностью.
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        
        # Добавляем текст метки действия рядом с прямоугольником на выходном кадре.
        # Текст начинается с (0, 85+num*40), используется шрифт cv2.FONT_HERSHEY_SIMPLEX с размером 1.
        # Цвет текста - (255,255,255) - белый.
        # Толщина текста - 2 пикселя.
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# Создаем массив строк с различными действиями, которые будут распознаваться моделью.
actions = np.array(['hello', 'thanks', 'iloveyou'])
sequence = []
sentence = []
predictions = []
threshold = 0.5


# Создаем контекстный менеджер для видеопотока
cap = cv2.VideoCapture(0)

# Блок Streamlit
st.title('Hand Gesture Recognition')
st.markdown('Use your hand gestures to form sentences!')

with mp_hands.Hands(max_num_hands=2) as holistic:
    # Начинаем бесконечный цикл обработки кадров
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Обнаруживаем ключевые точки на кадре
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            
        if np.argmax(res) in np.unique(predictions[-10:]):
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
                    
        if len(sentence) > 5:
            sentence = sentence[-5:]
        
        image = prob_viz(res, actions, image, colors)
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Отображаем кадр с помощью Streamlit
        st.image(image, channels="RGB", use_column_width=True)
        
        # Выход из цикла при нажатии клавиши ESC
        if cv2.waitKey(10) & 0xFF==27:
            break

cap.release()
cv2.destroyAllWindows()