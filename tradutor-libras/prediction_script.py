# prediction_script.py

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import mediapipe as mp

# Parâmetros globais
img_width, img_height = 64, 64
model_path = 'modelo_gestos.h5'  # Caminho do modelo salvo

# Carregar o modelo treinado
model = tf.keras.models.load_model(model_path)

# Inicializar MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Lista de gestos correspondentes às classes do modelo
gestures = ['gesto_1', 'gesto_2', 'gesto_3', ...]  # Substitua pelos nomes reais dos gestos

def predict_gesture(landmarks, model):
    landmarks = np.array(landmarks).flatten().reshape(1, -1) / 255.0
    predictions = model.predict(landmarks)
    predicted_class = np.argmax(predictions)
    return predicted_class

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Converter a imagem para RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # Desenhar as anotações da mão na imagem
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extrair os landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])

                # Prever o gesto usando o modelo
                predicted_class = predict_gesture(landmarks, model)
                gesture = gestures[predicted_class]
                print(f'Gesto reconhecido: {gesture}')

        cv2.imshow('Reconhecimento de Mãos', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
