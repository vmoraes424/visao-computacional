import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Inicializar o módulo de mãos do MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

# Carregar o modelo de reconhecimento de gestos
# Você precisará treinar um modelo e carregá-lo aqui
# Por exemplo: model = tf.keras.models.load_model('modelo_gestos.h5')

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Modificado para rastrear até 2 mãos
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
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extraia os pontos de referência (landmarks) da mão
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])

                # Aqui você pode passar os landmarks para o seu modelo para reconhecer o gesto
                # Exemplo: gesture = model.predict(np.array([landmarks]).reshape(1, -1))

        cv2.imshow('Reconhecimento de Mãos', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
