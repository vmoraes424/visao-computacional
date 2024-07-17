import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img  # Updated imports
import numpy as np
import cv2
import mediapipe as mp

# Parâmetros globais
img_width, img_height = 64, 64
model_path = './modelo_gestos.keras'  # Caminho do modelo salvo

# Lista os subdiretórios no diretório de treinamento
gestures = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'J', 
'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z']
print("Gestures:", gestures)

# Carregar o modelo treinado
model = tf.keras.models.load_model(model_path)

# Inicializar MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def predict_gesture(img, model):
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return predicted_class

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Converter a imagem para RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)

        # Desenhar as anotações da mão na imagem
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Capturar os pontos dos landmarks
                points = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in hand_landmarks.landmark], dtype=np.float32)

                # Verificar se há pelo menos um ponto antes de calcular o bounding rect
                if len(points) > 0:
                    bbox = cv2.boundingRect(points)
                    x, y, w, h = bbox
                    hand_img = frame[y:y+h, x:x+w]

                    # Verificar se hand_img não está vazia
                    if hand_img.size > 0:
                        hand_img = cv2.resize(hand_img, (img_width, img_height))
                        
                        # Prever o gesto usando o modelo
                        predicted_class = predict_gesture(hand_img, model)
                        
                        # Verificar se o índice da classe prevista está dentro do intervalo da lista gestures
                        if predicted_class < len(gestures):
                            gesture = gestures[predicted_class]
                            print(f'Gesto reconhecido: {gesture}')

                            # Adicionar o texto do gesto reconhecido na imagem
                            cv2.putText(image_bgr, gesture, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        else:
                            print(f'Classe prevista {predicted_class} fora do intervalo da lista de gestos.')
                    else:
                        print("Imagem da mão está vazia, ignorando este frame.")

        cv2.imshow('Reconhecimento de Mãos', image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
