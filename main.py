import cv2
from deepface import DeepFace
import os

IMG_FOLDER = "respostas_visuais"
if not os.path.exists(IMG_FOLDER):
    os.makedirs(IMG_FOLDER)

cap = cv2.VideoCapture(0)

print("Iniciando... Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emocao = results[0]['dominant_emotion']
        
        cv2.putText(frame, f"Detectado: {emocao}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        caminho_img = os.path.join(IMG_FOLDER, f"{emocao}.jpg")
        if os.path.exists(caminho_img):
            reacao = cv2.imread(caminho_img)
            cv2.imshow("Sua Reacao Visual", reacao)

    except Exception as e:
        pass

    cv2.imshow('Camera - Detector de Emocoes', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

