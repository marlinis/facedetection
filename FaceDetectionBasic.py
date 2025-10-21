import cv2
import mediapipe as mp
import time

# Inicialización
cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(model_selection=0, min_detection_confidence=0.75)

bbox_prev = None
smooth_factor = 0.7  # suavizado del movimiento del recuadro

while True:
    success, img = cap.read()
    if not success:
        print(" No se pudo leer el frame de la cámara.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            # suavizado de movimiento
            if bbox_prev is not None:
                bbox = (
                    int(bbox_prev[0] * smooth_factor + bbox[0] * (1 - smooth_factor)),
                    int(bbox_prev[1] * smooth_factor + bbox[1] * (1 - smooth_factor)),
                    int(bbox_prev[2] * smooth_factor + bbox[2] * (1 - smooth_factor)),
                    int(bbox_prev[3] * smooth_factor + bbox[3] * (1 - smooth_factor)),
                )
            bbox_prev = bbox

            cv2.rectangle(img, bbox, (0, 255, 0), 3)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # FPS
    cTime = time.perf_counter()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

    cv2.imshow("Face Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
