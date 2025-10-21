# Mejorado para incluir suavizado de movimiento y guardado controlado e inteligente
#Guarda solo cuando cambias de posición, te acercas o alejas

import cv2
import mediapipe as mp
import time
import os
import math

class FaceDetector:
    def __init__(self, minDetection=0.5, smooth_factor=0.7,
                 save_faces=False, save_dir="captures",
                 save_interval=2.0, movement_threshold=0.15):
        """
        Detección facial con suavizado, guardado controlado y detección de movimiento.

        Args:
            minDetection (float): confianza mínima de detección (0.0 - 1.0)
            smooth_factor (float): suavizado del movimiento (0 = sin suavizado)
            save_faces (bool): si True, guarda los rostros detectados
            save_dir (str): carpeta donde se guardan las imágenes
            save_interval (float): tiempo mínimo entre capturas (segundos)
            movement_threshold (float): umbral de cambio (0.0 - 1.0) para considerar movimiento
        """
        self.minDetection = minDetection
        self.smooth_factor = smooth_factor
        self.save_faces = save_faces
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.movement_threshold = movement_threshold

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetection)

        self.last_bbox = None
        self.last_save_time = 0

        if save_faces:
            os.makedirs(self.save_dir, exist_ok=True)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)

                # Suavizado de movimiento
                if self.last_bbox is not None:
                    bbox = (
                        int(self.last_bbox[0] * self.smooth_factor + bbox[0] * (1 - self.smooth_factor)),
                        int(self.last_bbox[1] * self.smooth_factor + bbox[1] * (1 - self.smooth_factor)),
                        int(self.last_bbox[2] * self.smooth_factor + bbox[2] * (1 - self.smooth_factor)),
                        int(self.last_bbox[3] * self.smooth_factor + bbox[3] * (1 - self.smooth_factor)),
                    )

                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    conf = int(detection.score[0] * 100)
                    cv2.putText(img, f'{conf}%', (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Guardado controlado e inteligente
                if self.save_faces and self.shouldSaveFace(bbox):
                    self.saveFace(img, bbox)

                self.last_bbox = bbox  # Actualizamos el último cuadro

        return img, bboxs

    def fancyDraw(self, img, bbox, l=25, t=4, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (0, 255, 0), rt)
        color = (0, 255, 0)
        for (x0, y0, x2, y2) in [
            (x, y, x + l, y), (x, y, x, y + l),
            (x1, y, x1 - l, y), (x1, y, x1, y + l),
            (x, y1, x + l, y1), (x, y1, x, y1 - l),
            (x1, y1, x1 - l, y1), (x1, y1, x1, y1 - l)
        ]:
            cv2.line(img, (x0, y0), (x2, y2), color, t)
        return img

    def shouldSaveFace(self, bbox):
        """Determina si debe guardarse el rostro según tiempo y movimiento."""
        current_time = time.time()
        if current_time - self.last_save_time < self.save_interval:
            return False  # Aún no pasó el tiempo mínimo

        if self.last_bbox is None:
            return True  # Primer guardado

        x, y, w, h = bbox
        x0, y0, w0, h0 = self.last_bbox

        # Distancia euclidiana entre centros
        dx = (x + w/2) - (x0 + w0/2)
        dy = (y + h/2) - (y0 + h0/2)
        dist = math.sqrt(dx**2 + dy**2)

        # Diferencia relativa de área
        area = w * h
        prev_area = w0 * h0
        area_diff = abs(area - prev_area) / prev_area if prev_area != 0 else 0

        # Normalización por ancho de imagen (~ umbral adaptable)
        movement = dist / max(w0, 1)

        # Si hay suficiente movimiento o cambio de tamaño
        if movement > self.movement_threshold or area_diff > self.movement_threshold:
            self.last_save_time = current_time
            return True
        return False

    def saveFace(self, img, bbox):
        """Guarda el rostro recortado."""
        x, y, w, h = bbox
        face = img[y:y+h, x:x+w]
        if face.size > 0:
            filename = f"{self.save_dir}/face_{int(time.time())}.jpg"
            cv2.imwrite(filename, face)
            print(f"💾 Guardado: {filename}")

# ---------- MAIN ----------
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0

    detector = FaceDetector(
        minDetection=0.6,
        smooth_factor=0.7,
        save_faces=True,
        save_interval=2,
        movement_threshold=0.2  # Cambia según qué tan sensible quieras
    )

    while True:
        success, img = cap.read()
        if not success:
            print("⚠️ No se pudo acceder a la cámara.")
            break

        img, bboxs = detector.findFaces(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime + 1e-6)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Face Detection", img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
