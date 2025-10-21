# Se mejora la forma de guardado de la detección de rostros faciales

import cv2
import mediapipe as mp
import time
import os

class FaceDetector:
    def __init__(self, minDetection=0.5, smooth_factor=0.7,
                 save_faces=False, save_dir="captures", save_interval=2.0):
        """
        Detección facial con suavizado y guardado controlado.

        Args:
            minDetection (float): confianza mínima de detección (0.0 - 1.0)
            smooth_factor (float): suavizado (0 = sin suavizado, 1 = muy suave)
            save_faces (bool): si True, guarda los rostros detectados
            save_dir (str): carpeta donde se guardan las imágenes
            save_interval (float): tiempo mínimo entre capturas (segundos)
        """
        self.minDetection = minDetection
        self.smooth_factor = smooth_factor
        self.save_faces = save_faces
        self.save_dir = save_dir
        self.save_interval = save_interval

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetection)

        self.last_bbox = None
        self.last_save_time = 0  # Control de tiempo de guardado

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

                # Suavizado del movimiento
                if self.last_bbox is not None:
                    bbox = (
                        int(self.last_bbox[0] * self.smooth_factor + bbox[0] * (1 - self.smooth_factor)),
                        int(self.last_bbox[1] * self.smooth_factor + bbox[1] * (1 - self.smooth_factor)),
                        int(self.last_bbox[2] * self.smooth_factor + bbox[2] * (1 - self.smooth_factor)),
                        int(self.last_bbox[3] * self.smooth_factor + bbox[3] * (1 - self.smooth_factor)),
                    )
                self.last_bbox = bbox

                bboxs.append([id, bbox, detection.score])

                if draw:
                    img = self.fancyDraw(img, bbox)
                    conf = int(detection.score[0] * 100)
                    cv2.putText(img, f'{conf}%', (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Guardado controlado
                if self.save_faces:
                    self.saveFace(img, bbox)

        return img, bboxs

    def fancyDraw(self, img, bbox, l=25, t=4, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (0, 255, 0), rt)
        color = (0, 255, 0)
        cv2.line(img, (x, y), (x + l, y), color, t)
        cv2.line(img, (x, y), (x, y + l), color, t)
        cv2.line(img, (x1, y), (x1 - l, y), color, t)
        cv2.line(img, (x1, y), (x1, y + l), color, t)
        cv2.line(img, (x, y1), (x + l, y1), color, t)
        cv2.line(img, (x, y1), (x, y1 - l), color, t)
        cv2.line(img, (x1, y1), (x1 - l, y1), color, t)
        cv2.line(img, (x1, y1), (x1, y1 - l), color, t)
        return img

    def saveFace(self, img, bbox):
        """Guarda el rostro solo si pasó suficiente tiempo desde el último guardado."""
        current_time = time.time()
        if current_time - self.last_save_time < self.save_interval:
            return  # ⏱ No guardar todavía

        self.last_save_time = current_time
        x, y, w, h = bbox
        face = img[y:y+h, x:x+w]
        if face.size > 0:
            filename = f"{self.save_dir}/face_{int(time.time())}.jpg"
            cv2.imwrite(filename, face)
            print(f"💾 Imagen guardada: {filename}")

# ---------- MAIN ----------
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector(minDetection=0.6, smooth_factor=0.7, save_faces=True, save_interval=2)

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
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
