import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, minDetection=0.5):
        self.minDetection = minDetection
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetection)

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
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=6, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (0, 255, 0), rt)

        # Esquinas
        for (x_start, y_start) in [(x, y), (x1, y), (x, y1), (x1, y1)]:
            x_dir = -l if x_start == x1 else l
            y_dir = -l if y_start == y1 else l
            cv2.line(img, (x_start, y_start),
                     (x_start - x_dir, y_start), (0, 255, 0), t)
            cv2.line(img, (x_start, y_start),
                     (x_start, y_start - y_dir), (0, 255, 0), t)
        return img

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector(0.7)

    while True:
        success, img = cap.read()
        if not success:
            break

        img, bboxs = detector.findFaces(img)
        print(bboxs)

        cTime = time.time()
        fps = 1 / (cTime - pTime + 1e-6)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

        cv2.imshow("Face Detection", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
