from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

cap = cv2.VideoCapture(0)


def detect(frame):
    boxes, _ = mtcnn.detect(frame)
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    if isinstance(boxes, type(np.array([0]))):
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    return frame_draw


while True:
    ret, frame = cap.read()
    frame = Image.fromarray(frame)
    frame = detect(frame)
    cv2.imshow("face detection", np.array(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
