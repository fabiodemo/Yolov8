from ultralytics import YOLO
import cv2

model=YOLO('../YOLO-Weights/yolov8n.pt')
results=model('../images/image1.jpg',show=True)

cv2.waitKey(0)