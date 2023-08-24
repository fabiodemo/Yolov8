from ultralytics import YOLO
import cv2

# Load yolov8 model
model = YOLO('yolov8n.pt')

# Load video
video_path = "./demo.mp4"
cap = cv2.VideoCapture(video_path)

ret = True
# Read frames
while ret:
    ret, frame = cap.read()
    if ret:
        # Detect objects
        # Track objects
        results = model.track(frame, persist=True)

        # Plot results
        frame_ = results[0].plot()
        # Alternatives to results[0].plot()
        # cv2.recatangle
        # cv2.putText

        # Visualize
        cv2.imshow("frame", frame_)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
