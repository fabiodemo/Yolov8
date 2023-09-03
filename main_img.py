from ultralytics import YOLO
import cv2

# Load yolov8 model
model = YOLO('yolov8n.pt')

# Load video
image_path = "./image1.jpg"
image = cv2.imread(image_path)

# Detect objects
# Track objects
results = model.track(image, persist=True)

if image is not None:
        # Plot results
        frame_ = results[0].plot()

        # Visualize
        cv2.imshow("image", frame_)
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()