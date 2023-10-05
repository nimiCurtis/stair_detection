from ultralytics import YOLO

# Load a model
model = YOLO('/home/nimrod/ros2_ws/src/stair_detection_pkg/stair_detection/models/yolov8/yolov8n.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['/home/nimrod/ros2_ws/src/stair_detection_pkg/stair_detection/dataset/Myself.v2i.yolov8/train/images/frame000000_jpg.rf.989012bb48fa7e1dafeecdc35ac06202.jpg'])  # return a list of Results objects