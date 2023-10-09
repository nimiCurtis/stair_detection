from ultralytics import YOLO, settings

# Update a setting
settings.update({'runs_dir': './stair_detection/experiments/runs'})

# Load a pretrained YOLO model (recommended for training)
model = YOLO('/home/nimrod/ros2_ws/src/stair_detection_pkg/stair_detection/models/yolov8/yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='/home/nimrod/ros2_ws/src/stair_detection_pkg/stair_detection/dataset/single_stair_detection.v3i.yolov8/data.yaml',
                    epochs=50,
                    batch=14,
                    pretrained=False,
                    device="cuda",
                    amp=False,
                    save=True)

# results = model.train(data='/home/nimrod/ros2_ws/src/stair_detection_pkg/stair_detection/dataset/Myself.v3i.yolov8/data.yaml',
#                     epochs=50,
#                     batch=14,
#                     pretrained=False,
#                     device="cuda",
#                     save=True)

# Evaluate the model's performance on the validation set
results = model.val()

