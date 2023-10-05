from ultralytics import YOLO, settings

# Update a setting
settings.update({'runs_dir': './stair_detection/experiments/runs'})

# Load a pretrained YOLO model (recommended for training)
model = YOLO('/home/nimrod/ros2_ws/src/stair_detection_pkg/stair_detection/models/yolov8/yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='/home/nimrod/ros2_ws/src/stair_detection_pkg/stair_detection/dataset/Myself.v2i.yolov8/data.yaml',
                    epochs=50,
                    batch=24,
                    pretrained=False,
                    device="cpu",
                    save=True)


# Evaluate the model's performance on the validation set
results = model.val()

