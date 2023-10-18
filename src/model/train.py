from ultralytics import YOLO, settings
import torch
torch.cuda.empty_cache()


# Update a setting
settings.update({'runs_dir': './experiments/runs'})

# Load a pretrained YOLO model (recommended for training)
model = YOLO('/home/nimrod/ros2_ws/src/stair_detection_pkg/stair_detection/models/yolov8/yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='/home/nimrod/ros2_ws/src/stair_detection_pkg/stair_detection/dataset/single_stair_detection.v6i.yolov8/data.yaml',
                    epochs=200,
                    imgsz=320,
                    batch=24,
                    cos_lr=False,
                    close_mosaic=100,
                    patience=100,
                    pretrained=False,
                    device='cuda',
                    amp=False,
                    save=True)


# Evaluate the model's performance on the validation set
results = model.val()

