from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from YAML
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(
    data="coco128.yaml",
    epochs=10,
    imgsz=736,
    project="mlops",
    name="IE-1002",
    device="cuda",
)
