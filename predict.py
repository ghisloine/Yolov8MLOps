from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("mymodels/best.pt", task="detect")
# Uncomment to load your best.pt or best.engine model
# model = YOLO("path/to/best.pt", task="detect")

# Run inference on an image
results = model("ultralytics/assets/bus.jpg")  # list of 1 Results object
