from ultralytics import YOLO
import json

# Load a model
model = YOLO("yolov8n.pt")  # load an official model
# Uncomment to load your best.pt model
# model = YOLO("path/to/best.pt")  # load a custom model

# Validate the model
metrics = model.val(
    save_json=True
)  # no arguments needed, dataset and settings remembered
print(metrics.box.map)  # map50-95
print(metrics.box.map50)  # map50
print(metrics.box.map75)  # map75
print(metrics.box.maps)  # a list contains map50-95 of each category


with open("myresults/metrics_engine.json", "w") as fp:
    data = {
        "map": metrics.box.map,
        "map50": metrics.box.map50,
        "map75": metrics.box.map75,
    }
    json.dump(data, fp)
