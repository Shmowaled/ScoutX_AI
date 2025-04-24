from ultralytics import YOLO

model = YOLO("yolov8x.pt")

model.train(
    data="yolo_data.yaml",
    epochs=100,
    imgsz=1280,
    batch=16,
    name="football_yolo_train"
)