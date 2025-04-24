from ultralytics import YOLO

model = YOLO("yolov8x.pt") 

results = model.predict(
    source="yolo_dataset/images/val", 
    save=True,
    save_txt=True,
    conf=0.5
)
