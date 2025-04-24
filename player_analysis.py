from ultralytics import YOLO
import cv2

def analyze_player_movements(video_path):
    model = YOLO("runs/detect/football_yolo_train/weights/best.pt")
    results = model.predict(source=video_path, save=False, stream=True)

    for frame_idx, result in enumerate(results):
        boxes = result.boxes.xyxy.cpu().numpy()
        print(f"\n🔍 Frame {frame_idx + 1}: {len(boxes)} player(s) detected.")
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            print(f"  Player {i + 1} Center: ({center_x:.2f}, {center_y:.2f})")

# مثال على استخدام الوظيفة
analyze_player_movements("yolo_dataset/football_video.mp4")