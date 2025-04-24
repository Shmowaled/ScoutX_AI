from ultralytics import YOLO
import cv2

model = YOLO("C:/Users/SHAHAD/OneDrive/Desktop/ScoutX/sports-ai-env/runs/detect/train/weights/best.pt")

video_path = "C:/Users/SHAHAD/OneDrive/Desktop/ScoutX/sports-ai-env/yolo_dataset/football_video.mp4"

results = model.predict(source=video_path, save=True, conf=0.25)

from player_analysis import analyze_player_movements
analyze_player_movements(results)
