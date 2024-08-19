from ultralytics import YOLO


model = YOLO("yolov8x")
model.predict("input_video/08fd33_4.mp4", save=True)