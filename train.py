from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=960,
    batch=8,
    patience=15,
    device=0,
    fliplr=0.5,
    flipud=0.2,
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    scale=0.5,
    translate=0.1
)
