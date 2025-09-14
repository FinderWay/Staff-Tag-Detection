import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture("sample.mp4")

# Get video info
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "results/output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    results = model.predict(frame, conf=0.5, verbose=False)
    annotated = results[0].plot()
    out.write(annotated)

    # Progress print (every 30 frames or last frame)
    if frame_num % 30 == 0 or frame_num == total_frames:
        progress = (frame_num / total_frames) * 100
        print(f"Processing: {frame_num}/{total_frames} frames ({progress:.1f}%)")

cap.release()
out.release()
print("Saved detections to output.mp4")
