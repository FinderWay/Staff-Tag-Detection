import cv2, os

cap = cv2.VideoCapture("sample.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps // 6)  # sample ~6 frames per second

os.makedirs("dataset/images", exist_ok=True)

count, saved = 0, 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_interval == 0:
        filename = f"dataset/images/frame_{saved:04d}.jpg"
        cv2.imwrite(filename, frame)
        saved += 1
    count += 1

cap.release()
print(f"Saved {saved} frames.")