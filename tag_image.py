import cv2
import os
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out_dir = "results/detected_frames"
os.makedirs(out_dir, exist_ok=True)

# Output text file for frames and coordinates
txt_file_path = os.path.join(out_dir, "detections.txt")
with open(txt_file_path, "w") as f:
    f.write("frame_num,x1,y1\n")  # header

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # Run detection
        results = model.predict(frame, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Staff: {x1},{y1}"
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

                # Write coordinates directly
                f.write(f"{frame_num},{x1},{y1}\n")

            save_path = os.path.join(out_dir, f"frame_{frame_num:04d}.png")
            cv2.imwrite(save_path, frame)

        # Progress print
        if frame_num % 30 == 0 or frame_num == total_frames:
            progress = (frame_num / total_frames) * 100
            print(f"Processing: {frame_num}/{total_frames} frames ({progress:.1f}%)")

cap.release()
print("All detections saved to", out_dir)
print("Detection coordinates saved to", txt_file_path)
