# Staff Tag Detection 
This project detects staff tags in video using YOLO.

### 1. `frame_extraction.py`
Extracts frames from an input video (`sample.mp4`) at approximately 6 frames per second and saves them in `dataset/images/` for data labelling.

### 2. `unlabelled_frame.py`
Creates empty label files for frames that do not contain staff tags, ensuring every frame has a corresponding `.txt` label file for dataset consistency.

### 3. `train_val_split.py`
Randomly splits the dataset into training (80%) and validation (20%) sets.
Images and labels are organized into separate folders under `dataset_split/`:

```
dataset_split/
    images/train
    images/val
    labels/train
    labels/val
```

### 4. `train.py`
Trains the YOLO model (`yolo11n.pt`) using the dataset.

### 5. `tag_image.py`
Processes the input video to produce:

* Annotated frames saved in `results/detected_frames/`.
* A text file (`detections.txt`) recording frame number and tag coordinates in the format: `frame_num,x1,y1`.

### 6. `tag_video.py`
Performs staff tag detection on the input video and produces a video with annotated detections (`results/output.mp4`).
