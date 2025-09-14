import os

# Paths
img_dir = "dataset/images"
lbl_dir = "dataset/labels"

os.makedirs(lbl_dir, exist_ok=True)

# Get all images
images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

for img in images:
    txt_name = img.replace(".jpg", ".txt")
    txt_path = os.path.join(lbl_dir, txt_name)

    # If no label file exists, create an empty one
    if not os.path.exists(txt_path):
        open(txt_path, 'w').close()

print("Empty label files created for all unlabeled frames.")
