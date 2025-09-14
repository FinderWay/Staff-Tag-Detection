import os, random, shutil

# Paths
img_dir = "dataset/images"
lbl_dir = "dataset/labels"
out_dir = "dataset_split"

# Make train/val folders
for split in ["train", "val"]:
    os.makedirs(f"{out_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{out_dir}/labels/{split}", exist_ok=True)

# Split
images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
random.shuffle(images)
split_idx = int(0.8 * len(images))

train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def move_files(img_list, split):
    for img in img_list:
        lbl = img.replace(".jpg", ".txt")
        shutil.copy(os.path.join(img_dir, img), f"{out_dir}/images/{split}/{img}")
        shutil.copy(os.path.join(lbl_dir, lbl), f"{out_dir}/labels/{split}/{lbl}")

move_files(train_imgs, "train")
move_files(val_imgs, "val")
