import os
import random
import shutil


DATASET_DIR = "dataset"
OUTPUT_DIR = "dataset_split"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

classes = ["cardboard", "glass",  "metal","paper", "plastic",  "trash"]

# Create train / val directories
for split in ["train", "val"]:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

for cls in classes:
    src_dir = os.path.join(DATASET_DIR, cls)
    images = os.listdir(src_dir)

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_index]
    val_images = images[split_index:]

    for img in train_images:
        shutil.copy(
            os.path.join(src_dir, img),
            os.path.join(OUTPUT_DIR, "train", cls, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(src_dir, img),
            os.path.join(OUTPUT_DIR, "val", cls, img)
        )

    print(f"{cls}: {len(train_images)} train | {len(val_images)} val")

print("\nDataset split completed successfully!")
