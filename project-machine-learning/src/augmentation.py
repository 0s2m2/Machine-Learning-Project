import os
import cv2
import numpy as np
from imgaug import augmenters as iaa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

base_input = os.path.join(PROJECT_ROOT, 'dataset_split', 'train')
base_output = os.path.join(PROJECT_ROOT, 'dataset_split', 'train_aug')


augmentation_pipeline = iaa.Sequential([
    iaa.Affine(
        rotate=(-15, 15),   
        scale=(0.9, 1.1)    
    ),
    iaa.Fliplr(0.5),        
    iaa.Multiply((0.8, 1.2)),  
    iaa.LinearContrast((0.8, 1.2))  
])

def augment_class(class_name, input_dir, output_dir, target_count):
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f'Input directory not found: {input_dir}')

    images = [img for img in os.listdir(input_dir) if img.lower().endswith(('.jpg','.jpeg','.png'))]
    valid_images = [img for img in images if cv2.imread(os.path.join(input_dir, img)) is not None]
    if not valid_images:
        raise RuntimeError(f'No valid images available in {input_dir}')
    current_count = len(valid_images)
    print(f"{class_name}: {current_count} → {target_count}")
    #copy original images into the augmented file
    for img_name in valid_images:
        img = cv2.imread(os.path.join(input_dir, img_name))
        if img is None:
            print(f'Skipping an empty image: {img_name}')
            continue
        cv2.imwrite(os.path.join(output_dir, img_name), img)
    
    #generate augmented images
    idx = 0
    max_attempts = target_count * 5
    attempts = 0
    while current_count < target_count and attempts < max_attempts:
        img_name = valid_images[idx% len(valid_images)]
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            idx+=1
            attempts+=1
            continue
        augmented_image = augmentation_pipeline(image = img)
        new_name = f'aug_{current_count:05d}.jpg'
        cv2.imwrite(os.path.join(output_dir, new_name), augmented_image)

        idx+=1
        current_count+=1
        attempts+=1
    print(f"{class_name} augmentation completed. Total images: {len(os.listdir(output_dir))}")


targets = {
    'cardboard': 500,
    'glass': 500,
    'metal': 500,
    'paper': 500,
    'plastic': 500,
    'trash': 500,
}

for cls, target in targets.items():
    print(os.path.join(base_output, cls))
    os.makedirs(os.path.join(base_output, cls), exist_ok=True)
    augment_class(cls, os.path.join(base_input, cls), os.path.join(base_output, cls), target)


