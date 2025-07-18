import json
import os
import random
from pathlib import Path

# === CONFIG ===
input_json = 'mmdetection/underwater_person_detection/annotations/instances_default.json'
output_dir = 'mmdetection/underwater_person_detection/annotations'
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
seed = 42

# === LOAD ORIGINAL JSON ===
with open(input_json, 'r') as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']
categories = coco['categories']

# === SPLIT IMAGES ===
random.seed(seed)
random.shuffle(images)

n_total = len(images)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)

train_images = images[:n_train]
val_images = images[n_train:n_train + n_val]
test_images = images[n_train + n_val:]

def get_ann_subset(images_subset):
    image_ids = set(img['id'] for img in images_subset)
    return [ann for ann in annotations if ann['image_id'] in image_ids]

splits = {
    'train': (train_images, get_ann_subset(train_images)),
    'val': (val_images, get_ann_subset(val_images)),
    'test': (test_images, get_ann_subset(test_images))
}

# === SAVE SPLIT FILES ===
os.makedirs(output_dir, exist_ok=True)

for split, (imgs, anns) in splits.items():
    out_path = os.path.join(output_dir, f'instances_{split}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'images': imgs,
            'annotations': anns,
            'categories': categories
        }, f)
    print(f'✅ Saved {split} split: {len(imgs)} images, {len(anns)} annotations → {out_path}')
