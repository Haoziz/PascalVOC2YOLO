import os
import json
import random
import shutil
import numpy as np
import xml.etree.ElementTree as ET


def voc2coco(ann_dir, img_dir, out_file, val_split):
    # Initialize COCO dataset dictionary
    coco = {}
    coco['images'] = []
    coco['annotations'] = []
    coco['categories'] = []

    # Define categories
    categories = {"aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3,
                  "bottle": 4, "bus": 5, "car": 6, "cat": 7, "chair": 8,
                  "cow": 9, "diningtable": 10, "dog": 11, "horse": 12,
                  "motorbike": 13, "person": 14, "pottedplant": 15,
                  "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19}  # Add your own categories here
    for cat_name, cat_id in categories.items():
        coco['categories'].append({'id': cat_id, 'name': cat_name})

    # Load annotations
    ann_id = 0
    for ann_file in os.listdir(ann_dir):
        tree = ET.parse(os.path.join(ann_dir, ann_file))
        root = tree.getroot()

        # Add image information to COCO dataset
        img_name = root.find('filename').text
        img_id = int(os.path.splitext(img_name)[0])
        img_path = os.path.join(img_dir, img_name)
        img = {'id': img_id, 'file_name': img_name, 'path': img_path}
        coco['images'].append(img)

        # Add object annotations to COCO dataset
        size = root.find('size')
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            x = np.float32(x / float(size.find('width').text))
            y = np.float32(y / float(size.find('height').text))
            w = np.float32(w / float(size.find('width').text))
            h = np.float32(h / float(size.find('height').text))
            ann = {'id': ann_id, 'image_id': img_id, 'category_id': categories[obj.find('name').text],
                   'bbox': [x, y, w, h]}
            coco['annotations'].append(ann)
            ann_id += 1

    # Split dataset into train and validation sets
    random.shuffle(coco['images'])
    val_size = int(len(coco['images']) * val_split)
    train_images = coco['images'][val_size:]
    val_images = coco['images'][:val_size]

    # Write COCO dataset to json file
    train_coco = {'images': train_images, 'annotations': coco['annotations'], 'categories': coco['categories']}
    val_coco = {'images': val_images, 'annotations': coco['annotations'], 'categories': coco['categories']}
    with open(out_file + 'train.json', 'w') as f:
        json.dump(train_coco, f)
    with open(out_file + 'val.json', 'w') as f:
        json.dump(val_coco, f)

    # Write txt file
    with open(out_file + 'train.txt', 'w') as f:
        for img in train_images:
            f.write(img['path'] + '\n')
    with open(out_file + 'val.txt', 'w') as f:
        for img in val_images:
            f.write(img['path'] + '\n')

    # For each picture in the training set and verification set, generate the corresponding txt file
    for img in train_images:
        img_id = img['id']
        with open('D:\\PythonProject\\datasets\\VOCdevkit\\VOC2012\\labels\\train\\' + os.path.splitext(img['file_name'])[0] + '.txt', 'w') as f:
            for ann in coco['annotations']:
                if ann['image_id'] == img_id:
                    f.write(str(ann['category_id']) + ' ' + ' '.join([str(i) for i in ann['bbox']]) + '\n')
    for img in val_images:
        img_id = img['id']
        with open('D:\\PythonProject\\datasets\\VOCdevkit\\VOC2012\\labels\\val\\' + os.path.splitext(img['file_name'])[0] + '.txt', 'w') as f:
            for ann in coco['annotations']:
                if ann['image_id'] == img_id:
                    f.write(str(ann['category_id']) + ' ' + ' '.join([str(i) for i in ann['bbox']]) + '\n')


def move_images(train_txt, val_txt, out_dir1, out_dir2):
    # Read the txt files that specify which images belong to the training and testing sets
    with open(train_txt, 'r') as f:
        train_images = f.readlines()
    train_images = [x.strip() for x in train_images]

    with open(val_txt, 'r') as f:
        test_images = f.readlines()
    test_images = [x.strip() for x in test_images]

    # Create the folders for the training and testing sets
    if not os.path.exists(out_dir1):
        os.mkdir(out_dir1)
    if not os.path.exists(out_dir2):
        os.mkdir(out_dir2)

    # Move the images to the corresponding folders based on the txt files
    for image in train_images:
        shutil.copy2(image, out_dir1)
    for image in test_images:
        shutil.copy2(image, out_dir2)


if __name__ == "__main__":
    voc2coco('D:\\PythonProject\\datasets\\VOCdevkit\\VOC2012\\Annotations\\',
             'D:\\PythonProject\\datasets\\VOCdevkit\\VOC2012\\JPEGImages\\',
             'D:\\PythonProject\\datasets\\VOCdevkit\\VOC2012\\',
             0.2)
    move_images('D:\\PythonProject\\datasets\\VOCdevkit\\VOC2012\\VOC2012\\train.txt',
                'D:\\PythonProject\\datasets\\VOCdevkit\\VOC2012\\VOC2012\\val.txt',
                'D:\\PythonProject\\datasets\\VOCdevkit\\VOC2012\\images\\train',
                'D:\\PythonProject\\datasets\\VOCdevkit\\VOC2012\\images\\test')
