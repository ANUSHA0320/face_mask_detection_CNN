import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = 128

def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bbox = root.find('object').find('bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    return [xmin, ymin, xmax, ymax]

def load_data(img_dir, ann_dir):
    X, y = [], []
    for fname in os.listdir(img_dir):
        if fname.endswith('.jpg') or fname.endswith('.png'):
            img_path = os.path.join(img_dir, fname)
            xml_path = os.path.join(ann_dir, os.path.splitext(fname)[0] + '.xml')
            if not os.path.exists(xml_path):
                continue
            img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
            X.append(np.array(img) / 255.0)
            bbox = parse_voc_xml(xml_path)
            # Normalize bbox to [0,1] relative to image size
            bbox = [b / img.size[0] if i % 2 == 0 else b / img.size[1] for i, b in enumerate(bbox)]
            y.append(bbox)
    return np.array(X), np.array(y)

X, y = load_data('archive/images', 'archive/annotations')

# Simple CNN for bounding box regression
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='sigmoid')  # 4 bbox coords
])

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=10, batch_size=8, validation_split=0.2)

model.save('mask_bbox_model.h5')