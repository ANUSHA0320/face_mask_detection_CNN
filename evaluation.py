import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
            bbox = [b / img.size[0] if i % 2 == 0 else b / img.size[1] for i, b in enumerate(bbox)]
            y.append(bbox)
    return np.array(X), np.array(y)

# Load your test data (or use all data for a quick check)
X, y = load_data('archive/images', 'archive/annotations')

# Load the model
model = tf.keras.models.load_model('mask_bbox_model.h5', compile=False)
model.compile(optimizer='adam', loss='mse')

# Evaluate
loss = model.evaluate(X, y)
print(f"Test MSE Loss: {loss:.4f}")

# Prepare the validation/test data generator
base_dir = 'archive/images'  # Same as before

test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

test_generator = test_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")