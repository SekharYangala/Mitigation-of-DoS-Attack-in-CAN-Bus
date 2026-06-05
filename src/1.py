# =========================
# 1. INSTALL LIBRARIES
# =========================
pip install tensorflow opencv-python pandas pillow scikit-learn

# =========================
# 2. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# =========================
# 3. UPLOAD DATASET
# =========================
from google.colab import files
uploaded = files.upload()

# =========================
# 4. LOAD DATASET (NO HEADERS)
# =========================
columns = [
    "Timestamp","CAN_ID","DLC",
    "D0","D1","D2","D3","D4","D5","D6","D7",
    "Class"
]

df = pd.read_csv("DoS_dataset.csv", names=columns)

# =========================
# 5. PREPROCESSING
# =========================

# HEX → DECIMAL
df['CAN_ID'] = df['CAN_ID'].apply(lambda x: int(str(x),16))

for col in ['D0','D1','D2','D3','D4','D5','D6','D7']:
    df[col] = df[col].apply(lambda x: int(str(x),16))

# Handle scientific notation
df['CAN_ID'] = pd.to_numeric(df['CAN_ID'], errors='coerce')

# Fill missing values
df = df.fillna(0)

# Convert labels
df['Class'] = df['Class'].replace({'R':0, 'T':1})

# Ensure numeric
feature_cols = ['CAN_ID','DLC','D0','D1','D2','D3','D4','D5','D6','D7']

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(0)

# =========================
# 6. FEATURE EXTRACTION
# =========================
features = df[feature_cols].values.astype(np.float32)
labels = df['Class'].values

# One-hot encoding (for softmax)
labels = to_categorical(labels, num_classes=2)

# =========================
# 7. CAN DATA → IMAGE
# =========================
image_dir = "/content/can_images"
os.makedirs(image_dir, exist_ok=True)

for i in range(len(features)-81):

    patch = features[i:i+81].flatten()

    if len(patch) < 243:
        continue

    img = np.array(patch[:243], dtype=np.uint8).reshape(9,9,3)

    img = cv2.resize(img,(224,224))

    label = np.argmax(labels[i])

    class_dir = os.path.join(image_dir,str(label))
    os.makedirs(class_dir, exist_ok=True)

    Image.fromarray(img).save(f"{class_dir}/{i}.png")

# =========================
# 8. LOAD IMAGE DATASET
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    image_dir,
    target_size=(224,224),
    batch_size=128,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    image_dir,
    target_size=(224,224),
    batch_size=128,
    class_mode='categorical',
    subset='validation'
)

# =========================
# 9. LOAD VGG16
# =========================
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# =========================
# 10. ADD CLASSIFIER
# =========================
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# 11. COMPILE
# =========================
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# 12. TRAIN
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100
)

# =========================
# 13. EVALUATE
# =========================
loss, acc = model.evaluate(val_data)
print("Final Accuracy:", acc)

# =========================
# 14. SAVE MODEL
# =========================
model.save("vgg16_dos_model.h5")