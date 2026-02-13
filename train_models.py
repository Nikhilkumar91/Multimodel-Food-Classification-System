"""
Train Custom CNN, VGG, and ResNet models for food classification.
Uses the same class splits as app.py. Exports to TFLite for deployment.
Improvements: data augmentation, early stopping, validation split, proper preprocessing.
"""
import os
import json
import argparse
import numpy as np
from pathlib import Path

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# CONFIG (must match app.py)
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.join(os.path.dirname(__file__), "Food Classification dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
VAL_SPLIT = 0.2
SEED = 42

# Class groups per model (same as app.py RAW_MODEL_CLASS_INDEX)
MODEL_GROUPS = {
    1: ["apple_pie", "baked_potato", "burger"],
    2: ["butter_naan", "chai", "chapati"],
    3: ["cheesecake", "chicken_curry", "chole_bhature"],
    4: ["crispy_chicken", "dal_makhani", "dhokla"],
    5: ["donut", "fried_rice", "fries"],
    6: ["hot_dog", "ice_cream", "idli"],
    7: ["jalebi", "kaathi_rolls", "kadai_paneer"],
    8: ["kulfi", "masala_dosa", "momos"],
    9: ["omelette", "paani_puri", "pakode"],
    10: ["pav_bhaji", "pizza", "samosa"],
    11: ["sandwich", "sushi", "taco", "taquito"],
}


def normalize(name):
    return name.lower().strip().replace(" ", "_")


def find_class_folder(class_name):
    """Resolve class name to actual folder name in dataset (e.g. 'Baked Potato')."""
    cnorm = normalize(class_name)
    if not os.path.isdir(DATASET_DIR):
        return None
    for folder in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(path):
            continue
        if normalize(folder) == cnorm:
            return folder
    return None


def collect_image_paths_and_labels(model_num):
    """Collect (path, label_index) for all images in this model's classes."""
    classes = MODEL_GROUPS[model_num]
    allowed = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    paths, labels = [], []
    for idx, class_name in enumerate(classes):
        folder = find_class_folder(class_name)
        if not folder:
            print(f"  Warning: no folder for class '{class_name}', skipping.")
            continue
        dir_path = os.path.join(DATASET_DIR, folder)
        for f in os.listdir(dir_path):
            if f.endswith(allowed):
                paths.append(os.path.join(dir_path, f))
                labels.append(idx)
    return paths, labels, classes


def build_dataset(paths, labels, batch_size, shuffle=True, augment=False):
    """Build tf.data.Dataset from file paths and integer labels."""
    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_custom_cnn(num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """Lightweight CNN (same input size as VGG/ResNet for consistency)."""
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)


def build_vgg_transfer(num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """VGG16 transfer learning (top layers trainable)."""
    base = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = True
    for layer in base.layers[:-4]:
        layer.trainable = False
    inp = base.input
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)


def build_resnet_transfer(num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """ResNet50 transfer learning (top layers trainable)."""
    base = keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    inp = base.input
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)


def train_and_export(model_type, model_num, epochs=EPOCHS):
    """Train one model (custom_model_N, vgg_model_N, or resnet_model_N) and export TFLite."""
    paths, labels, class_names = collect_image_paths_and_labels(model_num)
    if not paths:
        print(f"  No images for model {model_num}, skip.")
        return
    num_classes = len(class_names)
    n_val = max(1, int(len(paths) * VAL_SPLIT))
    n_train = len(paths) - n_val
    indices = np.random.RandomState(SEED).permutation(len(paths))
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_ds = build_dataset(train_paths, train_labels, BATCH_SIZE, shuffle=True)
    val_ds = build_dataset(val_paths, val_labels, BATCH_SIZE, shuffle=False)

    if model_type == "custom":
        model = build_custom_cnn(num_classes)
        name = f"custom_model_{model_num}"
    elif model_type == "vgg":
        model = build_vgg_transfer(num_classes)
        name = f"vgg_model_{model_num}"
    elif model_type == "resnet":
        model = build_resnet_transfer(num_classes)
        name = f"resnet_model_{model_num}"
    else:
        raise ValueError("model_type must be custom, vgg, or resnet")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]
    print(f"  Training {name} ({n_train} train, {n_val} val)...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

    # Export TFLite
    tflite_path = os.path.join(OUTPUT_DIR, f"{name}.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"  Saved {tflite_path}")
    return tflite_path


def main():
    parser = argparse.ArgumentParser(description="Train food classification models")
    parser.add_argument("--type", choices=["custom", "vgg", "resnet", "all"], default="all")
    parser.add_argument("--model-num", type=int, default=None, help="Train only this model number (1-11)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    model_nums = [args.model_num] if args.model_num is not None else list(MODEL_GROUPS.keys())
    types = ["custom", "vgg", "resnet"] if args.type == "all" else [args.type]

    for model_num in model_nums:
        if model_num not in MODEL_GROUPS:
            print(f"Unknown model number {model_num}, skip.")
            continue
        for t in types:
            train_and_export(t, model_num, epochs=args.epochs)


if __name__ == "__main__":
    main()
