import os
import tensorflow as tf
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, Rescaling

def load_train_data(dataset_dir, CLASS_NAMES):
    train_data = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels = 'infered',
        label_mode = 'categorical',
        class_names = CLASS_NAMES,
        color_mode = 'rgb',
        batchsize = 64,
        image_size = (224, 224),
        shuffle = True,
        seed = 100,
        subset = 'training',
        validation_split = 0.2,
    )
    return train_data

def load_val_data(dataset_dir, CLASS_NAMES):
    val_data = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels = 'infered',
        label_mode = 'categorical',
        class_names = CLASS_NAMES,
        color_mode = 'rgb',
        batchsize = 64,
        image_size = (224, 224),
        shuffle = True,
        seed = 200,
        subset = 'validation',
        validation_split = 0.2,
    )
    return val_data

augment_layers = tf.keras.Sequential([
    RandomFlip(mode = 'horizontal'),
    RandomRotation(factor = (-0,2, 0.2)),
    RandomContrast(0.5),
    # Rescaling(1./225)
])

def augment_layers(image, label):
    augment_layers = tf.keras.Sequential([
    RandomFlip(mode = 'horizontal'),
    RandomRotation(factor = (-0,2, 0.2)),
    RandomContrast(0.5),
    # Rescaling(1./225)
    ])
    return augment_layers(image, training = True), label

def process_train_data(train_data):
    train_dataset = (
        train_data.map(augment_layers, num_parallel_calls = tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_dataset

def process_val_data(val_data):
    val_dataset = (
        val_data.map(augment_layers, num_parallel_calls = tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return val_dataset
