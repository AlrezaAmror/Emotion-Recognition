import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Dropout, 
                                    Flatten, Dense, GlobalAveragePooling2D,
                                    MaxPooling2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import L2, L1


def EffNetV2B0(image_shape, num_class):
    # get pretrained model
    backbone = tf.keras.applications.EfficientNetV2B0(
        input_shape = (image_shape, image_shape, 3),
        include_top = False,
        weigts = 'imagenet'
    )
    backbone.trainable = False
    # build models
    model = Sequential([
        backbone,
        Flatten(),
        Dropout(0.5),
        Dense(num_class, activation = 'softmax')
    ])
    return model

def MobNetV3Large(image_shape, num_class):
    # get pretrained
    backbone = tf.keras.applications.MobileNetV3Large(
        input_shape = (image_shape, image_shape, 3),
        include_top = False,
        weigts = 'imagenet'
    )
    backbone.trainable = False
    # build model
    model = Sequential([
        backbone,
        GlobalAveragePooling2D(),
        Dense(128, activation = 'relu'),
        Dropout(0.25),
        Dense(32, activation = 'relu'),
        Dropout(0.25),
        Dense(128, activation = 'relu'),
        BatchNormalization(),
        Dense(num_class, activation = 'softmax')
    ])
    return model

def MyModel(image_shape, num_class):
    model = Sequential([
        Conv2D(32, (3, 3), activation = 'relu', input_shape = (image_shape, image_shape, 3),
               kernel_regularizer = L2(0.01)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation = 'relu', kernel_regularizer = L2(0.01)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation = 'relu', kernel_regularizer = L2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(6, activation = 'softmax')

    ])
    return model

if __name__==__main__:
    input_shape = 224
    num_class = 6
    # model = MobNetV3Large(input_shape, num_class)
    # model = MyModel(input_shape, num_class)
    model = EffNetV2B0(input_shape, num_class)