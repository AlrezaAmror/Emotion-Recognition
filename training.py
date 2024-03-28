import os
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metric import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model.ann import EffNetV2B0
from processing import load_val_data, load_train_data, process_train_data, process_val_data, augment_layers

# get data
dataset_dir = 'mypath/emotion/dataset'
x = 0
CLASS_NAMES = []
for emotion in os.listdir(dataset_dir):
    # print(f'{emotion}\t\t=> {len(os.listdir(dataset_dir+emotion))}')
    x += len(os.listdir(dataset_dir+emotion))
    CLASS_NAMES.append(emotion)

# load data
train_data = load_train_data(dataset_dir, CLASS_NAMES)
val_data = load_val_data(dataset_dir, CLASS_NAMES)

# processing data
train_dataset = process_train_data(train_data)
val_dataset = process_val_data(val_data)

# set the compiler model
loss_func = CategoricalCrossentropy()
metrics = [CategoricalAccuracy(name = 'accuracy')]
            # ,TopKCategoricalAccuracy(name = 'top_k_accuracy')]
checkpoint_callback = ModelCheckpoint('best_weights',
                                      monitor = 'val_accuracy',
                                      save_best_only = True)
early_stopping = EarlyStopping(monitor = 'val_accuracy',
                            #    min_delta = 0.01,
                               restore_best_weights = True,
                               patience = 5)

# load model
model = EffNetV2B0(224, 6)

model.compile(
    optimizer = Adam(learning_rate = 1e-2),
    loss = loss_func,
    metrics = metrics
)

history = model.fit(
    train_dataset, validation_data = val_dataset,
    epochs = 100, verbose = 1, callbacks = [early_stopping]
)

val_accuracy = model.evaluate(val_dataset)[1]
val = int(round(val_accuracy, 2)*100)
create_json = model.to_json()

with open(f'\model\EfficientNet\EffNet{val}.json', 'w') as json_file:
    json_file.write(create_json)

model.save(f'\model\EfficientNet\EffNet{val}.h5')
model.save(f'\model\EfficientNet\EffNet{val}.keras')