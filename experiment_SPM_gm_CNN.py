"""
Experiment using CNNs on GM SPM data.

Results:
MAE:
"""
from pathlib import Path
import time

import pandas as pd
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path.cwd()

# --------------------------------------------------------------------------
random_seed = 42
np.random.seed(random_seed)

# --------------------------------------------------------------------------
# Create experiment's output directory
output_dir = PROJECT_ROOT / 'output' / 'experiments'
output_dir.mkdir(exist_ok=True)

experiment_name = 'SPM_gm_CNN'  # Change here*
experiment_dir = output_dir / experiment_name
experiment_dir.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
tfrecords_path = PROJECT_ROOT / 'data' / 'SPM' / 'gm_train.tfrecords'
demographic_path = PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Training.csv'

demographic_df = pd.read_csv(demographic_path, index_col='subject_ID')

# --------------------------------------------------------------------------
batch_size = 32
train_buf = 2000

train_dataset = tf.data.TFRecordDataset(str(tfrecords_path))

image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'channels': tf.io.FixedLenFeature([], tf.int64),
    'age': tf.io.FixedLenFeature([], tf.int64),
    'site': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, image_feature_description)
    img = tf.io.decode_raw(example['image'], tf.float32)
    img = tf.reshape(img, [example['height'], example['width'], example['depth'], example['channels']])

    # Scale age [17,90] --> [-1,1]
    age = example['age']
    age = (tf.cast(age, dtype=tf.float32) - tf.cast(17.0, dtype=tf.float32)) / tf.cast((90.0-17.0), dtype=tf.float32)
    age = (age - tf.cast(0.5, dtype=tf.float32)) * tf.cast(2.0, dtype=tf.float32)

    return img, age


train_dataset = train_dataset.map(_parse_image_function, num_parallel_calls=10)
train_dataset = train_dataset.shuffle(buffer_size=600)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# --------------------------------------------------------------------------
# Model
# Using JamesNet
inputs = tf.keras.layers.Input(shape=(94, 120, 96, 1))
x = tf.keras.layers.Conv3D(filters=8, kernel_size=3, strides=1, padding='same', activation='linear', use_bias='True')(inputs)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.Conv3D(filters=8, kernel_size=3, strides=1, padding='same', activation='linear', use_bias='False')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2, padding='same')(x)

x = tf.keras.layers.Conv3D(filters=16, kernel_size=3, strides=1, padding='same', activation='linear', use_bias='True')(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.Conv3D(filters=16, kernel_size=3, strides=1, padding='same', activation='linear', use_bias='False')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2, padding='same')(x)

x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, strides=1, padding='same', activation='linear', use_bias='True')(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, strides=1, padding='same', activation='linear', use_bias='False')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2, padding='same')(x)

x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=1, padding='same', activation='linear', use_bias='True')(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.Conv3D(filters=46, kernel_size=3, strides=1, padding='same', activation='linear', use_bias='False')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2, padding='same')(x)

x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, strides=1, padding='same', activation='linear', use_bias='True')(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, strides=1, padding='same', activation='linear', use_bias='False')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2, padding='same')(x)

x = tf.keras.layers.Flatten()(x)
prediction = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.Model(inputs=inputs, outputs=prediction)

# --------------------------------------------------------------------------
# Loss function
mse = tf.keras.losses.MeanSquaredError()

# --------------------------------------------------------------------------
# Optimizer
nn_optimizer = tf.keras.optimizers.Adam(lr=3e-4)


# -------------------------------------------------------------------------------------------------------------
# Training function
@tf.function
def train_step(batch_x, batch_y):
    with tf.GradientTape() as tape:
        y_predicted = model(batch_x, training=True)
        loss_value = mse(batch_y, y_predicted)

    nn_grads = tape.gradient(loss_value, model.trainable_variables)
    nn_optimizer.apply_gradients(zip(nn_grads, model.trainable_variables))

    return loss_value


# -------------------------------------------------------------------------------------------------------------
# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    start = time.time()

    epoch_loss_avg = tf.metrics.Mean()
    for batch, (batch_x, batch_y) in enumerate(train_dataset):
        loss_value = train_step(batch_x, batch_y)
        epoch_loss_avg(loss_value)

    epoch_time = time.time() - start
    print('{:4d}: TIME: {:.2f} ETA: {:.2f} LOSS: {:.8f}' \
          .format(epoch, epoch_time,
                  epoch_time * (n_epochs - epoch),
                  epoch_loss_avg.result()))
