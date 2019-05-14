"""
Experiment using CNNs on GM SPM data.

Results:
MAE:
"""
from pathlib import Path
import time

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ShuffleSplit
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
numpy_data_path = PROJECT_ROOT / 'data' / 'SPM' / 'gm'
demographic_path = PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Training.csv'

# Reading data. If necessary, create new reader in helper_functions.
# x, demographic_df = read_gram_matrix(str(gram_matrix_path), str(demographic_path))
x1 = np.random.randn(500,121,145,121,1).astype('float32')
x2 = np.random.randn(500,121,145,121,1).astype('float32')
x3 = np.random.randn(500,121,145,121,1).astype('float32')
x4 = np.random.randn(500,121,145,121,1).astype('float32')
x5 = np.random.randn(640,121,145,121,1).astype('float32')

x = np.append(x1,x2,axis=0)
del x1
del x2
x = np.append(x,x3,axis=0)
del x3
x = np.append(x,x4,axis=0)
del x4
x = np.append(x,x5,axis=0)
del x5

demographic_df = pd.read_csv(demographic_path, index_col='subject_ID')

# --------------------------------------------------------------------------
# Using only age
y = demographic_df['age'].values

# If necessary, extract gender and site from demographic_df too.

# --------------------------------------------------------------------------
rs = ShuffleSplit(n_splits=1, test_size=.25, random_state=random_seed)

# --------------------------------------------------------------------------
train_idx, test_idx = next(rs.split(x))
n_subjs = 300
x_train, x_test = x[train_idx[:n_subjs]], x[test_idx]
y_train, y_test = y[train_idx[:n_subjs]], y[test_idx]

batch_size = 32
train_buf = n_subjs

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=train_buf)
train_dataset = train_dataset.batch(batch_size)

# --------------------------------------------------------------------------
# Model
inputs = tf.keras.layers.Input(shape=(121,145,121, 1))

x = tf.keras.layers.Conv3D(filters=16, kernel_size=5, strides=3, padding='same', activation='relu')(inputs)
x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
x = tf.keras.layers.Conv3D(filters=1, kernel_size=3, strides=2, padding='same', activation='relu')(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
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
batch_x, batch_y = next(iter(train_dataset))
#
n_epochs = 350
# for epoch in range(n_epochs):
epoch = 1
start = time.time()

# epoch_loss_avg = tf.metrics.Mean()
loss_value = train_step(batch_x, batch_y)
# epoch_loss_avg(loss_value)

epoch_time = time.time() - start
print('{:4d}: TIME: {:.2f} ETA: {:.2f} LOSS: {:.4f}' \
      .format(epoch, epoch_time,
              epoch_time * (n_epochs - epoch),
              # epoch_loss_avg.result()))
              loss_value))




