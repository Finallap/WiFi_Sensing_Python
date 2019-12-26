from keras.datasets import boston_housing
from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.975
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean()
train_data -= mean
std = train_data.std()
train_data /= std

test_data -= mean
test_data /= std


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add((layers.Dense(1)))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    # print(model.summary())
    return model


k = 4
num_val_samples = len(train_data) // k
num_epochs = 3500
all_scores = []
all_mae_history = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]

    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        epochs=num_epochs, verbose=0,
                        validation_data=(val_data, val_targets))
    mae_history = history.history['val_mean_absolute_error']
    all_mae_history.append(mae_history)
    val_mse, val_mae = model.evaluate(val_data, val_targets)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


average_mae_history = [
    np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)
]
smooth_mae_history = smooth_curve(average_mae_history[15:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
