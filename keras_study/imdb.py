from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import regularizers

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

num_words = 20000

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
)
decode_word_index = ' '.join(
    reverse_word_index.get(i - 3, '?') for i in train_data[0]
)


def vectorize_sequences(sequences, dimension=num_words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.array((train_labels).astype('float32'))
y_test = np.array((test_labels).astype('float32'))

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(num_words,), kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
results = model.evaluate(x_test, y_test)

print(results)
