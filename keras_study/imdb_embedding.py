from keras.datasets import imdb
from keras import preprocessing
from keras import layers
from keras import models

max_features = 1000
maxlen = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data()

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = models.Sequential()
model.add(layers.Embedding(max_features, 8, input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', ['acc'])
print(model.summary())

# model = models.Sequential()
# model.add(layers.Embedding(max_features, 32))
# model.add(layers.Bidirectional(layers.LSTM(32)))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile('adam', 'binary_crossentropy', ['acc'])
# print(model.summary())

history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
