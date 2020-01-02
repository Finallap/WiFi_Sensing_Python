from keras.datasets import imdb
from keras import preprocessing
from keras import layers
from keras import models
from keras import optimizers

max_features = 10000
maxlen = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data()

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=maxlen))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalAveragePooling1D())
# model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizers.RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])
print(model.summary())

history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
