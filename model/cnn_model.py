from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential

def cnn_model(sequence_max_len, input_feature, num_class):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 9), padding='same', activation='relu', input_shape=(sequence_max_len, input_feature, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(3, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(48, kernel_size=(3, 1), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(96, activation='relu'))
    model.add(Dropout(0.6))
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))

    # compile:loss, optimizer, metrics
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model