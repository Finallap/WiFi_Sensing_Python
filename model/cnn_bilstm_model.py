from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Dropout, Input, Masking, MaxPool2D, Embedding, \
    Convolution1D, MaxPooling1D, Conv1D, MaxPool1D
from keras.models import Model
from keras_trans_mask import RemoveMask, RestoreMask, CreateMask


def cnn_bilstm_model(sequence_max_len, input_feature, dropout_rate, num_class, hidden_unit_num, nb_filter, pool_length):
    conv_size = 2
    conv_stride_size = 2
    pooling_size = 2
    inp = Input(shape=(sequence_max_len, input_feature))
    # embedding= Embedding(input_dim=10,ouput_dim=15,mask_zero=True)(inp)
    # mask_layer = Masking(mask_value=0, input_shape=(sequence_max_len, input_feature))(inp)
    mask_layer = CreateMask(mask_value=0)(inp)
    # embed_layer = RestoreMask()([inp, mask_layer])
    removed_layer = RemoveMask()(mask_layer)
    # conv1 = Conv2D(filters=1, kernel_size=(conv_size, conv_size), strides=(conv_stride_size, conv_stride_size))(inp)
    # maxpool1 = MaxPool2D(pool_size=pooling_size, strides=1, padding="valid")(conv1)
    # conv2 = Conv2D(filters=1, kernel_size=(conv_size, conv_size), strides=(conv_stride_size, conv_stride_size))(maxpool1)
    # maxpool2 = MaxPool2D(pool_size=pooling_size, strides=1, padding="valid")(conv2)
    # flatten = TimeDistributed(Flatten())(maxpool2)

    # conv1 = Convolution1D(nb_filter=nb_filter,
    #                       filter_length=10,
    #                       border_mode='valid',
    #                       activation='relu')(removed_layer)
    # maxpool1 = MaxPooling1D(pool_length=pool_length)(conv1)
    conv1 = Conv1D(
        filters=32,
        kernel_size=3,
        padding='same',
    )(removed_layer)
    maxpool1 = MaxPool1D(pool_size=pooling_size, strides=1, padding="same")(conv1)
    # conv2 =  Convolution1D(nb_filter=nb_filter,
    #                 filter_length=10,
    #                 border_mode='valid',
    #                 activation='relu')(maxpool1)
    # maxpool2 = MaxPooling1D(pool_length=pool_length)(conv2)
    # flatten = TimeDistributed(Flatten())(maxpool2)
    restored_layer = RestoreMask()([maxpool1, mask_layer])

    lstm1 = Bidirectional(LSTM(hidden_unit_num, return_sequences=True))(restored_layer)
    dropout1 = Dropout(dropout_rate)(lstm1)
    lstm2 = Bidirectional(LSTM(hidden_unit_num))(dropout1)
    dropout2 = Dropout(dropout_rate)(lstm2)
    dense = Dense(num_class, activation='softmax')(dropout2)

    model = Model(input=inp, outputs=dense)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model
