from keras.layers import Dense, LSTM, Masking, Bidirectional, Dropout
from keras.models import Sequential


def bilstm_model(sequence_max_len, input_feature, dropout_rate, num_class, hidden_unit_num):
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(sequence_max_len, input_feature)))
    model.add(Bidirectional(LSTM(hidden_unit_num, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(hidden_unit_num)))
    model.add(Dropout(dropout_rate))
    # model.add(TimeDistributed(Dense(num_class)))
    # crf_layer = CRF(num_class, sparse_target=True)
    # model.add(crf_layer)
    model.add(Dense(num_class, activation='softmax'))

    # compile:loss, optimizer, metrics
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # plot_model(model,to_file= log_dir + 'model.png')
    model.summary()
    # model.compile('rmsprop', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])

    return model
