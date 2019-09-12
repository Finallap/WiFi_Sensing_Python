from keras.models import Sequential
from keras.layers import Dense, LSTM, Masking, Bidirectional, Dropout, TimeDistributed
from keras_contrib.layers.crf import CRF


def bilstm_crf_model(sequence_max_len, input_feature, dropout_rate, num_class, hidden_unit_num):
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(sequence_max_len, input_feature)))
    model.add(Bidirectional(LSTM(hidden_unit_num, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(hidden_unit_num, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(TimeDistributed(Dense(num_class)))
    crf = CRF(num_class, sparse_target=False)
    model.add(crf)

    # compile:loss, optimizer, metrics
    model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    # plot_model(model,to_file= log_dir + 'model.png')
    model.summary()

    return model
