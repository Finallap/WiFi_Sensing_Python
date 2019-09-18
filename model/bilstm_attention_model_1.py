from keras.layers import *
from keras.models import *


def attention_3d_block(inputs,sequence_max_len):
    TIME_STEPS = sequence_max_len
    # input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def bilstm_attention_model_1(sequence_max_len, input_feature, dropout_rate, num_class, hidden_unit_num):
    # build RNN model with attention
    inputs = Input(shape=(sequence_max_len, input_feature))
    drop1 = Dropout(dropout_rate)(inputs)
    lstm_out = Bidirectional(LSTM(hidden_unit_num, return_sequences=True), name='bilstm')(drop1)
    attention_mul = attention_3d_block(lstm_out,sequence_max_len)
    attention_flatten = Flatten()(attention_mul)
    drop2 = Dropout(dropout_rate)(attention_flatten)
    output = Dense(num_class, activation='sigmoid')(drop2)
    model = Model(inputs=inputs, outputs=output)

    # compile:loss, optimizer, metrics
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # plot_model(model,to_file= log_dir + 'model.png')
    model.summary()

    return model
