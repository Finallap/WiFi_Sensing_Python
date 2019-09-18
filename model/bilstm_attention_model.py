from keras.layers import *
from keras.models import *


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])


def attention_3d_block(inputs):
    TIME_STEPS = 20
    SINGLE_ATTENTION_VECTOR = False
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def Att(att_dim, inputs, name):
    V = inputs
    QK = Dense(att_dim, bias=None)(inputs)
    QK = Activation("softmax", name=name)(QK)
    MV = Multiply()([V, QK])
    return (MV)


def bilstm_attention_model(sequence_max_len, input_feature, dropout_rate, num_class, hidden_unit_num):
    # build model
    inp = Input(shape=(sequence_max_len, input_feature))
    masking = Masking(mask_value=0, input_shape=(sequence_max_len, input_feature))(inp)
    lstm1 = Bidirectional(LSTM(hidden_unit_num, return_sequences=True))(masking)
    dropout1 = Dropout(dropout_rate)(lstm1)
    lstm2 = Bidirectional(LSTM(hidden_unit_num))(dropout1)
    dropout2 = Dropout(dropout_rate)(lstm2)
    # att = AttLayer()(dropout2)
    att = Att(256, dropout2, "att")
    dense = Dense(num_class, activation='softmax')(att)
    model = Model(inp, dense)

    # attention_mul = attention_3d_block(dropout2)
    # # restnet
    # attention = concatenate(axis=-1)([inp, attention_mul])
    # attention = Flatten()(attention)
    # output = Dense(1, activation='sigmoid')(attention)
    # model = Model(input=[inp], output=output)


    # compile:loss, optimizer, metrics
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # plot_model(model,to_file= log_dir + 'model.png')
    model.summary()

    return model
