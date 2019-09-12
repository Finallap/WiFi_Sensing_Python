from keras.layers import Input, Dense, LSTM
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
import numpy as np

output_dim = 1
batch_size = 256  # 每轮训练模型时，样本的数量
epochs = 60  # 训练60轮次
seq_len = 5
hidden_size = 128

TIME_STEPS = 1000
INPUT_DIM = 180

lstm_units = 64

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
# drop1 = Dropout(0.3)(inputs)

x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)  # , padding = 'same'
# x = Conv1D(filters=128, kernel_size=5, activation='relu')(output1)#embedded_sequences
x = MaxPooling1D(pool_size=5)(x)
x = Dropout(0.2)(x)
print(x.shape)

lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
# lstm_out = LSTM(lstm_units,activation='relu')(x)
print(lstm_out.shape)

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers


# Attention GRU network  未用
class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 128
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


from keras.layers import Input, Dense, merge

# ATTENTION PART STARTS HERE
attention_probs = Dense(128, activation='sigmoid', name='attention_vec')(lstm_out)
# attention_mul=layers.merge([stm_out,attention_probs], output_shape],mode='concat',concat_axis=1))
attention_mul = Multiply()([lstm_out, attention_probs])
# attention_mul = merge([lstm_out, attention_probs],output_shape=32, name='attention_mul', mode='mul')

output = Dense(1, activation='sigmoid')(attention_mul)
# output = Dense(10, activation='sigmoid')(drop2)

model = Model(inputs=inputs, outputs=output)
print(model.summary())

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
y_pred = model.predict(X_test)
print('MSE Train loss:', model.evaluate(X_train, y_train, batch_size=batch_size))
print('MSE Test loss:', model.evaluate(X_test, y_test, batch_size=batch_size))
plt.plot(y_test, label='test')
plt.plot(y_pred, label='pred')
plt.legend()
plt.show()
