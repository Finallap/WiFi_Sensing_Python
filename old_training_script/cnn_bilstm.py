from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Dropout, Input, Masking, MaxPool2D, Embedding, \
    Convolution1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, CSVLogger
from keras.utils import plot_model
from keras_trans_mask import RemoveMask, RestoreMask
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np
import keras


class cnn_bilstm():
    def get_model(sequence_max_len, input_feature, nb_filter, pool_length, dropout_rate, sample_count):
        conv_size = 2
        conv_stride_size = 2
        pooling_size = (2, 2)
        inp = Input(shape=(sequence_max_len, input_feature))
        # embedding= Embedding(input_dim=10,ouput_dim=15,mask_zero=True)(inp)
        masking = Masking(mask_value=0, input_shape=(sequence_max_len, input_feature))(inp)
        removed_layer = RemoveMask()(masking)
        # conv1 = Conv2D(filters=1, kernel_size=(conv_size, conv_size), strides=(conv_stride_size, conv_stride_size))(inp)
        # maxpool1 = MaxPool2D(pool_size=pooling_size, strides=1, padding="valid")(conv1)
        # conv2 = Conv2D(filters=1, kernel_size=(conv_size, conv_size), strides=(conv_stride_size, conv_stride_size))(maxpool1)
        # maxpool2 = MaxPool2D(pool_size=pooling_size, strides=1, padding="valid")(conv2)
        # flatten = TimeDistributed(Flatten())(maxpool2)

        conv1 = Convolution1D(nb_filter=nb_filter,
                              filter_length=10,
                              border_mode='valid',
                              activation='relu')(removed_layer)
        maxpool1 = MaxPooling1D(pool_length=pool_length)(conv1)
        # conv2 =  Convolution1D(nb_filter=nb_filter,
        #                 filter_length=10,
        #                 border_mode='valid',
        #                 activation='relu')(maxpool1)
        # maxpool2 = MaxPooling1D(pool_length=pool_length)(conv2)
        # flatten = TimeDistributed(Flatten())(maxpool2)
        restored_layer = RestoreMask()([maxpool1, masking])

        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(maxpool1)
        dropout1 = Dropout(dropout_rate)(lstm1)
        lstm2 = Bidirectional(LSTM(128))(dropout1)
        dropout2 = Dropout(dropout_rate)(lstm2)
        dense = Dense(num_class, activation='softmax')(dropout2)

        model = Model(input=inp, outputs=dense)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        return model

    def scheduler(epoch):
        # 每隔8个epoch，学习率减小为原来的0.8
        if epoch % 8 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.8)
            print("lr changed to {}".format(lr * 0.8))
        return K.get_value(model.optimizer.lr)


if __name__ == "__main__":
    # parameters for LSTM
    input_feature = 180  # 特征个数
    num_class = 6
    dropout_rate = 0.2
    log_dir = 'F:\\Git repository\\Experimental result\\2019_09_10_keras\\conv1_maxpool1_bilstm2_lrfix\\'

    # 载入mat数据
    mat_data = loadmat("G:/无源感知研究/数据采集/2019_07_18/实验室（滤波后）.mat")

    # 读出csi，确定样本个数
    csi_train = mat_data["csi_train"]
    sample_count = len(csi_train)  # 训练样本个数
    csi_train = csi_train.reshape(sample_count)

    # 确定各个样本的长度
    sequenceLengths = []
    for i in range(sample_count):
        sequenceLengths.append(csi_train[i].shape[1])
    sequence_max_len = max(sequenceLengths)

    # 创建训练使用的数组，将csi训练数据复制进去，长度不够的进行padding
    # csi_train_data = keras.preprocessing.sequence.pad_sequences(csi_train, maxlen=sequence_max_len, padding='post', value=0.)
    csi_train_data = np.zeros((sample_count, sequence_max_len, input_feature))
    for i in range(sample_count):
        temp_data = np.vstack(
            (np.transpose(csi_train[i]), np.zeros((sequence_max_len - csi_train[i].shape[1], input_feature))))
        csi_train_data[i] = temp_data

    # 创建训练使用的标签数组，将动作名称标签复制进去（人物标签暂时不复制）
    csi_train_label = []
    for i in range(len(mat_data["csi_label"])):
        csi_train_label.append(mat_data["csi_label"][i][0][0])
    # 将训练标签数组进行one-hot编码
    lb = LabelBinarizer()
    csi_train_label = lb.fit_transform(csi_train_label)

    # 划分训练集和测试集
    train, test, train_label, test_label = train_test_split(csi_train_data, csi_train_label, test_size=0.3)

    # Convolution
    nb_filter = 64
    pool_length = 2

    # Training
    batch_size = 128
    nb_epoch = 100

    # build model
    model = cnn_bilstm.get_model(sequence_max_len, input_feature, nb_filter, pool_length, dropout_rate, sample_count)

    # callbacks
    reduce_lr = LearningRateScheduler(cnn_bilstm.scheduler)
    filepath = log_dir + 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csv_logger = CSVLogger(log_dir + 'training.csv')
    tb = TensorBoard(log_dir=log_dir,  # log 目录
                     histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                     batch_size=32,  # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=False,  # 是否可视化梯度直方图
                     write_images=False,  # 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)

    # fit model
    history = model.fit(train, train_label, epochs=150, batch_size=32, verbose=1,
                        callbacks=[checkpoint, tb, csv_logger], validation_data=(test, test_label))

    score = model.evaluate(test, test_label, batch_size=32, verbose=1)
    print(score)

    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig(log_dir + 'acc.png', dpi=900)

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig(log_dir + 'loss.png', dpi=900)
