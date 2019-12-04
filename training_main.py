from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, CSVLogger
import matplotlib.pyplot as plt
import keras.backend as K
import random
import numpy as np
from data_processing.mat_load_preprocessing import mat_load_preprocessing
from model.bilstm_model import bilstm_model
from model.bilstm_crf_model import bilstm_crf_model
from model.bilstm_attention_model import bilstm_attention_model
from model.bilstm_attention_model_1 import bilstm_attention_model_1
from model.cnn_bilstm_model import cnn_bilstm_model
from model.cnn_model import cnn_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def plot(log_dir):
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


def callback_maker(log_dir):
    def scheduler(epoch):
        # 每隔8个epoch，学习率减小为原来的0.8
        if epoch % 8 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.7)
            print("lr changed to {}".format(lr * 0.7))
        return K.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)
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
    # return [checkpoint, reduce_lr, tb, csv_logger]
    # return [checkpoint, tb, csv_logger]
    return [tb, csv_logger]


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # parameters for dataset
    mat_path = "G:/无源感知研究/数据采集/Cross Scene(201911)/Scene1(lab)(滤波后).mat"
    input_feature = 90  # 特征个数
    num_class = 12

    # parameters for LSTM model
    dropout_rate = 0.2
    hidden_unit_num = 100

    # parameters for Convolution model
    nb_filter = 64
    pool_length = 2

    # parameters for train
    epochs = 50
    batch_size = 64
    log_dir = 'F:\\Git repository\\Experimental result\\2019-12-03\\cnn_16_32_64_dense_64\\'

    [csi_train_data, csi_train_label] = mat_load_preprocessing(mat_path, input_feature)
    sample_count = csi_train_data.shape[0]
    sequence_max_len = csi_train_data.shape[1]
    input_feature = csi_train_data.shape[2]

    # shuffle一下数据
    index = list(range(len(csi_train_data)))
    random.shuffle(index)
    csi_train_data = csi_train_data[index]
    csi_train_label = csi_train_label[index]

    csi_train_data = csi_train_data.reshape((sample_count, sequence_max_len, input_feature, 1))

    # 划分训练集和测试集
    train, test, train_label, test_label = train_test_split(csi_train_data, csi_train_label, test_size=0.3)

    # build model
    # model = bilstm_model(sequence_max_len=sequence_max_len,
    #                      input_feature=input_feature,
    #                      dropout_rate=dropout_rate,
    #                      num_class=num_class,
    #                      hidden_unit_num=hidden_unit_num)
    # model = bilstm_crf_model(sequence_max_len, input_feature, dropout_rate, num_class, hidden_unit_num)
    # model = bilstm_attention_model(sequence_max_len, input_feature, dropout_rate, num_class, hidden_unit_num)
    # model = bilstm_attention_model_1(sequence_max_len, input_feature, dropout_rate, num_class, hidden_unit_num)
    model = cnn_model(sequence_max_len, input_feature, num_class)
    # model = cnn_bilstm_model(sequence_max_len, input_feature, dropout_rate,
    #                          num_class, hidden_unit_num, nb_filter, pool_length)

    # callbacks
    callback_list = callback_maker(log_dir)

    # fit model
    history = model.fit(train, train_label, epochs=epochs, batch_size=batch_size, verbose=1,
                        callbacks=callback_list, validation_data=(test, test_label))
    # history = model.fit_generator(generator(), epochs=epochs, steps_per_epoch=len(train) // (batch_size * epochs),
    #                               verbose=1, callbacks=callback_list, validation_data=(test, test_label))

    # evaluate model
    score = model.evaluate(test, test_label, batch_size=batch_size, verbose=1)
    print(score)

    # 绘制结果图
    plot(log_dir)
