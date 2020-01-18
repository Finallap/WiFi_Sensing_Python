import keras
import numpy as np
import time
import random
import os
from sklearn.model_selection import train_test_split
from data_processing.mat_load_preprocessing import mat_load_preprocessing
from utils.utils import create_directory
from utils.utils import read_all_datasets
from utils.utils import save_logs
from utils.utils import transform_labels

# parameters for data and results dir
root_dir = 'F:\\Git repository\\Experimental result\\2020-01-18\\'
results_dir = os.path.join(root_dir, 'results', '')
write_dir_root = os.path.join(root_dir, 'model-results', '')

ALL_DATASET_NAMES = ['实验室（3t3r）（滤波后）', '会议室(2t3r)(resample)(归一化)', '实验室(3t3r)(未统一长度)']

batch_size = 32
nb_epochs = 50
verbose = True


def build_model(input_shape, nb_classes, pre_model=None):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.normalization.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.normalization.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.normalization.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.pooling.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    if pre_model is not None:

        for i in range(len(model.layers) - 1):
            model.layers[i].set_weights(pre_model.layers[i].get_weights())

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(decay=0.001, ),
                  metrics=['accuracy'])

    return model


def callback_maker(model_save_path, log_dir):
    # reduce learning rate
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.75,
                                                  patience=10, min_lr=1e-5)
    # model checkpoint
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_loss',
                                                       save_best_only=True)
    csv_logger = keras.callbacks.CSVLogger(log_dir + 'training.csv')
    tb = keras.callbacks.TensorBoard(log_dir=log_dir,  # log 目录
                                     histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                     batch_size=32,  # 用多大量的数据计算直方图
                                     write_graph=True,  # 是否存储网络结构图
                                     write_grads=False,  # 是否可视化梯度直方图
                                     write_images=False,  # 是否可视化参数
                                     embeddings_freq=0,
                                     embeddings_layer_names=None,
                                     embeddings_metadata=None)
    return [model_checkpoint, reduce_lr, tb, csv_logger]


def train(x_train, y_train, x_test, y_test, pre_model=None):
    y_true_val = None
    y_pred_val = None

    mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # make the min to zero of labels
    y_train, y_test = transform_labels(y_train, y_test)

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64)

    # transform the labels from integers to one hot vectors
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    start_time = time.time()
    # remove last layer to replace with a new one
    input_shape = (None, x_train.shape[2])
    model = build_model(input_shape, nb_classes, pre_model)

    if verbose == True:
        model.summary()

    # b = model.layers[1].get_weights()

    hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                     verbose=verbose, validation_data=(x_test, y_test), callbacks=callbacks)

    # a = model.layers[1].get_weights()

    # compare_weights(a,b)

    model = keras.models.load_model(model_save_path)

    y_pred = model.predict(x_test)
    # convert the predicted from binary to integer
    y_pred = np.argmax(y_pred, axis=1)

    duration = time.time() - start_time

    df_metrics = save_logs(write_output_dir, hist, y_pred, y_true,
                           duration, lr=True, y_true_val=y_true_val,
                           y_pred_val=y_pred_val)

    print('df_metrics')
    print(df_metrics)

    keras.backend.clear_session()


def read_dataset(root_dir, dataset_name):
    mat_path = os.path.join(root_dir, 'datasets', (dataset_name + '.mat'))
    [csi_train_data, csi_train_label] = mat_load_preprocessing(mat_path)

    # shuffle一下数据
    index = list(range(len(csi_train_data)))
    random.shuffle(index)
    csi_train_data = csi_train_data[index]
    csi_train_label = csi_train_label[index]

    x_train, x_test, y_train, y_test = train_test_split(csi_train_data, csi_train_label, test_size=0.3)

    return x_train, x_test, y_train, y_test


for dataset_name in ALL_DATASET_NAMES:
    # get the directory of the model for this current dataset_name
    write_output_dir = os.path.join(write_dir_root, dataset_name, '')
    # set model output path
    model_save_path = os.path.join(write_output_dir, 'best_model.hdf5')
    # create directory
    create_directory(write_output_dir)

    x_train, x_test, y_train, y_test = read_dataset(root_dir, dataset_name)

    callbacks = callback_maker(model_save_path, write_output_dir)

    train(x_train, y_train, x_test, y_test)
