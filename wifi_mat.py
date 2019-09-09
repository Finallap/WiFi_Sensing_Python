from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM, Masking, Bidirectional, Dropout, TimeDistributed
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Optimizer
from keras_contrib.layers.crf import CRF
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np

def scheduler(epoch):
    # 每隔8个epoch，学习率减小为原来的0.8
    if epoch % 8 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.7)
        print("lr changed to {}".format(lr * 0.7))
    return K.get_value(model.optimizer.lr)

#parameters for LSTM
input_feature = 180  #特征个数
num_class = 6
dropout_rate = 0.2

#载入mat数据
mat_data = loadmat("G:/无源感知研究/数据采集/2019_07_18/实验室（滤波后）.mat")

#读出csi，确定样本个数
csi_train = mat_data["csi_train"]
sample_count = len(csi_train)#训练样本个数
csi_train = csi_train.reshape(sample_count)

#确定各个样本的长度
sequenceLengths = []
for i in range(sample_count):
    sequenceLengths.append(csi_train[i].shape[1])
sequence_max_len = max(sequenceLengths)

#创建训练使用的数组，将csi训练数据复制进去，长度不够的进行padding
csi_train_data = np.zeros((sample_count,sequence_max_len,input_feature))
for i in range(sample_count):
    temp_data = np.vstack((np.transpose(csi_train[i]),np.zeros((sequence_max_len - csi_train[i].shape[1] ,input_feature))))
    csi_train_data[i] = temp_data

#创建训练使用的标签数组，将动作名称标签复制进去（人物标签暂时不复制）
csi_train_label = []
for i in range(len(mat_data["csi_label"])):
    csi_train_label.append(mat_data["csi_label"][i][0][0])
#将训练标签数组进行one-hot编码
lb = LabelBinarizer()
csi_train_label = lb.fit_transform(csi_train_label)

#划分训练集和测试集
train,test,train_label,test_label = train_test_split(csi_train_data,csi_train_label,test_size=0.3)

#build model
model = Sequential()
model.add(Masking(mask_value= 0,input_shape=(sequence_max_len, input_feature)))
model.add(Bidirectional(LSTM(128,return_sequences=True)))
model.add(Dropout(dropout_rate))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(dropout_rate))
#model.add(TimeDistributed(Dense(num_class)))
#crf_layer = CRF(num_class, sparse_target=True)
#model.add(crf_layer)
model.add(Dense(num_class, activation='softmax'))

#compile:loss, optimizer, metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#model.compile('rmsprop', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])

#train: epcoch, batch_size
reduce_lr = LearningRateScheduler(scheduler)
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
history = model.fit(train, train_label, epochs=50, batch_size=64, verbose=1,
                    callbacks=[checkpoint, reduce_lr], validation_data=(test, test_label))

score = model.evaluate(test, test_label,batch_size=64, verbose=1)
print(score)

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig(r'acc.png', dpi=900)

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig(r'loss.png', dpi=900)