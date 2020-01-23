from builtins import print

import numpy as np
import pandas as pd 
import matplotlib
from pandas.tests.extension import decimal

matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
import os
import operator
from scipy.stats import wilcoxon

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from scipy.io import loadmat

ALL_DATASET_NAMES = ['50words', 'Adiac', 'ArrowHead', 'Beef', ]

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def create_directory(directory_path): 
    if os.path.exists(directory_path): 
        return None
    else: 
        try: 
            os.makedirs(directory_path)
        except: 
            # in case another machine created the path meanwhile !:(
            return None 
        return directory_path

def read_dataset(root_dir,dataset_name):
    datasets_dict = {}
    file_name = root_dir+'/'+dataset_name+'/'+dataset_name
    x_train, y_train = readucr(file_name+'_TRAIN')
    x_test, y_test = readucr(file_name+'_TEST')
    datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),
        y_test.copy())

    return datasets_dict

def read_all_datasets(root_dir,archive_name):
    datasets_dict = {}

    dataset_names_to_sort = []

    for dataset_name in ALL_DATASET_NAMES:
        root_dir_dataset =root_dir+'/archives/'+archive_name+'/'+dataset_name+'/'
        file_name = root_dir_dataset+dataset_name
        x_train, y_train = readucr(file_name+'_TRAIN')
        x_test, y_test = readucr(file_name+'_TEST')

        datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),
                y_test.copy())

        dataset_names_to_sort.append((dataset_name,len(x_train)))

    dataset_names_to_sort.sort(key=operator.itemgetter(1))

    for i in range(len(ALL_DATASET_NAMES)):
        ALL_DATASET_NAMES[i] = dataset_names_to_sort[i][0]

    return datasets_dict


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res

def transform_labels(y_train,y_test):
    """
    Transform label to min equal zero and continuous 
    For example if we have [1,3,4] --->  [0,1,2]
    """
    # init the encoder
    encoder = LabelEncoder()
    # concat train and test to fit
    y_train_test = np.concatenate((y_train,y_test),axis =0)
    # fit the encoder
    encoder.fit(y_train_test)
    # transform to min zero and continuous labels
    new_y_train_test = encoder.transform(y_train_test)
    # resplit the train and test
    new_y_train = new_y_train_test[0:len(y_train)]
    new_y_test = new_y_train_test[len(y_train):]
    return new_y_train, new_y_test


def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['acc']
    df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    if lr == True:
        # print('row_best_model')
        # print(row_best_model)
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)
    # print('df_best_model')
    # print(df_best_model)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics

def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_'+metric])
    plt.title('model '+metric)
    plt.ylabel(metric,fontsize='large')
    plt.xlabel('epoch',fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name,bbox_inches='tight')
    plt.close()