CONFIG = {
    'dir_path': "/home/shengby/Datasets/CSI_mat/Widar3_merge_filter_transpose",
    # 'dir_path':"G:/无源感知研究/数据采集/2019_07_18/",
    'data_name': "part3.mat",
    'source_name': "实验室(2t3r)(resample)(归一化).mat",
    'target_name': "会议室(2t3r)(resample)(归一化).mat",
    'num_workers': 1,
    'pin_memory': False,
    'batch_size': 32,
    'epochs': 1000,
    'log_interval': 1,
    'lr': 1e-3,
    'momentum': .9,
    'l2_decay': 0.01,
    'lambda': 10,
    'sequence_max_len': 512,
    'input_feature': 270,
    # 'sequence_max_len': 677,
    # 'input_feature': 270,
    'hidden_size': 300,
    'n_class': 6,
    'model_type': 'lstm',
    'model_save_path': '/home/shengby/Experimental_Results/Fang/Widar3_transfer/part3_orignal_lstm300/model.pkl',
    'tensorboard_log_path': '/home/shengby/Experimental_Results/Fang/Widar3_transfer/part3_orignal_lstm300',
    'diff_lr': True,
    'gamma': 1
}
