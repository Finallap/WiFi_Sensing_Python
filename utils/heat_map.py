acc_data = {
    'D1-S1': [0, 0, 0.796359499, -0.279485757, -3.50877193],
    'D1-S2': [0, 0, -0.227531366, -0.055897183, -3.50877193],
    'D2-S1': [0.884954901, -1.680672663, 0, -0.055897183, -4.260651617],
    'D3-S1': [-0.884957532, -0.84033573, 1.251422043, 0, 1.503759398],
    'D3-S2': [0.884954901, -0.840335759, -1.706484656, -0.335382851, 0],
}

epoch_data = {
    'D1-S1': [0, -5.691056911, -26.27118644, -10.18518519, -12.85714286],
    'D1-S2': [-20.33898305, 0, 13.22751323, 2.777777778, -9.285714286],
    'D2-S1': [-26.27118644, -5.691056911, 0, -10.18518519, -12.85714286],
    'D3-S1': [-33.05084746, -0.81300813, -6.349206349, 0, -7.857142857],
    'D3-S2': [-8.474576271, 3.703703704, 0, -17.59259259, 0],
}
index = ['D1-S1', 'D1-S2', 'D2-S1', 'D3-S1', 'D3-S2']
columns = ['D1-S1', 'D1-S2', 'D2-S1', 'D3-S1', 'D3-S2']
save_dir_out = 'G:\\无源感知研究\\论文\\TURC\\'

import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
# matplotlib.use('agg')
epoch_df = pd.DataFrame(epoch_data, index=index, columns=columns)
acc_df = pd.DataFrame(acc_data, index=index, columns=columns)
# fig, ax = plt.subplots(figsize=(10, 10))
fig, ax = plt.subplots()
plot_sns = sns.heatmap(ax=ax, data=epoch_df, cmap='RdBu', annot=True, fmt=".1f",
                       xticklabels=True, yticklabels=True, vmin=-40, vmax=40)
plot_sns.collections[0].colorbar.ax.tick_params(labelsize=10)
plot_sns.collections[0].colorbar.set_label('% of best model epoch variation after transfer', size=10)
# plt.show()
figure_name = save_dir_out + 'epoch_variation.pdf'
plot_sns.get_figure().savefig(figure_name, bbox_inches='tight')

fig, ax = plt.subplots()
plot_sns = sns.heatmap(ax=ax, data=acc_df, cmap='RdBu', annot=True, fmt=".1f",
                       xticklabels=True, yticklabels=True, vmin=-5, vmax=5)
plot_sns.collections[0].colorbar.ax.tick_params(labelsize=10)
plot_sns.collections[0].colorbar.set_label('% of best model accuracy variation after transfer', size=10)
# plt.show()
figure_name = save_dir_out + 'acc_variation.pdf'
plot_sns.get_figure().savefig(figure_name, bbox_inches='tight')
