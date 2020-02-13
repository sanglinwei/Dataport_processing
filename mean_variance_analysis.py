import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import os
if __name__ == '__main__':
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    city = ['Austin', 'Boulder']
    num_city = 0

    for num_mon in range(0, 1):
        # ------------------------------------
        # read the data
        # ------------------------------------
        df_pro = pd.read_csv('./processed_data_connected/{}_in_{}_connected.csv'.format(city[num_city], months[num_mon]))
        df_pro = df_pro.set_index('local_15min')
        df_pro = df_pro.drop(['avg'], axis=1)

        df_mean = pd.read_csv('./data_analysis_norm/{1}_mean/{1}_mean_{0}.csv'.format(months[num_mon], city[num_city]))
        df_std = pd.read_csv('./data_analysis_norm/{1}_std/{1}_std_{0}.csv'.format(months[num_mon], city[num_city]))

        df_mean_sum = df_mean['sum']
        df_mean = df_mean.drop(['Unnamed: 0', 'sum'], axis=1)
        df_mean_sum_theory = df_mean.sum(axis=1)

        df_variance_sum = df_std['sum'].apply(lambda x: x)
        df_variance = df_std.drop(['Unnamed: 0', 'sum'], axis=1).applymap(lambda x: x)
        df_variance_theory = df_variance.sum(axis=1)
        # i.i.d

