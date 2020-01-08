import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    city = ['Austin', 'Boulder']
    num_city = 0
    num_mon = 5
    path1 = './processed_data/{}_in_{}.csv'.format(city[num_city], months[num_mon])
    path2 = './processed_data/{}_in_{}.csv'.format(city[num_city], months[num_mon])

    df_pro = pd.read_csv(path1)
    df_pro = df_pro.drop(df_pro.index[-1], axis=0)

    nan_sum = df_pro.isna().sum()
    nan_sum = nan_sum[nan_sum.values < 20]
    df_pro = df_pro[nan_sum.index]

    df_pro = df_pro.fillna(method='ffill', axis=1)
    df_pro = df_pro.fillna(method='bfill', axis=1)

    df_pro = df_pro.drop(['avg'], axis=1)
    df_pro['avg'] = df_pro.mean(axis=1)

    if path1 == './processed_data/{}_in_{}.csv'.format(city[1-1], months[3-1]):
        df_replace = df_pro.iloc[967 - 96 + 1:967 - 96 + 1 + 4]
        replace_local_15min = ['2018-03-11 02:00:00-05', '2018-03-11 02:15:00-05', '2018-03-11 02:30:00-05',
                               '2018-03-11 02:45:00-05']
        df_replace.drop(['local_15min'], axis=1)
        df_replace['local_15min'] = replace_local_15min

        df_pro = pd.concat([df_pro, df_replace], ignore_index=True, axis=0)

        df_pro = df_pro.sort_values(by=['local_15min'])

    df_pro = df_pro.set_index('local_15min')
    df_pro.to_csv('./processed_data_connected/{}_in_{}_connected.csv'.format(city[num_city], months[num_mon]))

    # ------------------------------------
    # plot the user's histogram
    # ------------------------------------
    df_user = df_pro['59'].to_numpy().reshape((-1, 96))
    df_user_mean = np.mean(df_user, axis=0)
    fig, ax = plt.subplots(1, 1)
    for k in range(27):
        plt.plot(df_user[k, :])
    plt.xlabel('time points')
    plt.ylabel('Power/kW')
    plt.show()
    plt.close()

    # ------------------------------------
    # plot the user's average histogram
    # ------------------------------------
    fig, ax = plt.subplots(1, 1)
    for col in df_pro.columns:
        df_user = df_pro[col].to_numpy().reshape((-1, 96))
        df_user_mean = np.mean(df_user, axis=0)
        plt.plot(df_user_mean)
    plt.xlabel('time points')
    plt.ylabel('Power/kW')
    plt.show()
    plt.close()





