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

    if path1 == './processed_data/{}_in_{}.csv'.format(city[1 - 1], months[3 - 1]):
        df_replace = df_pro.iloc[967 - 96 + 1:967 - 96 + 1 + 4]
        replace_local_15min = ['2018-03-11 02:00:00-05', '2018-03-11 02:15:00-05', '2018-03-11 02:30:00-05',
                               '2018-03-11 02:45:00-05']
        df_replace.drop(['local_15min'], axis=1)
        df_replace['local_15min'] = replace_local_15min

        df_pro = pd.concat([df_pro, df_replace], ignore_index=True, axis=0)

        df_pro = df_pro.sort_values(by=['local_15min'])

    df_pro = df_pro.set_index('local_15min')
    # df_pro.to_csv('./processed_data_connected/{}_in_{}_connected.csv'.format(city[num_city], months[num_mon]))
    df_pro = df_pro.drop(['avg'], axis=1)
    # ------------------------------------
    # plot the user's histogram
    # ------------------------------------
    # df_user = df_pro['59'].to_numpy().reshape((-1, 96))
    # df_user_mean = np.mean(df_user, axis=0)
    # fig, ax = plt.subplots(1, 1)
    # for k in range(27):
    #     plt.plot(df_user[k, :])
    # plt.xlabel('time points')
    # plt.ylabel('Power/kW')
    # plt.show()
    # plt.close()
    # ------------------------------------
    # deal with meta data
    # ------------------------------------
    meta_data = pd.read_csv('./meta_data/metadata.csv')
    col_df = df_pro.columns
    col_df = col_df.to_numpy()
    col_df = col_df.astype('int64')
    meta_data = meta_data.set_index('dataid')
    meta_data_extracted = pd.DataFrame(meta_data, index=col_df)
    # ------------------------------------
    # plot the user's average
    # ------------------------------------
    # fig, ax = plt.subplots(1, 1)
    # for col in df_pro.columns:
    #     df_user = df_pro[col].to_numpy().reshape((-1, 96))
    #     df_user_mean = np.mean(df_user, axis=0)
    #     plt.plot(df_user_mean)
    # plt.xlabel('time points')
    # plt.ylabel('Power/kW')
    # plt.title('average users')
    # plt.show()
    # plt.close()
    # ------------------------------------------------
    # plot scatter data
    # ------------------------------------------------
    for col in df_pro.columns:
        fig, ax = plt.subplots(1, 1)
        df_user = df_pro[col].to_numpy().reshape((-1, 96))
        for j in range(df_user.shape[0]):
            plt.scatter(np.arange(96), df_user[j, :], c='k', alpha=0.3, s=4)
        plt.xlabel('time points')
        plt.ylabel('Power/kW')
        plt.title('average users {} in {} with total footage {}'.
                  format(col, months[num_mon], meta_data_extracted.loc[int(col), 'total_square_footage']))
        plt.savefig('./month_load_image/scatter_plot/{}/average users {} in {}'.
                    format(months[num_mon], col, months[num_mon]))
        plt.close()
    # ------------------------------------------------
    # plot average data
    # ------------------------------------------------
    for col in df_pro.columns:
        fig, ax = plt.subplots(1, 1)
        df_user = df_pro[col].to_numpy().reshape((-1, 96))
        df_user_mean = np.mean(df_user, axis=0)
        plt.plot(df_user_mean)
        plt.xlabel('time points')
        plt.ylabel('Power/kW')
        plt.title('average users {} in {} with total footage {}'.
                  format(col, months[num_mon], meta_data_extracted.loc[int(col), 'total_square_footage']))
        plt.savefig('./month_load_image/average_image/{}/average users {} in {}'.
                    format(months[num_mon], col, months[num_mon]))
        plt.close()
    # ------------------------------------------------
    # plot box figure
    # ------------------------------------------------
    for col in df_pro.columns:
        fig, ax = plt.subplots(1, 1)
        # every four point output one thing
        df_user = df_pro[col].to_numpy().reshape((-1, 24, 4))
        df_user = np.swapaxes(df_user, 1, 2).reshape(-1, 24)
        df_user_pd = pd.DataFrame(df_user, columns=np.arange(24))
        df_user_pd.boxplot(column=list(np.arange(0, 24)))
        plt.xlabel('time points')
        plt.ylabel('Power/kW')
        plt.title('box_fig users {} in {} with total footage {}'.
                  format(col, months[num_mon], meta_data_extracted.loc[int(col), 'total_square_footage']))
        plt.savefig(('./month_load_image/boxplot/{}/average users {} in {}'.
                    format(months[num_mon], col, months[num_mon])))
        plt.close()
    # # ------------------------------------------------
    # # plot hist figure
    # # ------------------------------------------------
    # for col in df_pro.columns:
    #     fig, ax = plt.subplots(1, 1)
    #     df_pro[col].plot.hist(bins=12)
    #     plt.xlabel('time points')
    #     plt.ylabel('Power/kW')
    #     plt.title('box_fig users {} in {} with total footage {}'.
    #               format(col, months[num_mon], meta_data_extracted.loc[col, 'total_square_footage']))
    #     plt.savefig('./month_load_image/hist_plot/hist_user_{}.png'.format(col))
    #     plt.show()
    #     plt.close()
    # ------------------------------------------------
    # meta_data analysis
    # ------------------------------------------------
# meta_data = pd.read_csv('./meta_data/metadata.csv')
# col_df = df_pro.columns.drop(['avg', 'sum'])
# meta_data_extracted = pd.DataFrame(meta_data, index=col_df)

