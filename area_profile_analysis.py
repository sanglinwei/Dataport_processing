import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == '__main__':
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    city = ['Austin', 'Boulder']
    num_city = 0

    for num_mon in range(0, 6):
        # ------------------------------------
        # read the data
        # ------------------------------------
        df_pro = pd.read_csv('./processed_data_connected/{}_in_{}_connected.csv'.format(city[num_city], months[num_mon]))
        df_pro = df_pro.set_index('local_15min')
        df_pro = df_pro.drop(['avg'], axis=1)

        # ------------------------------------
        # deal with meta data
        # ------------------------------------
        meta_data = pd.read_csv('./meta_data/metadata.csv')
        meta_data = meta_data[['dataid', 'total_square_footage']]
        col_df = df_pro.columns
        col_df = col_df.to_numpy()
        col_df = col_df.astype('int64')
        meta_data = meta_data.set_index('dataid')
        meta_data_extracted = pd.DataFrame(meta_data, index=col_df)
        # ------------------------------------
        # save the document data in dictionary with key as columns number
        # ------------------------------------
        dc = dict()
        dc_qu = dict()
        dc_len = dict()
        for col in df_pro.columns:
            df_user = df_pro[col].to_numpy().reshape((-1, 96))
            df_user_pd = pd.DataFrame(df_user, columns=np.arange(96))
            df_user_pd_qu = df_user_pd.quantile([0.25, 0.5, 0.75])
            df_user_pd_qu = df_user_pd_qu.T
            df_user_pd_qu['length'] = df_user_pd_qu[0.75] - df_user_pd_qu[0.25]
            dc[col] = df_user_pd
            dc_qu[col] = df_user_pd_qu
            dc_len[col] = df_user_pd_qu['length']
        df_sum_area = df_pro.sum(axis=1)
        df_sum_area = df_sum_area.to_numpy().reshape((-1, 96))
        df_sum_area_pd = pd.DataFrame(df_sum_area, columns=np.arange(96))
        df_sum_area_qu = df_sum_area_pd.quantile([0.25, 0.5, 0.75])
        df_sum_area_qu = df_sum_area_qu.T
        df_sum_area_qu['length'] = df_sum_area_qu[0.75] - df_sum_area_qu[0.25]

        # -----------------------------------------------------------------
        # set the criterion to choose the users and show the performance
        # need to consider the trade off between the uncertainty and the load power
        # the (kW)
        # the length choose is whether the good parameters???
        # -----------------------------------------------------------------
        critic_ls = [100, 10, 5, 2.5, 1, 0.5]
        for critic in critic_ls:
            # from the perspective of time scale
            selected_power = dict()
            ls = list()
            selected_flag = pd.DataFrame(np.zeros([96, df_pro.shape[1]]), index=np.arange(96), columns=df_pro.columns)
            for t in np.arange(96):
                selected_power[t] = pd.Series(np.zeros(df_sum_area.shape[0]))
                for col in df_pro.columns:
                    if dc_len[col][t] < critic:
                        selected_power[t] += dc[col][t]
                        selected_flag.loc[t, col] = True
                    else:
                        selected_flag.loc[t, col] = False
            selected_power_pd = pd.DataFrame(selected_power)

            # box plot the aggregated performance
            fig, ax = plt.subplots(1, 1)
            selected_power_pd.boxplot(column=list(np.arange(96)))
            plt.xlabel('time points')
            plt.ylabel('Power/kW')
            plt.title('the aggregated performance in boxplot with critic as {}'.format(critic))
            plt.savefig('./aggregated_image/{}_{}_aggregated/box_plot/the_aggregated_with_critic_as_{}.png'.
                        format(months[num_mon], city[num_city], critic))
            plt.close()

            # scatter plot the aggregated performance
            fig, ax = plt.subplots(1, 1)
            for j in range(selected_power_pd.shape[0]):
                plt.scatter(np.arange(96), selected_power_pd.iloc[j, :], c='k', alpha=0.3, s=16)
            plt.xlabel('time points')
            plt.ylabel('Power/kW')
            plt.title('the aggregated performance in scattered plot with critic as {}'.format(critic))
            plt.savefig('./aggregated_image/{}_{}_aggregated/scatter_plot/the_aggregated_with_critic_as_{}.png'.
                        format(months[num_mon], city[num_city], critic))
            plt.close()
            print('the num of month is {}'.format(months[num_mon]))
    # --------------------------------------------------------------------------
    # show the user's uncertainty changing during the day and the season
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # discuss the time resolution for the aggregated performance
    # --------------------------------------------------------------------------
    # discuss the number of users and the aggregated performance
    # --------------------------------------------------------------------------
    # discuss the satisfied performance
    # --------------------------------------------------------------------------




