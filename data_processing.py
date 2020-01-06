import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
    path1 = './test_data/2018-01-15min.csv'
    path2 = './test_data/2018-02-15min.csv'
    path3 = './test_data/2018-03-15min.csv'
    path4 = './test_data/2018-04-15min.csv'
    path5 = './test_data/2018-05-15min.csv'
    path6 = './test_data/2018-06-15min.csv'
    path_meta = './meta_data/metadata.csv'

    df1 = pd.read_csv(path5)
    # df2 = pd.read_csv(path)

    # the month list:
    num_of_month = 6 - 1

    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']

    # read the meta information
    meta_data = pd.read_csv(path_meta)

    # calculate the whole information
    df1_1 = df1[['dataid', 'local_15min', 'grid', 'solar', 'solar2', 'shed1']]
    meta_data_1 = meta_data[['dataid', 'city']]
    df1_1['use'] = df1_1['grid'] + df1_1['solar'].fillna(value=0) + df1_1['solar2'].fillna(value=0) \
                   + df1_1['shed1'].fillna(value=0)
    df1_merge = pd.merge(df1_1, meta_data, how='inner', on=['dataid'])
    df1_merge.to_csv('./processed_data/merge-2014-01-15min.csv')

    metacity_info = meta_data['city'].value_counts()
    metacity_info.to_csv('./processed_data/metacity_info.csv')

    City_df = []
    # typical city and state
    # Texas
    # Austin 956
    # actual 256 ok
    df1_Austin = df1_merge[df1_merge['city'] == 'Austin']
    City_df.append(df1_Austin)
    # Houston 88
    # actual 4
    df1_Houston = df1_merge[df1_merge['city'] == 'Houston']

    # Dallas
    # actual 3
    df1_Dallas = df1_merge[df1_merge['city'] == 'Dallas']

    # Plano
    # actual 2
    df1_Plano = df1_merge[df1_merge['city'] == 'Plano']

    # Spring
    # actual 1
    df1_Spring = df1_merge[df1_merge['city'] == 'Spring']

    # Fort Worth
    # actual 1
    df1_Fort_Worth = df1_merge[df1_merge['city'] == 'Fort Worth']

    # Colorado
    # Boulder 58
    # actual 21 ok
    df1_Boulder = df1_merge[df1_merge['city'] == 'Boulder']
    City_df.append(df1_Boulder)
    # California
    # San Diego 57
    # actual 3
    df1_San_Diego = df1_merge[df1_merge['city'] == 'San Diego']

    # City name
    City_name = ['Austin', 'Boulder']

    df_2D = []
    df_avg = []
    df_dataid_columns = []

    # ------------------------------------
    # processing the raw data
    # ------------------------------------
    for city_name, df in zip(City_name, City_df):

        df_extracted = df[['dataid', 'local_15min', 'use']]
        print('the {} city user num = {}'.format(city_name, df_extracted['dataid'].value_counts().shape[0]))

        df_extracted['date_time'] = pd.to_datetime(df_extracted['local_15min'])
        df_extracted['date'] = df_extracted['date_time'].apply(lambda x: x.date())
        df_extracted['time'] = df_extracted['date_time'].apply(lambda x: x.time())
        df_extracted = df_extracted.sort_values(by=['dataid', 'date', 'time'])

        framed_df = pd.Series(df_extracted['use'].to_numpy(), index=[df_extracted['dataid'],
                                                                     df_extracted['local_15min']])
        # the first fillna the reframed process na
        # frame 2 include the avg of all users
        framed2_df = framed_df.unstack(level=0)

        # 　drop the columns with nan
        if city_name == 'Austin':
            framed2_df = framed2_df.drop(columns=[2233, 2361, 6121, 3778, 8142, 9938, 5109])
        if city_name == 'Boulder':
            framed2_df = framed2_df.drop(columns=[2824, 5187])
        framed2_df['avg'] = framed2_df.mean(axis=1, skipna=True)
        framed2_df.to_csv('./processed_data/{}_in_{}.csv'.format(city_name, months[num_of_month]))

        # the second fillna the original data forward fill
        framed3_df = framed2_df.fillna(method='ffill', axis=1)
        framed3_df = framed3_df.fillna(method='bfill', axis=1)
        framed3_df = framed3_df.dropna(how='all', axis=1)
        # framed3_df = framed2_df.fillna(value=0)

        # drop the last index value
        framed3_df = framed3_df.drop(framed3_df.tail().index[-1])
        framed3_df.to_csv('./fillna_with_0/{}_in_{}_no_na.csv'.format(city_name, months[num_of_month]))
        df_dataid_columns.append(framed3_df)

        # fill the nan
        framed_df_np = framed3_df.to_numpy()
        average_the_dataid = []
        # document the value
        df_2D.append(framed_df_np)

        # utilize framed3 for generation in this part

        # ------------------------------------
        # do average operation in the whole month
        # ------------------------------------
        # average according to the dataid in the whole month
        for k in range(framed_df_np.shape[1]):
            data_month = framed_df_np[:, k]
            data_month = data_month.reshape((96, -1))
            dataid_mean = data_month.mean(axis=1)
            average_the_dataid.append(dataid_mean)

        # plot the average picture of different month
        fig, ax = plt.subplots(1, 1)
        for avg in average_the_dataid:
            plt.plot(avg, alpha=0.5)
        plt.plot(average_the_dataid[-1], 'k-', linewidth=2, label=['average load'])
        plt.title('the average user profile of {} in {}'.format(city_name, months[num_of_month]))
        plt.xlabel('Time point')
        plt.ylabel('Load/kW')
        plt.grid('True')
        plt.savefig('./image/the_average_of_{}_in_{}.png'.format(months[num_of_month], city_name))
        plt.show()
        plt.close()

        # ------------------------------------
        # plot the specific day of different users
        # ------------------------------------
        day_of_month = 2
        day_points = 96
        start = (day_of_month - 1) * day_points
        end = day_of_month * day_points
        num_user = framed_df_np.shape[1]

        fig, ax = plt.subplots(1, 1)
        for i in range(num_user - 1):
            plt.plot(framed_df_np[start:end, i], alpha=0.5)
        plt.plot(framed_df_np[start:end, num_user - 1], 'k-', linewidth=2, label=['average load'])
        plt.title('the user profile of {} in {} {}'.format(city_name, months[num_of_month], day_of_month))
        plt.xlabel('Time point')
        plt.ylabel('Load/kW')
        plt.grid('True')
        plt.savefig('./image/the_{}_day_of_{}_in_{}.png'.format(day_of_month, months[num_of_month], city_name))
        plt.show()
        plt.close()

    # ------------------------------------
    # prepared generate proper data profile 96
    # ------------------------------------
    k = 0
    for df_np in df_2D:
        num_of_user = df_np.shape[1]
        df_np_flatten = df_np.flatten()
        df_labels = pd.cut(df_np_flatten, bins=96, labels=np.arange(96), right=False)
        df_labels = np.reshape(df_labels, (96 * 30, -1))
        df_labels = df_labels.transpose()
        df_labels = df_labels.reshape((num_of_user, -1, 96))
        df_labels = df_labels.reshape((-1, 96))
        df_one_hot = np.zeros((num_of_user * 30, 96, 96))
        for i in range(df_labels.shape[0]):
            for j in range(df_labels.shape[1]):
                for m in range(df_labels[i, j]):
                    df_one_hot[i, j, m] = 1
        np.save('./df_one_hot/{}_one_hot'.format(City_name[k]), df_one_hot)
        k = k + 1
    del k

    # ------------------------------------
    # do KMeans classification
    # ------------------------------------

    kmeans_models = []
    kmeans_labels = []

    n_clusters = 5
    colors = ['C0', 'C1', 'C2', 'C3', 'C3', 'C4', 'C5', 'C6', 'C7', 'C9']
    colors_classic = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '']
    colors_html = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
                   '#7f7f7f', '#bcbd22', '#17becf']

    for df_np in df_2D:
        df_np_t = np.transpose(df_np)
        kmeans_models.append(KMeans(n_clusters=n_clusters, random_state=0).fit(df_np_t))
    for k_model in kmeans_models:
        kmeans_labels.append(k_model.labels_)

    # attach the label with the dataid
    label_id = []
    for df, labels in zip(df_dataid_columns, kmeans_labels):
        label_id.append(pd.Series(labels, index=df.columns, name='labels'))
    label_pd = pd.concat([label_id[0], label_id[1]])
    label_pd.drop(labels=['avg'])
    label_pd = label_pd.rename('labels')
    meta_data = pd.merge(meta_data, label_pd, how='left', on=['dataid'])
    meta_data.to_csv('./processed_meta_data/processed_metadata_{}.csv'.format(months[num_of_month]))

    # plot different classes the special data
    for labels, city_name, df in zip(kmeans_labels, City_name, df_2D):

        framed_df_np = df

        day_of_month = 2
        day_points = 96
        start = (day_of_month - 1) * day_points
        end = day_of_month * day_points
        num_user = framed_df_np.shape[1]

        fig, ax = plt.subplots(1, 1)
        for i in range(num_user - 1):
            plt.plot(framed_df_np[start:end, i], colors[labels[i]], alpha=0.5)
        plt.title('the classified user profile of {} in {} {}'.format(city_name, months[num_of_month], day_of_month))
        plt.xlabel('Time point')
        plt.ylabel('Load/kW')
        plt.grid('True')
        plt.savefig('./image/the_classified_{}_day_of_{}_in_{}.png'.format(day_of_month,
                                                                           months[num_of_month], city_name))
        plt.show()
        plt.close()

    # ------------------------------------
    # analysis the label's relative coefficient with other features, different months show different features????
    # ------------------------------------
    meta_data_extracted = meta_data[meta_data['labels'].notnull()]
    meta_data_extracted_corr_pearson = meta_data_extracted.corr(method='pearson')['labels']
    meta_data_extracted_corr_kendall = meta_data_extracted.corr(method='kendall')['labels']
    meta_data_extracted_corr_spearman = meta_data_extracted.corr(method='spearman')['labels']
    # the top coefficients
    # number_of_nests: -0.282122
    # amount_of_west_facing_pv:-0.230524
    # half_floor_square_footage:-0.199926
    # amount_of_east_facing_pv:-0.140372
    # total_amount_of_pv: -0.110623
    # lower_level_square_footage: 0.108677
