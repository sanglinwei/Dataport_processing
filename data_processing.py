import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans


if __name__ == '__main__':
    path1 = './test_data/2018-01-15min.csv'
    path2 = './test_data/2018-02-15min.csv'
    path3 = './test_data/2018-03-15min.csv'
    path4 = './test_data/2018-04-15min.csv'
    path5 = './test_data/2018-05-15min.csv'
    path_meta = './meta_data/metadata.csv'

    df1 = pd.read_csv(path1)
    # df2 = pd.read_csv(path2)
    meta_data = pd.read_csv(path_meta)

    df1_1 = df1[['dataid', 'local_15min', 'grid', 'solar', 'solar2', 'shed1']]
    meta_data_1 = meta_data[['dataid', 'city']]
    df1_1['use'] = df1_1['grid'] + df1_1['solar'].fillna(value=0) + df1_1['solar2'].fillna(value=0)\
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
    # plot all the possible results
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
        framed2_df = framed_df.unstack(level=0, fill_value=0)
        framed2_df['avg'] = framed2_df.mean(axis=1, skipna=True)
        framed2_df.to_csv('./processed_data/{}_in_January.csv'.format(city_name))
        # the second fillna the original data na
        framed3_df = framed2_df.fillna(value=0)
        # drop the last index value
        framed3_df = framed3_df.drop(framed3_df.tail().index[-1])
        framed3_df.to_csv('./fillna_with_0/{}_in_January_no_na.csv'.format(city_name))

        # fill the nan
        framed_df_np = framed3_df.to_numpy()
        average_the_dataid = []

        # average according to the dataid
        for k in range(framed_df_np.shape[1]):
            data_month = framed_df_np[:, k]
            data_month = data_month.reshape((96, 30))
            dataid_mean = data_month.mean(axis=1)
            average_the_dataid.append(dataid_mean)

        # plot the average picture
        fig, ax = plt.subplots(1, 1)
        for avg in average_the_dataid:
            plt.plot(avg, alpha=0.5)
        plt.plot(average_the_dataid[-1], 'k-', linewidth=2, label=['average load'])
        plt.title('the average user profile of {} in January'.format(city_name))
        plt.xlabel('Time point')
        plt.ylabel('Load/kW')
        plt.grid('True')
        plt.savefig('./image/the_average_of_Jan_in_{}.png'.format(city_name))
        plt.show()
        plt.close()

        # plot the comparison picture
        df_2D.append(framed_df_np)
        day_of_month = 2
        day_points = 96
        start = (day_of_month - 1) * day_points
        end = day_of_month * day_points
        num_user = framed_df_np.shape[1]

        fig, ax = plt.subplots(1, 1)
        for i in range(num_user-1):
            plt.plot(framed_df_np[start:end, i], alpha=0.5)
        plt.plot(framed_df_np[start:end, num_user-1], 'k-', linewidth=2, label=['average load'])
        plt.title('the user profile of {} in January {}'.format(city_name, day_of_month))
        plt.xlabel('Time point')
        plt.ylabel('Load/kW')
        plt.grid('True')
        plt.savefig('./image/the_{}_day_of_Jan_in_{}.png'.format(day_of_month, city_name))
        plt.show()
        plt.close()

    # plot the average


    # plot the classfying results
    kmeans_models = []
    kmeans_labels = []
    for df_np in df_2D:
        df_np_t = np.transpose(df_np)
        kmeans_models.append(KMeans(n_clusters=2, random_state=0).fit(df_np_t))
    for k_model in kmeans_models:
        kmeans_labels.append(k_model.labels_)

    # plot different class
    for labels, city_name, df in zip(kmeans_labels, City_name, df_2D):

        framed_df_np = df

        day_of_month = 2
        day_points = 96
        start = (day_of_month - 1) * day_points
        end = day_of_month * day_points
        num_user = framed_df_np.shape[1]

        fig, ax = plt.subplots(1, 1)
        for i in range(num_user - 1):
            if labels[i] == 1:
                plt.plot(framed_df_np[start:end, i], 'g-', alpha=0.5)
            if labels[i] == 0:
                plt.plot(framed_df_np[start:end, i], 'b-', alpha=0.5)
        # plt.plot(framed_df_np[start:end, num_user - 1], 'k-', linewidth=2, label=['average load'])
        plt.title('the classified user profile of {} in January {}'.format(city_name, day_of_month))
        plt.xlabel('Time point')
        plt.ylabel('Load/kW')
        plt.grid('True')
        plt.savefig('./image/the_classified_{}_day_of_Jan_in_{}.png'.format(day_of_month, city_name))
        plt.show()
        plt.close()





