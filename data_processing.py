import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == '__main__':
    path1 = './test_data/2014-01-15min.csv'
    path2 = './test_data/2014-02-15min.csv'
    path3 = './meta_data/metadata.csv'

    df1 = pd.read_csv(path1)
    # df2 = pd.read_csv(path2)
    meta_data = pd.read_csv(path3)

    df1_1 = df1[['dataid', 'local_15min', 'grid', 'solar', 'solar2']]
    meta_data_1 = meta_data[['dataid', 'city']]
    df1_1['use'] = df1_1['grid'] + df1_1['solar'].fillna(value=0) + df1_1['solar2'].fillna(value=0)

    df1_merge = pd.merge(df1_1, meta_data, how='inner', on=['dataid'])
    df1_merge.to_csv('./processed_data/merge-2014-01-15min.csv')

    metacity_info = meta_data['city'].value_counts()
    metacity_info.to_csv('./processed_data/metacity_info.csv')

    City_df = []
    # typical city and state
    # Texas
    # Austin 956
    # actual 20
    df1_Austin = df1_merge[df1_merge['city'] == 'Austin']
    City_df.append(df1_Austin)
    # Houston 88
    # actual 15
    df1_Houston = df1_merge[df1_merge['city'] == 'Houston']
    City_df.append(df1_Houston)

    # New York
    # Ithaca 88
    # actual 0
    df1_Ithaca = df1_merge[df1_merge['city'] == 'Ithaca']
    City_df.append(df1_Ithaca)
    # Colorado
    # Boulder 58
    # actual 0
    df1_Boulder = df1_merge[df1_merge['city'] == 'Boulder']
    City_df.append(df1_Boulder)
    # California
    # San Diego 57
    # actual 19
    df1_San_Diego = df1_merge[df1_merge['city'] == 'San Diego']
    City_df.append(df1_San_Diego)

    # City name
    City_name = ['Austin', 'Houston', 'Ithaca', 'Boulder', 'San Diego']

    # plot all the possible results
    for city_name, df in zip(City_name, City_df):

        df_extracted = df[['dataid', 'local_15min', 'use']]

        df_extracted['date_time'] = pd.to_datetime(df_extracted['local_15min'])
        df_extracted['date'] = df_extracted['date_time'].apply(lambda x: x.date())
        df_extracted['time'] = df_extracted['date_time'].apply(lambda x: x.time())
        df_extracted = df_extracted.sort_values(by=['dataid', 'date', 'time'])

        framed_df = pd.Series(df_extracted['use'].to_numpy(), index=[df_extracted['dataid'],
                                                                        df_extracted['local_15min']])
        framed2_df = framed_df.unstack(level=0, fill_value=0)
        framed2_df['avg'] = framed2_df.mean(axis=1, skipna=True)

        # plot the comparison pic
        framed_df_np = framed2_df.to_numpy()
        day_of_month = 1
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
        plt.savefig('./image/the_{}_day_of_Jan_in_{}.pdf'.format(day_of_month, city_name))
        plt.show()
        plt.close()





