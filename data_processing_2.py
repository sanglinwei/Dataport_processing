import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == '__main__':
    path1 = './test_data/2014-01-15min.csv'
    path2 = './test_data/2014-02-15min.csv'
    path3 = './meta_data/metadata.csv'

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    meta_data = pd.read_csv(path3)

    df1_1 = df1[['dataid', 'local_15min', 'grid', 'solar']]

    df1_merge = pd.merge(df1_1, meta_data, how='inner', on=['dataid'])
    df1_merge.to_csv('./processed_data/merge-2014-01-15min.csv')

    metacity_info = meta_data['city'].value_counts()
    metacity_info.to_csv('./processed_data/metacity_info.csv')

    City_df = []
    # typical city and state
    # Texas
    # Austin 956
    df1_Austin = df1_merge[df1_merge['city'] == 'Austin']
    City_df.append(df1_Austin)
    # Houston 88
    df1_Houston = df1_merge[df1_merge['city'] == 'Houston']
    City_df.append(df1_Houston)
    # New York
    # Ithaca 88
    df1_Ithaca = df1_merge[df1_merge['city'] == 'Ithaca']
    City_df.append(df1_Ithaca)
    # Colorado
    # Boulder 58
    df1_Boulder = df1_merge[df1_merge['city'] == 'Boulder']
    City_df.append(df1_Boulder)
    # California
    # San Diego 57
    df1_San_Diego = df1_merge[df1_merge['city'] == 'San Diego']
    City_df.append(df1_San_Diego)

    # City name
    City_name = ['Austin', 'Houston', 'Ithaca', 'Boulder', 'San Diego']

    # plot all the possible results
    for df, city_name in zip(City_df[4], City_name[4]):
        df_extracted = df[['dataid', 'local_15min', 'grid_x']]

        df_extracted['date_time'] = pd.to_datetime(df_extracted['local_15min'])
        df_extracted['date'] = df_extracted['date_time'].apply(lambda x: x.date())
        df_extracted['time'] = df_extracted['date_time'].apply(lambda x: x.time())
        df_extracted = df_extracted.sort_values(by=['dataid', 'date', 'time'])

        framed_df = pd.Series(df_extracted['grid_x'].to_numpy(), index=[df_extracted['dataid'],
                                                                        df_extracted['local_15min']])
        framed_df = framed_df.unstack()
        framed_df['avg'] = framed_df.mean(axis=1, skipna=True)
        framed_df.head()

        # plot the comparison pic
        framed_df_np = framed_df.to_numpy()
        day_of_month = 1
        start = (day_of_month - 1) * 96
        end = day_of_month * 96
        num_user = framed_df_np.shape[1]
        print('the number of users {}'.format(num_user))

        fig, ax = plt.subplots(1, 1)
        ax.plot(framed_df_np[start:end, num_user-1], 'k-', linewidth=2, label='average load')
        for i in range(num_user-1):
            ax.plot(framed_df_np[start:end, i], alpha=0.5)
        plt.title('the user profile of {} in January {}'.format(city_name, day_of_month))
        plt.xlabel('Time point')
        plt.ylabel('Load/kW')
        plt.grid('True')
        plt.savefig('./image/the_{}_day_of_Jan_in_{}'.format(day_of_month, city_name))
        plt.close()





