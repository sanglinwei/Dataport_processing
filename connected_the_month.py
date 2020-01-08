import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    city = ['Austin', 'Boulder']
    num_city = 1
    path_init = './processed_data_connected/Boulder/{}_in_{}_connected.csv'.format(city[1], months[0])
    df_connected_Austin = pd.read_csv(path_init)
    for num_mon in range(1, 6):
        path = './processed_data_connected/Boulder/{}_in_{}_connected.csv'.format(city[num_city], months[num_mon])
        df = pd.read_csv(path)
        df_connected_Austin = pd.concat([df_connected_Austin, df], ignore_index=True, axis=0, join='inner')
    df_connected_Austin = df_connected_Austin.set_index('local_15min')
    df_connected_Austin.to_csv('./processed_data_connected/accumulate_frame/Boulder_from_Jan_to_June.csv')