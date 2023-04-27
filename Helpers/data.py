import pandas as pd
import numpy as np

class Read():
    def read_prices_volume():
        daily_data = pd.read_csv('Data/Prices and Volume/data.csv', index_col=0)
        daily_data = daily_data.replace('—', np.nan)
        daily_data = daily_data.astype('float64')
        daily_data.index = pd.to_datetime(daily_data.index, format = '%d/%m/%y %H:%M')
        daily_data['log_return'] = np.log(daily_data['Weighted Price']) - np.log(daily_data['Weighted Price'].shift(1))
        daily_data['log_vol'] = np.log(daily_data['Volume (BTC)'])
        daily_data = daily_data.dropna()
        return daily_data
    
    def read_search_queries():
        monthly_data = pd.read_csv('Data/Search Queries/multiTimeline.csv', skiprows=1, index_col=0).rename(columns={'Bitcoin: (Estados Unidos)': 'SQ'})
        monthly_data.index = pd.to_datetime(monthly_data.index, format = '%Y-%m')
        monthly_data = monthly_data.loc[monthly_data.index >= '2011-09-01']
        return monthly_data
    