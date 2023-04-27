import numpy as np
import pandas as pd

class Functions():
    def realized_volatility(returns, resampling):
        sq_ret = returns**2
        RV = sq_ret.resample(resampling).sum()
        RV = np.sqrt(RV)
        RV = pd.DataFrame(RV)
        return RV
    
    def estadisticas(datos):
        stats = list(datos.describe().loc[['mean','std','max', 'min']].values[:,0])+[datos.skew().values[0], datos.kurtosis().values[0]]
        return stats
    
    def consolidate_data(daily_data, log_SQ, log_RV):
        m_vols =  daily_data[['Volume (BTC)']].resample('M').sum()
        log_VO =  np.log(m_vols)
        m_prices = daily_data[['Weighted Price']].resample('M').last()
        log_R = pd.DataFrame(np.log(m_prices['Weighted Price']) - np.log(m_prices['Weighted Price'].shift(1)))
        all_data = pd.concat([log_SQ, log_VO, log_RV, log_R],axis = 1)
        all_data.SQ = all_data.SQ.ffill()
        all_data = all_data.dropna()
        all_data = all_data.rename(columns={'Volume (BTC)': 'VO', 'log_return': 'RV', 'Weighted Price': 'R'})
        return all_data