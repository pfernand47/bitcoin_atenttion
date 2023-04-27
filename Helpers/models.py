from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import numpy as np

class Models():
    def modelos(all_data, lags):
        # Modelo 1 
        model_1 = VAR(all_data[['RV', 'SQ']], freq='M')
        # Modelo 2
        model_2 = VAR(all_data[['SQ', 'VO']], freq='M')
        # Modelo 3
        model_3 = VAR(all_data[['R', 'SQ']], freq='M')
        if lags == []: # escoger los lags del modelo basados en el BIC
            results_1 = model_1.fit(maxlags=7, ic='bic')
            lags_mod1 = results_1.k_ar
            results_2 = model_2.fit(maxlags=7, ic='bic')
            lags_mod2 = results_2.k_ar
            results_3 = model_3.fit(maxlags=7, ic='bic')
            lags_mod3 = results_3.k_ar
        else: # lags predeterminados que vienen dados en el vector lags
            lags_mod1, lags_mod2, lags_mod3 = lags
            results_1 = model_1.fit(lags_mod1)
            results_2 = model_2.fit(lags_mod2)
            results_3 = model_3.fit(lags_mod3)
        return results_1, results_2, results_3, lags_mod1, lags_mod2, lags_mod3 
    
    def tabla_resultados(results_1, results_2, results_3, lags_mod1, lags_mod2, lags_mod3):
        # Codigos de significancia de los coeficientes de los modelos
        codes_1 = results_1.pvalues.apply(lambda x: sig_codes(x))
        const_cod_1 = sig_codes(results_1.pvalues_dt[0])
        codes_2 = results_2.pvalues.apply(lambda x: sig_codes(x))
        const_cod_2 = sig_codes(results_2.pvalues_dt[0])
        codes_3 = results_3.pvalues.apply(lambda x: sig_codes(x))
        const_cod_3 = sig_codes(results_3.pvalues_dt[0])
        max_sq = max(lags_mod1, lags_mod2, lags_mod3)
        SQ_lags = ['SQ_t-{}'.format(i+1) for i in range(max_sq)]
        RV_lags = ['RV_t-{}'.format(i+1) for i in range(lags_mod1)]
        VO_lags = ['VO_t-{}'.format(i+1) for i in range(lags_mod2)]
        R_lags = ['R_t-{}'.format(i+1) for i in range(lags_mod3)]
        tabla = pd.DataFrame(' ', index = SQ_lags + RV_lags + VO_lags + R_lags + ['Constant'] , columns = np.zeros(6))
        columns=[('Model 1','RV_t'),('Model 1','SQ_t'), ('Model 2','SQ_t'),('Model 2','VO_t'), ('Model 3','R_t'),('Model 3','SQ_t')]
        tabla.columns=pd.MultiIndex.from_tuples(columns)
        for lag in range(lags_mod1):
            tabla.loc['SQ_t-{}'.format(lag+1), ('Model 1','RV_t')] = str(round(results_1.coefs[lag][0,1],4)) + codes_1.loc['L{}.SQ'.format(lag+1), 'RV']
            tabla.loc['SQ_t-{}'.format(lag+1), ('Model 1','SQ_t')] = str(round(results_1.coefs[lag][1,1],4)) + codes_1.loc['L{}.SQ'.format(lag+1), 'SQ']
            tabla.loc['RV_t-{}'.format(lag+1), ('Model 1','RV_t')] = str(round(results_1.coefs[lag][0,0],4)) + codes_1.loc['L{}.RV'.format(lag+1), 'RV']
            tabla.loc['RV_t-{}'.format(lag+1), ('Model 1','SQ_t')] = str(round(results_1.coefs[lag][1,0],4)) + codes_1.loc['L{}.RV'.format(lag+1), 'SQ']
        for lag in range(lags_mod2):
            tabla.loc['VO_t-{}'.format(lag+1), ('Model 2','SQ_t')] = str(round(results_2.coefs[lag][0,1],4)) + codes_2.loc['L{}.VO'.format(lag+1), 'SQ']
            tabla.loc['VO_t-{}'.format(lag+1), ('Model 2','VO_t')] = str(round(results_2.coefs[lag][1,1],4)) + codes_2.loc['L{}.VO'.format(lag+1), 'VO']
            tabla.loc['SQ_t-{}'.format(lag+1), ('Model 2','SQ_t')] = str(round(results_2.coefs[lag][0,0],4)) + codes_2.loc['L{}.SQ'.format(lag+1), 'SQ']
            tabla.loc['SQ_t-{}'.format(lag+1), ('Model 2','VO_t')] = str(round(results_2.coefs[lag][1,0],4)) + codes_2.loc['L{}.SQ'.format(lag+1), 'VO']
        for lag in range(lags_mod3):
            tabla.loc['SQ_t-{}'.format(lag+1), ('Model 3','R_t')] = str(round(results_3.coefs[lag][0,1],4)) + codes_3.loc['L{}.SQ'.format(lag+1), 'R']
            tabla.loc['SQ_t-{}'.format(lag+1), ('Model 3','SQ_t')] = str(round(results_3.coefs[lag][1,1],4)) + codes_3.loc['L{}.SQ'.format(lag+1), 'SQ']
            tabla.loc['R_t-{}'.format(lag+1), ('Model 3','R_t')] = str(round(results_3.coefs[lag][0,0],4)) + codes_3.loc['L{}.R'.format(lag+1), 'R']
            tabla.loc['R_t-{}'.format(lag+1), ('Model 3','SQ_t')] = str(round(results_3.coefs[lag][1,0],4)) + codes_3.loc['L{}.R'.format(lag+1), 'SQ']
        tabla.loc['Constant', ('Model 1','RV_t')] = str(round(results_1.intercept[0],4)) + const_cod_1[0]
        tabla.loc['Constant', ('Model 1','SQ_t')] = str(round(results_1.intercept[1],4)) + const_cod_1[1]
        tabla.loc['Constant', ('Model 2','SQ_t')] = str(round(results_2.intercept[0],4)) + const_cod_2[0]
        tabla.loc['Constant', ('Model 2','VO_t')] = str(round(results_2.intercept[1],4)) + const_cod_2[1]
        tabla.loc['Constant', ('Model 3', 'R_t')] = str(round(results_3.intercept[0],4)) + const_cod_3[0]
        tabla.loc['Constant', ('Model 3', 'SQ_t')] = str(round(results_3.intercept[1],4)) + const_cod_3[1]
        return tabla

    def granger_test(all_data, max_lag):
        # Test de granger. H0: serie x no causa-Granger a y 
        granger = pd.DataFrame()
        tests = {
            'RV does not Granger Cause SQ': ['SQ', 'RV'],
            'Volume does not Granger Cause SQ': ['SQ', 'VO'],
            'Returns does not Granger Cause SQ': ['SQ', 'R'],
            'SQ does not Granger Cause RV' : ['RV', 'SQ'],
            'SQ does not Granger Cause Volume': ['VO','SQ'],
            'SQ does not Granger Cause Returns': ['R', 'SQ']
        }
        for test in tests.keys():
            names = tests[test]
            g = grangercausalitytests(all_data[names], maxlag=[max_lag], verbose = False)
            granger.loc[test, 'F'] = g[max_lag][0]['ssr_ftest'][0]
            granger.loc[test, 'p-value'] = g[max_lag][0]['ssr_ftest'][1]
        granger['Significancia'] = sig_codes(granger['p-value'])
        return granger

def sig_codes(x):
    codes = []
    for i in x:
        if i < 0.1:
            if i < 0.05:
                codes+= ['**']
            else:
                if i < 0.01:
                    codes+= ['***']
                else:
                    codes+= ['*']
        else:
            codes+= [' ']
    return codes