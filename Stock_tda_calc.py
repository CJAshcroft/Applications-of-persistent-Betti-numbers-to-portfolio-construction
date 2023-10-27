import numpy as np
import pandas as pd

import os
from tqdm import tqdm


from ripser import ripser
from persim import plot_diagrams
from persim.landscapes import (
    PersLandscapeApprox,
    average_approx,
    snap_pl,
    plot_landscape,
    plot_landscape_simple
)
import gudhi as gd

from return_utils import collecting_stock_percentages, collecting_stock_data_avg


s_and_p_all = pd.read_csv(
    'sp500_tickers.csv').iloc[:, 1]
s_and_p_all = list(s_and_p_all.map(
    lambda x: x.replace('\n', '')).values)
post_2017 = ['KDP', 'FOX', 'FOXA', 'WBD', 'AMCR', 'ON', 'VICI', 'CPT', 'MOH',
             'NDSN', 'CEG', 'SBNY', 'SEDG', 'FDS', 'EPAM', 'MTCH', 'CDAY',
             'BRO', 'TECH', 'MRNA', 'OGN', 'CRL', 'PTC', 'NXPI', 'PENN',
             'GNRC', 'CZR', 'MPWR', 'TRMB', 'ENPH', 'TSLA', 'VNT', 'POOL',
             'ETSY', 'TER', 'CTLT', 'BIO', 'TDY', 'TYL', 'WST', 'DPZ',
             'DXCM', 'OTIS', 'CARR', 'HWM', 'IR', 'PAYC', 'LYV', "ZBRA",
             'STE', 'ODFL', 'WRB', 'NOW', 'LVS', 'NVR', 'CDW', 'LDOS', 'IEX',
             'TMUS', 'MKTX', 'AMCRT', 'DD', 'CTVA', 'DOW', 'WAB', 'ATO', 'TFX',
             'FRC', 'CE', 'LW', 'MXIM', 'FANG', 'JKHY', 'KEYS', 'FTNT',
             'ROL', 'WCG', 'ANET', 'CPRT', 'FLT', 'BR', 'HFC', "TWTR",
             'EVRG', 'ABMD', 'MSCI', 'TTWO', 'SIVB', 'NKTR', 'IPGP', 'HII',
             'NCLH', 'CDNS', 'DWDP', 'SBAC', 'Q', 'BHF', 'DRE', 'AOS', 'PKG',
             'RMD', 'MGM', 'HLT', 'ALGN', 'ANSS', 'RE', 'INFO', 'BMS']
s_and_p = list(set(s_and_p_all)-set(post_2017))


def time_series_pers_p_norm(wind, dim, pn):
    '''
    time_series_pers_p_norm

    inputs:
        wind: array or list type
            containing window from which we calculate persistance landscape p-norm

        dim: int
            embedding dim

        pn: int
            p-norm we wish to calculate

    returns:
        norm: float
            p norm of the window
    '''
    if any(wind.isna()) or len(wind) % dim != 0:
        return(np.nan)
    else:
        wind_array = np.reshape(np.asarray(wind), (int(len(wind)/dim), dim))
        pers_digms = ripser(wind_array, maxdim=3)['dgms']
        if not np.any(pers_digms[1]):
            norm = 0
        else:
            allowed_ = np.repeat(False, pers_digms[1].shape[0])
            for i in range(0, 1):
                allowed_[i] = (pers_digms[1][i][1] -
                               pers_digms[1][i][0]) > 1e-5
            allowed_indxs = [i for i, x in enumerate(allowed_) if x]
            if(len(allowed_indxs) == 0):
                return 0
            pers_digms[1] = pers_digms[1][allowed_indxs]
            pers = PersLandscapeApprox(dgms=pers_digms, hom_deg=1)
            norm = pers.p_norm(pn)
        return(norm)


def tda_pers_stock(stock, dim, pn):
    '''
    tda_pers_stock

    inputs:

        stock: float
            ticker of stock, stock dataframe will be located at relative path
            sp500/stock_dataframes/{stock}.csv

        dim: int
            embedding dims to apply

        pns: int
            p norm we wish to calculate

    returns:
        none
        calculates persistence norms and saves to 'Persim_vs_Betti/sp500/{stock}.csv'


    '''
    if not os.path.exists(f'Persim_vs_Betti/landscape/{stock}.csv'
                          ):
        df = pd.read_csv(
            f'sp500/stock_dataframes/{stock}.csv', index_col='Date')
        df['log_returns'] = np.log(
            df['Adj Close']/df['Adj Close'].shift(1))
        tda_df = df[['log_returns']].rolling(
            window=7*dim).apply(lambda x: time_series_pers_p_norm(x, dim, pn))
        tda_df.index = df.index.values
        tda_df.columns = ['Persistence landscape norm']
        df = pd.concat([df, tda_df], axis=1)
        df.to_csv(f'Persim_vs_Betti/landscape/{stock}.csv')


def unweighted_time_series_avg_betti(wind, dim):
    '''
    unweighted_time_series_avg_betti

    inputs:
        wind: array or list type
            containing window from which we calculate persistance landscape one norm

        dim: int
            embedding dim

    returns:
        bets/count: average persistent betti number
    '''
    if any(wind.isna()) or len(wind) % dim != 0:
        return(np.nan)
    else:
        wind_array = np.reshape(np.asarray(wind), (int(len(wind)/dim), dim))
        ac = gd.AlphaComplex(points=wind_array)
    stree = ac.create_simplex_tree()
    diag = stree.compute_persistence()
    filt = list(dict.fromkeys([x[1] for x in stree.get_filtration()]))
    bets = 0
    count = 0
    for i in range(0, len(filt)):
        for j in range(i+1, len(filt)):
            count += 1
            bets += sum(stree.persistent_betti_numbers(
                filt[i], filt[j]))
    return(bets/count)


def betti_stock(stock, bnh, dim, points, datpath, start='2017-08-03', end='2022-12-31'):
    '''
    tda_pers_stock

    inputs:

        stock: string
            ticker of stock, stock dataframe will be located at relative path
            sp500/Persim_vs_Betti/sp500/{stock}.csv

        bnh

        dim: int
            embedding dims to apply

        points: int
            number of points we have in R^{dims}

    outputs:
        none
        calculates persistence betti numbers and saves to f'{datpath}/betti_dim_{dim}_points_{points}/{stock}.csv'


    '''
    if not os.path.exists(f'{datpath}/betti_dim_{dim}_points_{points}/{stock}.csv'):
        df = pd.read_csv(
            f'sp500/stock_dataframes/{stock}.csv', index_col='Date').loc[start:end]

        df.index = pd.to_datetime(df.index.values)
        df['% returns'] = df['Adj Close']/df['Adj Close'].shift(1) - 1
        df['% relative returns'] = df['% returns'] - \
            bnh[bnh.index.isin(df.index)]
        tda_df = df[['% returns']].rolling(window=points*dim).apply(
            lambda x: unweighted_time_series_avg_betti(x, dim))
        tda_df.columns = ['Avg_%_Betti']
        tda_df2 = df[['% relative returns']].rolling(window=points*dim).apply(
            lambda x: unweighted_time_series_avg_betti(x, dim))
        tda_df2.columns = ['Avg_rel_%_Betti']
        df = pd.concat([df, tda_df, tda_df2], axis=1)
        df.to_csv(
            f'{datpath}/betti_dim_{dim}_points_{points}/{stock}.csv')


if __name__ == '__main__':
    #    for t in tqdm(s_and_p):
    #        tda_pers_stock(stock=t, dim=3, pn=7)
    print(s_and_p)
    dimlow = 2
    dimup = 6
    pointlow = 5
    pointup = 9
    datpath = 'Betti_experiments'
    stock_closes_df = collecting_stock_data_avg(tickers=s_and_p)
    stock_perc_df = collecting_stock_percentages(s_and_p)
    stock_perc_df.index = pd.to_datetime(
        stock_perc_df.index.values, format='%Y-%m-%d')
    stock_perc_df.fillna(stock_perc_df.mean(axis=1), inplace=True)
    bnh = stock_perc_df.mean(axis=1)
    bnh.index = pd.to_datetime(stock_perc_df.index)
    for d in range(dimlow, dimup+1):
        for point in range(pointlow, pointup+1):
            print(f'dim {d} points {point}')
            if not os.path.exists(f'{datpath}/betti_dim_{d}_points_{point}'):
                os.makedirs(
                    f'{datpath}/betti_dim_{d}_points_{point}')
            for t in tqdm(s_and_p):
                if not os.path.exists(
                    f'{datpath}/betti_dim_{d}_points_{point}/{t}.csv'
                ):
                    betti_stock(t, bnh=bnh, dim=d,
                                points=point, datpath=datpath)
