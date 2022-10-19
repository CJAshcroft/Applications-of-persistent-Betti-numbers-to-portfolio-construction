import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm
from return_utils import collecting_stock_percentages,collecting_stock_data_avg
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
#from concurrent import futures
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
import os

s_and_p = pd.read_csv('csvs/sp500/sp500_tickers.csv').iloc[:, 1]
s_and_p = list(s_and_p.map(
    lambda x: x.replace('\n', '')).values)
post_2017 = ['KDP', 'FOXA', 'ON', 'VICI', 'CPT', 'MOH', 'NDSN', 'CEG', 'SBNY', 'SEDG', 'FDS', 'EPAM', 'MTCH', 'CDAY', 'BRO', 'TECH', 'MRNA', 'OGN', 'CRL', 'PTC', 'NXPI', 'PENN', 'GNRC', 'CZR', 'MPWR', 'TRMB', 'ENPH', 'TSLA', 'VNT', 'POOL', 'ETSY', 'TER', 'CTLT', 'BIO', 'TDY', 'TYL', 'WST', 'DPZ', 'DXCM', 'OTIS', 'CARR', 'HWM', 'IR', 'PAYC', 'LYV', "ZBRA", 'STE', 'ODFL', 'WRB', 'NOW', 'LVS',
             'NVR', 'CDW', 'LDOS', 'IEX', 'TMUS', 'MKTX', 'AMCRT', 'DD', 'CTVA', 'DOW', 'WAB', 'ATO', 'TFX', 'FRC', 'CE', 'LW', 'MXIM', 'FANG', 'JKHY', 'KEYS', 'FTNT', 'ROL', 'WCG', 'ANET', 'CPRT', 'FLT', 'BR', 'HFC', "TWTR", 'EVRG', 'ABMD', 'MSCI', 'TTWO', 'SIVB', 'NKTR', 'IPGP', 'HII', 'NCLH', 'CDNS', 'DWDP', 'SBAC', 'Q', 'BHF', 'DRE', 'AOS', 'PKG', 'RMD', 'MGM', 'HLT', 'ALGN', 'ANSS', 'RE', 'INFO']
s_and_p = list(set(s_and_p)-set(post_2017))

if __name__ == '__main__':
    shortwindow = 21
    longwindow = 60
    pn = 3


# def download_stock(stock):
#     """ try to query the iex for a stock, if failed note with print """
#     try:
#         pd.read_csv(f'stock dataframes/{stock}_data.csv')
#     except:
#         try:
#             print(stock)
#             stock_df = web.DataReader(stock, 'yahoo', start_time, now_time)
#             stock_df['Name'] = stock
#             stock_df.to_csv('stock dataframes/{}_data.csv'.format(stock))
#         except:
#             bad_names.append(stock)
#             print('bad: %s' % (stock))


# dl = False
# if __name__ == '__main__' and dl == True:

#     """ set the download window """
#       bad_names = []
#       start_time = datetime(2017, 6, 1)
#       now_time = datetime(2022, 6, 1)

#     bad_names = []  # to keep track of failed queries

#     """here we use the concurrent.futures module's ThreadPoolExecutor
# 		to speed up the downloads buy doing them in parallel
# 		as opposed to sequentially """

#     # set the maximum thread number
#     max_workers = 50

#     # in case a smaller number of stocks than threads was passed in
#     workers = min(max_workers, len(s_and_p))
#     with futures.ThreadPoolExecutor(workers) as executor:
#         res = executor.map(download_stock, s_and_p)

#     """ Save failed queries to a text file to retry """
#     if len(bad_names) > 0:
#         with open('failed_queries.txt', 'w') as outfile:
#             for name in bad_names:
#                 outfile.write(name+'\n')

#     # timing:
#     finish_time = datetime.now()
#     duration = finish_time - now_time
#     minutes, seconds = divmod(duration.seconds, 60)
#     print('getSandP_threaded.py')
#     print(
#         f'The threaded script took {minutes} minutes and {seconds} seconds to run.')
#     The threaded script took 0 minutes and 31 seconds to run.





def unweighted_time_series_avg_betti(wind, dim):
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


def betti_stock(stock, bnh, dims, points, outpath): 
    if not os.path.exists(f'{outpath}/relative_unweighted_betti_dim_{dims}_points_{points}/{stock}.csv'):
        df = pd.read_csv(
            f'sp500/stock dataframes/{stock}_data.csv', index_col='Date')
        df.index=pd.to_datetime(df.index.values)
        df['% returns'] = df['Adj Close']/df['Adj Close'].shift(1)-bnh
        df2 = df[['% returns']]
        tda_df = df2.rolling(window=points*dims).apply(
            lambda x: unweighted_time_series_avg_betti(x, dims))
        tda_df.columns = ['Avg_Rel_%_Betti']
        df = pd.concat([df, tda_df], axis=1)
        df.to_csv(
            f'{outpath}/relative_unweighted_betti_dim_{dims}_points_{points}/{stock}.csv')


if __name__ == '__main__':
    dimlow = 2
    dimup = 6
    pointlow = 5
    pointup = 9
    outpath = '/Users/calum/Documents/python/60:40 test/Betti_experiments'
    stock_closes_df = collecting_stock_data_avg(tickers=s_and_p)
    stock_perc_df = collecting_stock_percentages(s_and_p)
    stock_perc_df.index=pd.to_datetime(stock_perc_df.index.values,format='%Y-%m-%d')
    stock_perc_df.fillna(stock_perc_df.mean(axis=1),inplace=True)
    bnh=stock_perc_df.mean(axis=1)
    for d in tqdm(range(dimlow, dimup+1)):
        for point in tqdm(range(pointlow, pointup+1)):
            if not os.path.exists(f'{outpath}/relative_unweighted_betti_dim_{d}_points_{point}'):
                os.makedirs(
                    f'{outpath}/relative_unweighted_betti_dim_{d}_points_{point}')
            for t in tqdm(s_and_p):
                if not os.path.exists(
                    f'{outpath}/relative_unweighted_betti_dim_{d}_points_{point}/{t}.csv'
                ):
                    betti_stock(t,bnh=bnh, dims=d, points=point, outpath=outpath)
