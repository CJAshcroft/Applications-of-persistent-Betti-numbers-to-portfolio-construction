
import numpy as np
import pandas as pd
from scipy import stats
import pyfolio as pf
import empyrical


def spearman_corr(x):
    if any(x.isna()):
        return(np.nan)
    else:
        rank = [i for i in range(1, len(x)+1)]
        sc, _ = stats.spearmanr(x, rank)
        return(sc)


def proportional_corr(x):
    if any(x.isna()):
        return(np.nan)
    else:
        rank = [i for i in range(1, len(x)+1)]
        sc, _ = stats.spearmanr(x, rank)
        return(sc*(x[-1]**(np.sign(sc))))


##################################################################
# General Utilities for collecting stock returns etc
##################################################################

def collecting_stock_data_avg(tickers, start='2017-07-19', end='2022-06-01', value_name='Adj Close', output_name="Close", inputpath='sp500/stock_dataframes/'):
    """collecting_stock_percentages: returns a dataframe containing
    information on closes

    inputs:
            tickers: a list of strings
                contains the tickers for the stocks

            inputpath: string, optional
                path for where the individual stock data is stored
                default: sp500/stock_dataframes/

                each stock with ticker t should have its daily return data stored
                in a csv at the location inputpath+t+'.csv

            start: string, optional
                start date of strategy
                default 2018-01-01

            end: string, optional
                end date of strategy
                default 2022-06-01

    returns:
            collected_stocks_df: pd.DataFrame
                contains the columns t Adj Close for each stock t, indexed by date"""

    collected_stocks_df = pd.DataFrame(dtype=np.float64)
    for t in tickers:
        stock_df = (

            pd.read_csv(
                inputpath+t+'.csv',
                index_col='Date')
            .loc[start:end]

        )
        df = pd.DataFrame(
            {f'{t} {output_name}': stock_df[f'{value_name}']}
        )
        collected_stocks_df = pd.concat(
            [collected_stocks_df, df],
            axis=1)
    return(collected_stocks_df)


def collecting_stock_percentages(tickers, start='2017-07-19', end='2022-06-01', inputpath='sp500/stock_dataframes/'):
    """collecting_stock_percentages: function that returns a dataframe containing
    information on % return

    inputs:
            tickers: a list of strings
                contains the tickers for the stocks

            inputpath: string, optional
                path for where the individual stock data is stored

                default  sp500/stock_dataframes/

                each stock with ticker t should have its daily return data stored
                in a csv at the location inputpath+t+'.csv


            start: string, optional
                start date of strategy
                default 2018-01-01

            end: string, optional
                end date of strategy
                default 2022-06-01

    returns:
            collected_stocks_df: pd.DataFrame
                contains the columns t % Return for each stock t, indexed by date"""
    collected_stocks_df = pd.DataFrame(dtype=np.float64)
    for t in tickers:
        stock_df = (

            pd.read_csv(
                inputpath+t+'.csv',
                index_col='Date')
            .loc[start:end]

        )
        df2 = (

            (
                (
                    stock_df
                    .loc[:, ["Adj Close"]]
                    - stock_df
                    .loc[:, ["Adj Close"]]
                    .shift(1)
                )
                /
                (
                    stock_df
                    .loc[:, ["Adj Close"]]
                    .shift(1)
                )
            ).set_axis(
                {t+' % Return'},
                axis=1
            )
        )
        collected_stocks_df = pd.concat([collected_stocks_df, df2], axis=1)
    return(collected_stocks_df)


##################################################################
#  Find bins from stock factors & turn to Positions
##################################################################


def find_bins(stock_factor_df, start='2017-07-19', end='2022-06-01'):
    """find bins:   returns a dataframe containing
    the stock bins associated to a dataframe of factors

    inputs:
    -------
            stock_factor_df: pd.DataFrame
                indexed by date

                columns indexed by stock tickers

                each entry is a factor value for corresponding stock/date

            start: string, optional
                start date of strategy
                default 2018-01-01

            end: string, optional
                end date of strategy
                default 2022-06-01

    returns:
            bins df: pd.DataFrame
                contains the columns 'buy bins' (bottom third), 'ignore bins'
                (middle third), and 'sell bins'

                each 'bin' entry is a list of approximately n/3 stock tickers (strings)
                corresponding to the sorted factor values"""
    stock_factor_df = stock_factor_df.loc[start:end]
    index_li = list(stock_factor_df.index.values)
    bins_df = pd.DataFrame(dtype=np.float64)
    for i in index_li:
        df = stock_factor_df.loc[i]
        df = df.dropna()
        df = (

            stock_factor_df
            .loc[i]
            .sort_values()
            .reset_index()

        )
        columnlist = df['index']
        bins_df = pd.concat([
            bins_df,
            pd.DataFrame(
                {
                    'buy bins':
                        [
                            columnlist[0:len(columnlist)//3-1]
                            .values
                        ],
                    'ignore bins':
                        [
                            columnlist[len(columnlist)//3:
                                       len(columnlist)-len(columnlist)//3-1]
                            .values
                        ],
                    'sell bins':
                        [
                            columnlist[len(columnlist) -
                                       len(columnlist)//3:len(columnlist)-1]
                            .values]
                }, index=[i])]
        )
    bins_df.index = pd.to_datetime(
        stock_factor_df.index.values, format='%Y-%m-%d')
    return(bins_df)


def bin_position_calc(tickers, bin_signals, collected_stocks_df, start='2017-07-19', end='2022-06-01'):
    """bin_position_calc: returns a dataframe containing
    the stock positions associated to a collection of bins

    inputs:
            tickers:  list of strings
                contains the tickers for the stocks

            bin_signals: pd.DataFrame
                indexed by date containing the columns 'buy bins'
                (bottom third), 'ignore bins' (middle third), and 'sell bins'

                each 'bin' entry is a list of approximately n/3 stock tickers (strings)
                corresponding to the sorted factor values

            collected_stocks_df: pd.DataFrame
                contains the columns t % Return for each stock t, indexed by date

            start: string, optional
                start date of strategy
                default 2018-01-01

            end: string, optional
                end date of strategy
                default 2022-06-01


    returns:
             positions_df: pd.DataFrame
                indexed by date
                contains columns t Position ,t % Return for t a stock ticker.

                positions_df.loc[date,stock Position] = 1 if stock is in column 'buy bin'
                on time_jump*date//time_jump and np.nan otherwise"""

    bin_signals.index = pd.to_datetime(
        bin_signals.index.values, format='%Y-%m-%d')
    bin_signals = bin_signals.loc[start:end]
    collected_stocks_df.index = pd.to_datetime(
        collected_stocks_df.index.values, format='%Y-%m-%d')
    collected_stocks_df = collected_stocks_df.loc[bin_signals.index.values]
    positions_df = pd.DataFrame(dtype=np.float64)
    for t in tickers:
        df = pd.DataFrame(
            np.nan, columns=[t+' Position'], index=bin_signals.index)
        (

            df
            .loc[
                :, t+' Position'
            ]

        ) = (

            bin_signals
            ['buy bins']
            .map(lambda x: (t in x)*1)

        )
        positions_df = pd.concat([positions_df, df], axis=1)
    positions_df.replace(0, np.nan, inplace=True)
    positions_df = pd.concat([collected_stocks_df, positions_df], axis=1)
    return(positions_df)


def factors_to_pos(stock_factor_df, collected_stocks_df, tickers, start='2018-01-01', end='2022-06-01'):
    """factors_to_pos: returns a dataframe containing
        the positions of each stock

        inputs
        ------
                positions_df: a pd.DataFrame
                    indexed by date
                    contains columns t Position ,t % Return for t a stock ticker

                    positions_df.loc[date,stock Position] = 1 if stock is in column 'buy bin'
                    in collected_stocks_df.loc[time_jump * (date//time_jump)] (i.e.
                    on the date we reassess our position), and np.nan otherwise

                tickers:  list of strings
                    contains the tickers for the stocks

                start: string, optional
                    start date of strategy
                    default 2018-01-01

                end: string, optional
                    end date of strategy
                    default 2022-06-01
        returns:
                positions_df: pd.DataFrame
                    indexed by date
                    contains columns t Position ,t % Return for t a stock ticker.

                    positions_df.loc[date,stock Position] = 1 if stock is in column 'buy bin'
                    on time_jump*date//time_jump and np.nan otherwise"""
    bins = find_bins(stock_factor_df=stock_factor_df, start=start, end=end)
    bins.index = pd.to_datetime(
        bins.index.values,
        format="%Y-%m-%d"
    )
    collected_stocks_df.index = pd.to_datetime(
        collected_stocks_df.index.values,
        format="%Y-%m-%d"
    )
    pos_df = bin_position_calc(
        tickers=tickers,
        bin_signals=bins,
        collected_stocks_df=collected_stocks_df,
        start=start,
        end=end
    )
    return(pos_df)


##################################################################
# Calculating returns
##################################################################


def pos_to_return(positions_df, tickers, start='2018-01-01', end='2022-06-01',  hold_len=5):
    """pos_to_return: returns a dataframe containing
        the returns on a naive stock portfolio

    inputs:
            positions_df: a pd.DataFrame
                indexed by date
                contains columns t Position ,t % Return for t a stock ticker

                positions_df.loc[date,stock Position] = 1 if stock is in column 'buy bin'
                in collected_stocks_df.loc[time_jump * (date//time_jump)] (i.e.
                on the date we reassess our position), and np.nan otherwise

            tickers:  a list of strings
            contains the tickers for the stocks

            start: string, optional
                start date of strategy
                default 2018-01-01

            end: string, optional
                end date of strategy
                default 2022-06-01

            hold_len: int, optional
                the number of days for which we hold our positions
                default 5

    returns:         
            strategy_df: pd.DataFrame
                indexed by date
                returns of strategy"""

    positions_df = positions_df.loc[start:end]
    strategy_return_df = pd.DataFrame()
    for t in tickers:
        stock_pos_df = pd.DataFrame(

            np.nan,
            index=positions_df.index.values,
            columns=[t]

        )
        (
            stock_pos_df
            .loc[
                stock_pos_df.index.values[
                    range(0, len(positions_df.index.values)-hold_len+1, hold_len)
                ], t]

        ) = (

            positions_df
            .loc[
                positions_df.index.values[
                    range(0, len(positions_df.index.values)-hold_len+1, hold_len)
                ], t+' Position']

        )
        if hold_len > 1:
            stock_pos_df.ffill(inplace=True, limit=hold_len-1)
        strategy_return_df = pd.concat(
            [strategy_return_df, stock_pos_df[t]*positions_df[t+' % Return']], axis=1)
    strategy_return_df = strategy_return_df.mean(axis=1)
    strategy_return_df.index.name = 'Date'
    strategy_return_df.columns = ['portfolio return']
    return(strategy_return_df)
