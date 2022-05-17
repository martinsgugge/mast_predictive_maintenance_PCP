import datetime
import os
import numpy as np

from Lesing_og_plotting import plot
from CSV import CSV
# from Henting_og_skriving import select_high_hyg, select_high_hyg_agg
from plots import histogram, correlation_plots, histogram_ly
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from array_manipulation import dual_date_list_interval_per_unit, date_list, split_sequences, merge_df, one_hot_encode, \
    search_string_like_in_list, search_string_exact_in_list, reindex_tags, reindex_tags_static, equalize_tag_length, \
    quantile_remove_up, quantile_remove_down, remove_on_off
# from PCA_PU19 import pca
from postgres import *


def pca_vibration(date_start, date_stop, tagnames):
    """
    Connects to DB, gets data for given tagnames in the time range between date_start and date_stop
    formats data to fit for pca
    executes pca
    :param date_start: string, start date in format YYYY-MM-DD or YYYY-MM-DD HH-mm-ss
    :param date_stop: string, stop date in format YYYY-MM-DD or YYYY-MM-DD HH-mm-ss
    :param tagnames: list of tag names
    :return:
    """


    tags = []
    for i in range(len(tagnames)):
        tags.append(Tag(tagnames[i]))
        tags[i].get_measurement(date_start, date_stop, table='vibration')

    tags = equalize_tag_length(tags)
    data = [tags[0].timestamp]
    for x in tags:
        data.append(x.measurements)
    rha = ['_P_', '_V_']
    # plot("Hyg_PU19 Raw + MA " + date_start[0:10], tagnames, data, rha, single_time=True)
    # CSV.ArrayTocsv(tagnames, data, 'Hyg_PU19 Raw + MA ' + date_start[0:10] + ".csv")

    data = np.array(data, dtype=object)
    data = np.transpose(data)

    tag = ['meas_time']
    for x in tagnames:
        tag.append(x)

    # pca(tag, data, 3)

def get_narrow_table_extract_over_time(date_start, date_stop, tagnames, table, min_per_h=None, agg_time=None,
                                       avg=False, std_dev=False):
    """
    Connects to DB, gets data for given tagnames in the time range between date_start and date_stop with
    min_per_h minutes per hour in the range
    :param date_start: string, start time
    :param date_stop: string, stop time
    :param tagnames: list of strings, names of the tags
    :param min_per_h: int, number of minutes per hour
    :param table: string, name of table, either 'high_speed_big', 'high_speed_big'
    :return: list of tagnames, matrix of data
    """
    # Narrow tables
    tags = []
    failure = False
    # Split time frame up
    if min_per_h != None:
        date_start = date_start.replace(tzinfo=None)
        date_stop = date_stop.replace(tzinfo=None)

        date_1, date_2 = dual_date_list_interval_per_unit(date_start, date_stop, min_per_h)

        if not avg and not std_dev:
            for i in range(len(tagnames)):
                tags.append(Tag(tagnames[i]))
                tags[i].get_measurement(date_1[0], date_2[0], table=table)

                for j in range(1, len(date_1)):
                    tags[i].append_measurement(date_1[j], date_2[j], table=table)
        else:
            if avg:
                agg = 'avg'
            elif std_dev:
                agg = 'stddev_samp'
            for i in range(len(tagnames)):
                tags.append(Tag(tagnames[i]))
                tags[i].tag += agg
                tags[i].get_avg_measurement(date_1[0], date_1[1], agg_time, agg, table=table)

                for j in range(1, len(date_1) - 1):
                    tags[i].append_agg_measurement(date_1[j], date_1[j + 1], agg_time, agg, table=table)

    else:
        if table == 'measurement' or table == 'measurement2':
            date_1 = date_list(date_start, date_stop, 1, day=True)
        else:
            date_1 = date_list(date_start, date_stop, 12, hour=True)

        if not avg and not std_dev:
            for i in range(len(tagnames)):
                tags.append(Tag(tagnames[i]))
                tags[i].get_measurement(date_1[0], date_1[1], table=table)

                for j in range(1, len(date_1)-1):
                    tags[i].append_measurement(date_1[j], date_1[j+1], table=table)
        else:

            if avg:
                agg = 'avg'
            elif std_dev:
                agg = 'stddev_samp'
            for i in range(len(tagnames)):
                tags.append(Tag(tagnames[i]))
                tags[i].tag += agg
                tags[i].get_avg_measurement(date_1[0], date_1[1], agg_time, agg, table=table)
                # if tags[i].failed:
                #     print(date_1[0], date_1[1])
                #     failure = True
                for j in range(1, len(date_1) - 1):

                    tags[i].append_agg_measurement(date_1[j], date_1[j + 1] - datetime.timedelta(minutes=10), agg_time, agg, table=table)
                    #tags[i].failed = False
                    # if tags[i].failed:
                    #     failure = True
                    #     print(date_1[j], date_1[j+1])
    #if not failure:

    for x in tags:
        if not x.timestamp:
            failure = True

    if not failure:
        df = reindex_tags_static(tags, avg or std_dev)
    else:
        df = None
    tags = []
    # else:
    #     df = None

    return df

def get_wide_table_extract_over_time(date_start, date_stop, tagnames, table, min_per_h=None, agg_time=None,
                                     avg=False, std_dev=False):
    """
    Connects to DB, gets data for given tagnames in the time range between date_start and date_stop with
    min_per_h minutes per hour in the range
    :param date_start: string, start time
    :param date_stop: string, stop time
    :param tagnames: list of strings, names of the tags
    :param min_per_h: int, number of minutes per hour
    :param table: string, name of table, either 'high_speed_big', 'high_speed_big'
    :return: matrix of data
    """
    # Split time frame up
    date_start = date_start.replace(tzinfo=None)
    date_stop = date_stop.replace(tzinfo=None)
    date_1, date_2 = dual_date_list_interval_per_unit(date_start, date_stop, min_per_h)
    print(date_1)
    print(avg)
    print(std_dev)
    if not avg and not std_dev:
        data = select_high_hyg(date_1[0], date_2[0], tagnames, table=table, numpy_arr=False)
        for j in range(1, len(date_1)):
            print(date_1[j])
            rows = select_high_hyg(date_1[j], date_2[j], tagnames, table=table, numpy_arr=False)
            for row in rows:
                data.append(row)

    else:
        if std_dev:
            agg = 'stddev_samp'
        elif avg:
            agg = 'avg'

        data = select_high_hyg_agg(date_1[0], date_2[0], agg_time, tagnames, table=table, numpy_arr=False,
                                   aggregation=agg)
        for j in range(1, len(date_1)):
            print(date_1[j])
            rows = select_high_hyg_agg(date_1[j], date_2[j], agg_time, tagnames, table=table, numpy_arr=False,
                                       aggregation=agg)
            for row in rows:
                data.append(row)

    data = np.array(data, dtype=object)

    # Convert to pandas dataframe
    df = pd.DataFrame(data)
    print(tagnames)
    print(df.head())
    df.columns = tagnames
    df['meas_time'] = pd.DatetimeIndex(df['meas_time'])


    # Set datetime datatype
    df['meas_time'] = df['meas_time'].astype('datetime64[ns]')
    # Set float datatype for numeric values
    for name in tagnames[1:]:
        df[name] = df[name].astype('float')

    return df


def get_df(date_start, date_stop, tagnames, table, min_per_h=None, agg_time=None, avg=False, std_dev=False,
           n_steps=None, sequence=False):
    """
    Connects to DB, gets data for given tagnames in the time range between date_start and date_stop with
    min_per_h minutes per hour in the range
    formats data to fit for pca
    executes pca
    :param date_start: string, start time
    :param date_stop: string, stop time
    :param tagnames: list of strings, names of the tags
    :param min_per_h: int, number of minutes per hour
    :param table: string, name of table, either 'measurement', 'vibration', 'high_speed_big', 'high_speed_big'
    :return: pandas dataframes, training set, test set
    """
    state = None
    on_sig = None
    tagnames_ = tagnames.copy()
    state = search_string_like_in_list('_State', tagnames_)
    if state is not None:
        tagnames_.remove(state)
        print('state defined')

    on_sig = search_string_exact_in_list(tagnames[0][:8], tagnames_)
    if on_sig is not None:
        tagnames_.remove(on_sig)
        print(f'on signal defined {on_sig}')

    write = False
    # Narrow tables
    if table == 'measurement' or table == 'vibration' or table == 'measurement2':
        if os.path.exists('./' + 'raw ' + 'agg_time=' + str(agg_time) + 'avg=' + str(avg) + 'std_dev=' + str(
                std_dev) + 'min_per_h=' +
                          str(min_per_h) + table + ' ' + str(date_start)[:10] + ' to ' + str(date_stop)[:10] + '.csv'):
            try:
                df = pd.read_csv('./' + 'raw ' + 'agg_time=' + str(agg_time) + 'avg=' + str(avg) + 'std_dev=' +
                                 str(std_dev) + 'min_per_h=' + str(min_per_h) + table + ' ' + str(date_start)[:10] +
                                 ' to ' + str(date_stop)[:10] + '.csv',
                                 usecols=['meas_time', 'PU19_PW_PV', 'PU19_TQ_PV', 'PU19_MO', 'PU19_SF_PV', 'FT02',
                                          'PT15',
                                          'PT16', 'PU19_V_L', 'PU19_V_I', 'PU19_V_O', 'PU19_P_L', 'PU19_P_I',
                                          'PU19_P_O'])
            except ValueError as e:
                df = pd.read_csv('./' + 'raw ' + 'agg_time=' + str(agg_time) + 'avg=' + str(avg) + 'std_dev=' +
                                 str(std_dev) + 'min_per_h=' + str(min_per_h) + table + ' ' + str(date_start)[:10] +
                                 ' to ' + str(date_stop)[:10] + '.csv',
                                 usecols=['meas_time', 'PU19_PW_PV', 'PU19_TQ_PV', 'PU19_MO', 'PU19_SF_PV', 'FT02',
                                          'PT15', 'PT16'])

            df['meas_time'] = df['meas_time'].apply(lambda x: x[0:18])
            df['meas_time'] = pd.to_datetime(df['meas_time'])
            # Sort df2
            df['meas_time'] = pd.DatetimeIndex(df['meas_time'])

            write = False
        else:
            df = get_narrow_table_extract_over_time(date_start, date_stop, tagnames_, table,
                                                    min_per_h=min_per_h, agg_time=agg_time, avg=avg, std_dev=std_dev)

    # Wide tables
    elif table == 'high_speed_big' or table == 'high_speed_big_reduced':
        if os.path.exists('./' + 'raw ' + 'agg_time=' + str(agg_time) + 'avg=' + str(avg) + 'std_dev=' + str(
                std_dev) + 'min_per_h=' +
                          str(min_per_h) + table + ' ' + str(date_start)[:10] + ' to ' + str(date_stop)[:10] + '.csv'):
            df = pd.read_csv('./' + 'raw ' + 'agg_time=' + str(agg_time) + 'avg=' + str(avg) + 'std_dev=' + str(
                std_dev) + 'min_per_h=' +
                             str(min_per_h) + table + ' ' + str(date_start)[:10] + ' to ' + str(date_stop)[
                                                                                            :10] + '.csv',
                             usecols=tagnames_)

            df['meas_time'] = df['meas_time'].apply(lambda x: x[0:18])
            df['meas_time'] = pd.to_datetime(df['meas_time'])
            # Sort df2
            df['meas_time'] = pd.DatetimeIndex(df['meas_time'])

            write = False
        else:
            df = get_wide_table_extract_over_time(date_start, date_stop, tagnames_, min_per_h=min_per_h, table=table,
                                                  agg_time=agg_time, avg=avg, std_dev=std_dev)

    if write:
        df.to_csv('raw ' + 'agg_time=' + str(agg_time) + 'avg=' + str(avg) + 'std_dev=' + str(std_dev) + 'min_per_h=' +
                  str(min_per_h) + table + ' ' + str(date_start)[:10] + ' to ' + str(date_stop)[:10] + '.csv')
    print(on_sig)
    if on_sig is not None:
        if os.path.exists('./HYG_PU19 ' + str(date_start)[:10] + ' to ' + str(date_stop)[:10] + '.csv'):
            df2 = pd.read_csv('./HYG_PU19 ' + str(date_start)[:10] + ' to ' + str(date_stop)[:10] + '.csv',
                              usecols=['meas_time', 'PU19'])

            df2['meas_time'] = df2['meas_time']  # .apply(lambda x: x[0:18])
            # df2['meas_time'] = pd.to_datetime(df2['meas_time'])
            df2['meas_time'] = df2['meas_time'].astype('datetime64[ns]')
            # Sort df22
            df2['meas_time'] = pd.DatetimeIndex(df2['meas_time'])



        else:
            df2 = get_narrow_table_extract_over_time(date_start, date_stop, on_sig, table=table, min_per_h=60)

            df2.to_csv('HYG_PU19 ' + str(date_start)[:10] + ' to ' + str(date_stop)[:10] + '.csv')

        print(df.head())
        print(df2.head())
        df = merge_df(df, df2)

    if state != None:
        print('Get state values')
        # df2 = get_narrow_table_extract_over_time(date_start, date_stop, state, table='measurement', min_per_h=60)
        df2 = pd.read_csv('HYG_PU19 Status.csv', usecols=['meas_time', 'PU19_State'])

        df2['meas_time'] = df2['meas_time'].apply(lambda x: x[0:18])
        df2['meas_time'] = pd.to_datetime(df2['meas_time'])
        # Sort df2
        df2['meas_time'] = pd.DatetimeIndex(df2['meas_time'])
        df2.sort_index(axis=1, ascending=True)

        df = merge_df(df, df2)

    # Define parts of tagnames which are on right hand side axis of line plot
    rha = ['_P_', '_V_']
    # Plot the data to line chart
    # plot("Hyg_PU19 Raw + MA " + date_start[0:10], tagnames, data, rha, single_time=True)
    df['meas_time'] = pd.DatetimeIndex(df['meas_time'])
    for x in df.columns[1:]:
        df[x] = df[x].astype('float')

    return df


def get_df(date_start, date_stop, tagnames, table, min_per_h=None, agg_time=None, avg=False, std_dev=False):
    """
    Connects to DB, gets data for given tagnames in the time range between date_start and date_stop with
    min_per_h minutes per hour in the range
    :param date_start: string, start time
    :param date_stop: string, stop time
    :param tagnames: list of strings, names of the tags
    :param table: string, name of table, either 'measurement', 'vibration', 'high_speed_big', 'high_speed_big'
    :param min_per_h: int, number of minutes per hour
    :param agg_time: Time to aggregate over
    :param avg: True if data should be averaged
    :param std_dev: True if data should be aggregated to standard deviation
    :return: pandas dataframe
    """

    tagnames_ = tagnames.copy()
    state = search_string_like_in_list('_State', tagnames_)
    if state is not None:
        tagnames_.remove(state)
        if agg_time is not None:
            state = None

    on_sig = search_string_exact_in_list(tagnames[0][:8], tagnames_)
    if on_sig is not None:
        tagnames_.remove(on_sig)
        if agg_time is not None:
            on_sig = None

    # Narrow tables
    if table == 'measurement' or table == 'vibration' or table == 'measurement2':
        df = get_narrow_table_extract_over_time(date_start, date_stop, tagnames_, table,
                                                    min_per_h=min_per_h, agg_time=agg_time, avg=avg, std_dev=std_dev)

    # Wide tables
    elif table == 'high_speed_big' or table == 'high_speed_big_reduced':
        df = get_wide_table_extract_over_time(date_start, date_stop, tagnames_, min_per_h=min_per_h, table=table,
                                                  agg_time=agg_time, avg=avg, std_dev=std_dev)

        df = df.rename(columns={"meas_time": "Time"})
    if df is not None:
        df['Time'] = [x.replace(tzinfo=None) for x in df['Time']]

    if on_sig is not None:
        df2 = get_narrow_table_extract_over_time(date_start, date_stop, [on_sig], table=table, min_per_h=None)
        try:
            df2['Time'] = [x.replace(tzinfo=None) for x in df2['Time']]
            df = merge_df(df, df2)
        except TypeError as e:
            print(e)



    if state is not None:
        df2 = get_narrow_table_extract_over_time(date_start, date_stop, [state], table='measurement_temp',
                                                 min_per_h=None)
        try:
            df2['Time'] = [x.replace(tzinfo=None) for x in df2['Time']]
            df = merge_df(df, df2)
        except TypeError as e:
            print(e)

    return df




def get_combined_aggregate_wide_raw_narrow(date_start, date_stop):
    table_narrow = 'measurement'
    min_per_h = 60
    table_wide = 'high_speed_big'
    narrow = get_df(date_start, date_stop, tagnames, table_narrow, min_per_h=min_per_h)

def pca_vibration_over_time(date_start, date_stop, tagnames, table, min_per_h=None, agg_time=None, avg=False, std_dev=False):
    """
    Connects to DB, gets data for given tagnames in the time range between date_start and date_stop with
    min_per_h minutes per hour in the range
    formats data to fit for pca
    executes pca
    :param date_start: string, start time
    :param date_stop: string, stop time
    :param tagnames: list of strings, names of the tags
    :param min_per_h: int, number of minutes per hour
    :param table: string, name of table, either 'measurement', 'vibration', 'high_speed_big', 'high_speed_big'
    :return:
    """
    state = None
    on_sig = None
    tagnames_ = tagnames.copy()
    if "HYG_PU19_State" in tagnames:

        state = ["HYG_PU19_State"]
        tagnames_.remove("HYG_PU19_State")
        print('state defined')

    if "HYG_PU19" in tagnames_:
        on_sig = ["HYG_PU19"]
        tagnames_.remove("HYG_PU19")

    write = True
    #Narrow tables
    if table == 'measurement' or table == 'vibration':
        if os.path.exists('./' + 'raw ' + 'agg_time=' + str(agg_time) + 'avg='+str(avg) + 'std_dev=' + str(std_dev) + 'min_per_h=' +
              str(min_per_h) + table + ' ' + str(date_start)[:10] + ' to ' + str(date_stop)[:10] + '.csv'):
            try:
                df = pd.read_csv('./' + 'raw ' + 'agg_time=' + str(agg_time) + 'avg='+str(avg) + 'std_dev=' + str(std_dev) + 'min_per_h=' +
                  str(min_per_h) + table + ' ' + str(date_start)[:10] + ' to ' + str(date_stop)[:10] + '.csv',
                                 usecols=['meas_time','PU19_PW_PV','PU19_TQ_PV','PU19_MO','PU19_SF_PV','FT02','PT15','PT16','PU19_V_L',
                              'PU19_V_I','PU19_V_O','PU19_P_L','PU19_P_I','PU19_P_O'])
            except ValueError as e:
                df = pd.read_csv('./' + 'raw ' + 'agg_time=' + str(agg_time) + 'avg=' + str(avg) + 'std_dev=' +
                                 str(std_dev) + 'min_per_h=' + str(min_per_h) + table + ' ' + str(date_start)[:10] +
                                 ' to ' + str(date_stop)[:10] + '.csv',
                                 usecols=['meas_time', 'PU19_PW_PV', 'PU19_TQ_PV', 'PU19_MO', 'PU19_SF_PV', 'FT02',
                                          'PT15', 'PT16'])
            df['meas_time'] = df['meas_time'].apply(lambda x: x[0:18])
            df['meas_time'] = pd.to_datetime(df['meas_time'])
            # Sort df2
            df['meas_time'] = pd.DatetimeIndex(df['meas_time'])

            write = False
        else:
            df = get_narrow_table_extract_over_time(date_start, date_stop, tagnames_, table,
                                               min_per_h=min_per_h, agg_time=agg_time, avg=avg, std_dev=std_dev)

    #Wide tables
    elif table == 'high_speed_big' or table == 'high_speed_big_reduced':
        if os.path.exists('./' + 'raw ' + 'agg_time=' + str(agg_time) + 'avg='+str(avg) + 'std_dev=' + str(std_dev) + 'min_per_h=' +
              str(min_per_h) + table + ' ' + str(date_start)[:10] + ' to ' + str(date_stop)[:10] + '.csv'):
            df = pd.read_csv('./' + 'raw ' + 'agg_time=' + str(agg_time) + 'avg='+str(avg) + 'std_dev=' + str(std_dev) + 'min_per_h=' +
              str(min_per_h) + table + ' ' + str(date_start)[:10] + ' to ' + str(date_stop)[:10] + '.csv',
                             usecols=tagnames_)

            df['meas_time'] = df['meas_time'].apply(lambda x: x[0:18])
            df['meas_time'] = pd.to_datetime(df['meas_time'])
            # Sort df2
            df['meas_time'] = pd.DatetimeIndex(df['meas_time'])

            write = False
        else:
            df = get_wide_table_extract_over_time(date_start, date_stop, tagnames_, min_per_h=min_per_h, table=table,
                                              agg_time=agg_time, avg=avg, std_dev=std_dev)

    if write:
        df.to_csv('raw ' + 'agg_time=' + str(agg_time) + 'avg='+str(avg) + 'std_dev=' + str(std_dev) + 'min_per_h=' +
                  str(min_per_h) + table + ' ' + str(date_start)[:10] + ' to ' + str(date_stop)[:10] + '.csv')

    if on_sig != None:
        if os.path.exists('./HYG_PU19 ' + str(date_start)[:10] + ' to ' + str(date_stop[:10]) + '.csv'):
            df2 = pd.read_csv('./HYG_PU19 ' + str(date_start)[:10] + ' to ' + str(date_stop[:10]) + '.csv',
                              usecols=['meas_time', 'PU19'])

            df2['meas_time'] = df2['meas_time']  # .apply(lambda x: x[0:18])
            # df2['meas_time'] = pd.to_datetime(df2['meas_time'])
            df2['meas_time'] = df2['meas_time'].astype('datetime64[ns]')
            # Sort df22
            df2['meas_time'] = pd.DatetimeIndex(df2['meas_time'])



        else:
            df2 = get_narrow_table_extract_over_time(date_start, date_stop, on_sig, table='measurement', min_per_h=60)

            df2.to_csv('HYG_PU19 ' + str(date_start)[:10] + ' to ' + str(date_stop)[:10] + '.csv')

        df = merge_df(df, df2)

    if state != None:
        print('Get state values')
        #df2 = get_narrow_table_extract_over_time(date_start, date_stop, state, table='measurement', min_per_h=60)
        df2 = pd.read_csv('HYG_PU19 Status.csv', usecols=['meas_time', 'PU19_State'])

        df2['meas_time'] = df2['meas_time'].apply(lambda x: x[0:18])
        df2['meas_time'] = pd.to_datetime(df2['meas_time'])
        #Sort df2
        df2['meas_time'] = pd.DatetimeIndex(df2['meas_time'])
        df2.sort_index(axis=1, ascending=True)

        df = merge_df(df, df2)

    #Define parts of tagnames which are on right hand side axis of line plot
    rha = ['_P_', '_V_']
    #Plot the data to line chart
    # plot("Hyg_PU19 Raw + MA " + date_start[0:10], tagnames, data, rha, single_time=True)

    train, test = train_test_split(df.copy(deep=True), test_size=0.3, random_state=42)

    df = None

    df = train.copy(deep=True)
    df.dropna()
    df = remove_on_off(df)
    histogram_ly(train.copy(deep=True), table + ' ' + str(date_start)[:10] + ' to ' + str(date_stop)[:10])
    correlation_plots(train.copy(deep=True), table + ' ' + str(date_start)[:10] + ' to ' + str(date_stop)[:10])
    # pca(df.copy(deep=True), 3, table + ' ' + str(date_start)[:10] + ' to ' + str(date_stop)[:10])




def call_histogram(date_start, date_stop, tagnames, min_per_h, table, avg_time, no_of_bins=50):
    # Narrow tables
    if table == 'measurement' or table == 'vibration':
        df= get_narrow_table_extract_over_time(date_start, date_stop, tagnames, min_per_h, table, avg_time)

    # Wide tables
    elif table == 'high_speed_big' or table == 'high_speed_big_reduced':
        df = get_wide_table_extract_over_time(date_start, date_stop, tagnames, min_per_h, table)


    histogram(df, no_of_bins)
    correlation_plots(df)
    """
    for i in df.columns.to_flat_index():
        df = removeOutliers(df, i)

    histogram(df, no_of_bins)
    correlation_plots(df)"""


def call_correlation(date_start, date_stop, tagnames, min_per_h, table, no_of_bins=50):
    # Narrow tables
    if table == 'measurement' or table == 'vibration':
        tag, data = get_narrow_table_extract_over_time(date_start, date_stop, tagnames, min_per_h, table)

    # Wide tables
    elif table == 'high_speed_big' or table == 'high_speed_big_reduced':
        data = get_wide_table_extract_over_time(date_start, date_stop, tagnames, min_per_h, table)
        tag = tagnames




if __name__ == '__main__':
    """
    pump_id = Tag('HYG_PU19_Pump_No')
    pump_id.get_measurement('2020-03-16', '2021-10-01')
    """
    pump_id = pd.read_csv('./HYG_PU19 Device ID.csv')
    pump_id.dropna()
    pump_id['Time'] = pump_id['Time'].apply(lambda x: x[0:20])
    pump_id['Time'] = pd.to_datetime(pump_id['Time'])

    # Measurement

    tagnames = ["HYG_PU19_PW_PV", "HYG_PU19_TQ_PV", "HYG_PU19_MO", "HYG_PU19_SF_PV", "HYG_FT02", "HYG_PT15", "HYG_PT16",
                "HYG_PU19_State", "HYG_PU19"
                # "HYG_PU19_V_L", "HYG_PU19_V_I", "HYG_PU19_V_O", "HYG_PU19_P_L", "HYG_PU19_P_I", "HYG_PU19_P_O"
                ]
    for i in range(len(pump_id['Time']) - 1):
        if pump_id['Time'][i].date() > datetime.date(2021, 9, 1):
            pca_vibration_over_time(pump_id['Time'][i], pump_id['Time'][i + 1], tagnames, 'measurement', std_dev=False,
                                    min_per_h=60, avg=True, agg_time=10)

    tagnames = ["HYG_PU19_PW_PV", "HYG_PU19_TQ_PV", "HYG_PU19_MO", "HYG_PU19_SF_PV", "HYG_FT02", "HYG_PT15", "HYG_PT16",
                "HYG_PU19_V_L", "HYG_PU19_V_I", "HYG_PU19_V_O", "HYG_PU19_P_L", "HYG_PU19_P_I", "HYG_PU19_P_O",
                "HYG_PU19_State", "HYG_PU19"]

    #pca_vibration('2021-06-20 00:00:00', '2021-07-04 00:00:00', tagnames)

    #pca_vibration_over_time('2020-04-01 00:00:00', '2021-08-31 00:00:00', tagnames, 5, 'measurement')
    # Vibration
    for i in range(len(pump_id['Time']) - 1):

        if pump_id['Time'][i].date() > datetime.date(2021, 9, 1):
            pca_vibration_over_time(pump_id['Time'][i], pump_id['Time'][i+1], tagnames, 'vibration', min_per_h=60,
                                    std_dev=False, avg=True, agg_time=1/3)

    #High_speed_big_reduced
    tagnames = ['meas_time', 'pu19_pw_pv', 'pu19_tq_pv', 'pu19_sf_pv', 'pu19_mo', 'pt15', 'pt16', "HYG_PU19_State",
                "HYG_PU19"]
    for i in range(len(pump_id['Time']) - 1):
        if pump_id['Time'][i].date() > datetime.date(2021, 9, 1):
            pca_vibration_over_time(pump_id['Time'][i], pump_id['Time'][i+1], tagnames, 'high_speed_big_reduced',
                                    min_per_h=10, std_dev=False, avg=True, agg_time=1)







    #call_histogram('2021-05-06', '2021-09-10', tagnames, 60, 'measurement', no_of_bins=500)
