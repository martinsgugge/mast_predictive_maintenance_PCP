import datetime
import math
import pandas as pd
import numpy as np
import datetime as dt
import time
from sklearn.model_selection import train_test_split
from XML import XML
from postgres import Tag


def find_longest(lists):
    """
    Return the index of the longest sublist in lists
    :param lists: list of lists
    :return: index of longest sublist
    """

    l = len(lists[0])
    longest = 0
    for i in range(1,len(lists)):
        length = len(lists[i])
        if l < length:
            l = length
            longest = i

    return longest


def merge_df(df1, df2, method='ffill'):
    """
    Merges two Pandas Dataframes together on a Time axis using nearest value to fill empty slots
    :param df1: Arbitrary dataframe with a Time index which will be appended
    :param df2: Arbitrary dataframe with a Time index which will be appended to df1
    :return: dataframe, df1 merged with df2, only one time axis
    """

    try:
        # Remove duplicates of both dataframes
        df1 = df1.drop_duplicates(keep='first')
        df1 = df1.set_index('Time')
        df1.sort_index(inplace=True)

        df2 = df2.drop_duplicates(keep='first')
        df2 = df2.set_index('Time')
        df2.sort_index(inplace=True)

        # Reindex the shorter array
        B = df2.reindex(df1.index, method=method).reset_index()

        #Merge dataset
        df = pd.merge_asof(df1, B, on='Time')
    except TypeError as e:
        print('Type error i merge_df')
        try:
            df = pd.merge_asof(df1, df2, on='Time')
        except pd.errors.MergeError as e:
            print('Merge error i merge_df')
            print(e)
            df = None

    except ValueError as e:
        df = pd.merge_asof(df1, df2, on='Time')
    except IndexError as e:
        df = None
    except AttributeError as e:
        print('Attribute error i merge_df')
        print(e)
        df = None

    return df

def convert_tag_to_df(tag: Tag):
    data = np.array([np.array(tag.timestamp), np.array(tag.measurements)]).T
    df = pd.DataFrame(data, columns=['Time', tag.tag])
    df['Time'] = [x.replace(tzinfo=None) for x in df['Time']]
    df['Time'] = pd.to_datetime(df['Time'])
    df[tag.tag] = pd.to_numeric(df[tag.tag], downcast='float')

    return df

def reindex_tags(tags, aggregated=False):
    """
    Makes each of the measurements lists equally long
    :param tags: list of tags
    :return: DataFrame, one time axis, length of values are the largest
    """
    #Find index of longest list

    timelist = [x.timestamp for x in tags]
    longest_list = find_longest(timelist)

    #Format data for data frame
    #convert_tag_to_df(tags[longest_list])
    data = np.array([np.array(tags[longest_list].timestamp), np.array(tags[longest_list].measurements)]).T
    #Add longest
    df = pd.DataFrame(data, columns=['Time', tags[longest_list].tag])
    #df['Time'] = pd.to_datetime(df['Time'])
    df[tags[longest_list].tag] = pd.to_numeric(df[tags[longest_list].tag], downcast='float')
    #df.set_index('Time')

    del tags[longest_list]

    #Add remaining tag data by fitting them to the longest data set using the closest value
    for tag in tags:
        if aggregated:

            df[tag.tag] = pd.to_numeric(np.array(tag.measurements), downcast='float')

        else:
            data = np.array([np.array(tag.timestamp), np.array(tag.measurements)]).T
            df2 = pd.DataFrame(data, columns=['Time', tag.tag])
            df2['Time'] = pd.to_datetime(df2['Time'])
            df2[tag.tag] = pd.to_numeric(df2[tag.tag], downcast='float')

            df = merge_df(df, df2, method='nearest')

    return df

def reindex_tags_static(tags, aggregated=False):
    """
    Makes each of the measurements lists equally long
    :param tags: list of tags
    :return: DataFrame, one time axis, length of values are the largest
    """
    #Format data for data frame
    if len(tags[0].timestamp) > len(tags[0].measurements):
        diff = len(tags[0].measurements) - len(tags[0].timestamp)
        del tags[0].timestamp[diff:]
    elif len(tags[0].timestamp) < len(tags[0].measurements):
        diff = len(tags[0].timestamp) - len(tags[0].measurements)
        del tags[0].measurement[diff:]

    timestamp = np.array(tags[0].timestamp)
    measurement = np.array(tags[0].measurements)
    data = np.array([timestamp, measurement], dtype=object).T

    #Add longest
    df = pd.DataFrame(data, columns=['Time', tags[0].tag])
    #df['Time'] = pd.to_datetime(df['Time'])
    df[tags[0].tag] = pd.to_numeric(df[tags[0].tag], downcast='float')
    #df.set_index('Time')

    del tags[0]

    #Add remaining tag data by fitting them to the longest data set using the closest value
    for tag in tags:
        if aggregated:

            df[tag.tag] = pd.to_numeric(np.array(tag.measurements), downcast='float')

        else:
            data = np.array([np.array(tag.timestamp), np.array(tag.measurements)]).T
            df2 = pd.DataFrame(data, columns=['Time', tag.tag])
            df2['Time'] = pd.to_datetime(df2['Time'])
            df2[tag.tag] = pd.to_numeric(df2[tag.tag], downcast='float')
            #if len(df[tag.tag].index) > 0:
            df = merge_df(df, df2, method='nearest')

    return df

def timedelta_to_time(data):
    print(data)
    days, seconds = data.days, data.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours < 10:
        hours = f'0{hours}'
    if minutes < 10:
        minutes = f'0{minutes}'
    if seconds < 10:
        seconds = f'0{seconds}'

    tmp = time.strptime(f'{hours}:{minutes}:{seconds}', 'HH:MM:ss')

    return tmp

def convert_to_date_or_datetime(datetime_var):
    if isinstance(datetime_var, str):
        try:
            datetime_result = dt.datetime.strptime(datetime_var, '%Y-%m-%d')
        except ValueError as e:
            try:
                datetime_result = dt.datetime.strptime(datetime_var, '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                try:
                    datetime_result = dt.datetime.strptime(datetime_var, '%Y-%m-%d %H:%M:%S.%f')
                except ValueError as e:
                    try:
                        datetime_result = dt.datetime.strptime(datetime_var, '%d.%m.%Y %H:%M')
                    except ValueError as e:
                        datetime_result = dt.datetime.strptime(datetime_var, '%Y-%m-%d %H:%M:%S+%r')

        return datetime_result

    else:
        return datetime_var

def date_list(date_start, date_stop, interval, ms=True, second=False, min=False,
                 hour=False, day=False):
    """
    Creates a list of datetime objects from date_start to date_stop with interval defined by optional flags and
    interval, ms is standard

    :param date_start: string, Time start
    :param date_stop: string, time stop
    :param interval: int, time of interval in ms to create date list of
    :param ms: bool, if only this is true, give list of ms interval
    :param second: bool, if only this is true, give list of ms interval
    :param min: bool, if only this is true, give list of ms interval
    :param hour: bool, if only this is true, give list of ms interval
    :param day: bool, if only this is true, give list of ms interval
    :return:
    """
    """    print(type(date_start))
        #print(isinstance(date_start, dt.date))
        if not isinstance(date_start, dt.datetime) or not isinstance(date_start, pd.Timestamp) \
                or not isinstance(date_start, dt.date):
            print(not isinstance(date_start, dt.datetime) or not isinstance(date_start, pd.Timestamp)
                or not isinstance(date_start, dt.date))
            print(type(date_start))
            print(date_start)"""
    if isinstance(date_start, str):
        date_stop = convert_to_date_or_datetime(date_stop)
        date_start = convert_to_date_or_datetime(date_start)

    list_ms = []
    list_ms.append(date_start)

    # Makes list of days interval
    if day:
        while list_ms[-1] < date_stop:
            list_ms.append(list_ms[-1] + dt.timedelta(days=interval))

    # Makes list of hours interval
    elif hour:
        while list_ms[-1] < date_stop:
            list_ms.append(list_ms[-1] + dt.timedelta(hours=interval))

    # Makes list of minutes interval
    elif min:
        while list_ms[-1] < date_stop:
            list_ms.append(list_ms[-1] + dt.timedelta(minutes=interval))

    # makes list of seconds interval
    elif second:
        while list_ms[-1] < date_stop:
            list_ms.append(list_ms[-1] + dt.timedelta(seconds=interval))

    # Makes lists of ms intervals
    elif ms:
        while list_ms[-1] < date_stop:
            list_ms.append(list_ms[-1] + dt.timedelta(milliseconds=interval))

    return list_ms

def dual_date_list_interval_per_unit(date_start, date_stop, time_per_unit):
    """
    makes two lists of datetimes with a difference of time per unit
    :param date_start: string, start time
    :param date_stop: string, stop time
    :param time_per_unit: int, number of minutes per hour
    :return: date_1 is a list of intervals with 1h from initial time to initial end,
    date_2 is a list of intervals with 1h from initial time + time_per_unit
    """
    date_start = str(date_start)
    date_stop = str(date_stop)

    date_1 = date_list(date_start, date_stop, interval=1, hour=True)

    try:
        date_2 = date_list(str(datetime.datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S') +
                               datetime.timedelta(minutes=time_per_unit)),
                           str(datetime.datetime.strptime(date_stop, '%Y-%m-%d %H:%M:%S') +
                               datetime.timedelta(minutes=time_per_unit)),
                           interval=1, hour=True)
    except ValueError:
        try:
            date_2 = date_list(str(datetime.datetime.strptime(date_start, '%Y-%m-%d') +
                                   datetime.timedelta(minutes=time_per_unit)),
                               str(datetime.datetime.strptime(date_stop, '%Y-%m-%d') +
                                   datetime.timedelta(minutes=time_per_unit)),
                               interval=1, hour=True)
        except ValueError:
            date_2 = date_list(str(datetime.datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S.%f') +
                                   datetime.timedelta(minutes=time_per_unit)),
                               str(datetime.datetime.strptime(date_stop, '%Y-%m-%d %H:%M:%S.%f') +
                                   datetime.timedelta(minutes=time_per_unit)),
                               interval=1, hour=True)



    except TypeError:
        date_2 = date_list((date_start +
                               datetime.timedelta(minutes=time_per_unit)),
                           (date_stop +
                               datetime.timedelta(minutes=time_per_unit)),
                           interval=1, hour=True)

    date_1 = date_1[:-1]
    date_2 = date_2[:-1]
    return date_1, date_2

def aggregate(start, stop, data, tagname):
    """
    Removes timezone info, finds delta and adds data to a list of lists
    :param start: datetime, start measurement time
    :param stop: datetime, stop measurement time
    :param data: data structure to be appended
    :param tagname: name of tag to be added to the structure
    :return: appended data structure
    """
    try:
        start = start.replace(tzinfo=None)
        stop = stop.replace(tzinfo=None)
    except TypeError as e:
        pass
    temp_uptime = stop - start


    if data != []:
        total_uptime = temp_uptime + data[-1][3]
        j = 1 + data[-1][0]
        #print(temp_uptime, stop, start, data[-1][0])
    else:
        j = 1
        total_uptime = temp_uptime


    data.append([j, start, temp_uptime, total_uptime, tagname])

    return data

# multivariate data preparation
from numpy import array
from numpy import hstack
from sys import getsizeof


def split_sequences(sequences, n_steps, pad=False):
    features = get_features(sequences)
    x = sequences[features].to_numpy()
    length = len(x)
    print(x.shape)
    new_length = length//n_steps

    X, y = list(), list()
    # print(len(sequences))
    # print(type(sequences))
    # counter = 0
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        if pad:
            if i % n_steps == 0:
                #print(lengths of numpy array)
                print('lists: ', getsizeof(X), getsizeof(y))
                print('df: ', type(sequences[features].iloc[i:end_ix].to_numpy()))
                print('df: ', getsizeof(sequences[features].iloc[i:end_ix].to_numpy()))
                seq_x = arr[i:end_ix]
                print(seq_x)
                print('seqx: ', getsizeof(seq_x))
                print('indices; ', i, ' ', end_ix)
                # seq_x, seq_y = sequences[features].iloc[i:end_ix].to_numpy(), sequences['State'].iloc[i:end_ix].to_numpy()
                X.append(seq_x)
                y.append(arr[-1][i:end_ix])
        else:
            seq_x, seq_y = sequences[features].iloc[i:end_ix].to_numpy(), sequences['State'].iloc[
                                                                          i:end_ix].to_numpy()
            X.append(seq_x)
            y.append(seq_y)
    del sequences
    #del seq_x
    #del seq_y
    return np.array(X), np.array(y)


def split_sequencesv2(sequences, n_steps, pad=False):
    features = get_features(sequences)
    x = sequences[features].to_numpy()
    y = sequences['State'].to_numpy()
    length = len(x)
    print(x.shape)
    print(y.shape)
    new_length = length//n_steps
    print(new_length)
    X = np.zeros((length, n_steps, len(features)))
    Y = np.zeros((length, n_steps))
    #X, y = list(), list()
    # print(len(sequences))
    # print(type(sequences))
    # counter = 0
    for i in range(new_length):
        # find start and stop of this iteration
        start_ix = i * n_steps
        end_ix = (i + 1) * n_steps

        #insert data
        X[i] = x[start_ix:end_ix]
        Y[i] = y[start_ix:end_ix]
    print(X.shape)
    print(Y.shape)
    return X, Y

def create_dataset(Time, X, y, time_steps=1):
    Xs, ys = [], []

    skipped_series = 0

    for i in range(len(X) - time_steps):
        # Start and end time of work data
        start = pd.to_datetime(Time.iloc[i])
        end = pd.to_datetime(Time.iloc[i + time_steps])

        if end <= start + pd.DateOffset(minutes=61):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        else:
            skipped_series += 1
    print("Skipped series: ", skipped_series)
    return np.array(Xs), np.array(ys)

def test_split_sequence():
    # define input sequence
    in_seq1 = array([20, 10, 20, 30, 40, 50, 60, 70, 80, 90, 10, 10, 20, 10, 20, 30, 40, 50, 60, 70, 80, 90, 10, 20, 30, 40, 50, 60, 70, 80, 90, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    in_seq2 = array([25, 15, 25, 35, 45, 55, 65, 75, 85, 95, 15, 15, 25, 15, 25, 35, 45, 55, 65, 75, 85, 95, 15, 25, 35, 45, 55, 65, 75, 85, 95, 15, 25, 35, 45, 55, 65, 75, 85, 95])
    out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
    #print(in_seq1)
    df = pd.DataFrame(array([in_seq1, in_seq2, out_seq]).T)
    print(df)
    # convert to [rows, columns] structure
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    #print(in_seq1)

    # horizontally stack columns
    dataset = hstack((in_seq1, in_seq2, out_seq))
    # choose a number of time steps
    n_steps = 4
    # convert into input/output
    X, y = split_sequences(df, n_steps)

    print('test')
    df = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    print(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    print('xtrain')
    print(X_train)
    print('xtest')
    print(X_test)
    print('ytrain')
    print(y_train)
    print('ytest')
    print(y_test)
    print(X_test.shape, y_test.shape)

def remove_quantiles(df, upper, lower, columns):
    """
    https://nextjournal.com/schmudde/how-to-remove-outliers-in-data
    :param df:
    :param upper: value between 0 and 1
    :param lower:value between 0 and 1
    :param columns: column names in df
    :return:
    """
    for x in columns:
        temp = df[x]
        removed_outliers = temp.between(temp.quantile(lower), temp.quantile(upper))
        index = df[~removed_outliers].index
        df.drop(index, inplace=True)
    return df


def one_hot_encode(sequence, n_unique=3):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[int(value)] = 1
        encoding.append(vector)

    return np.array(encoding)


def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]


def search_string_like_in_list(search_word, list_of_strings):
    result = None
    for line in list_of_strings:
        if line.__contains__(search_word):
            result = line
    return result

def search_string_exact_in_list(search_word, list_of_strings):
    result = None
    if search_word in list_of_strings:
        print(search_word)
        result = search_word
    return result

def assert_length_in_list_of_string(length, list_of_strings):
    result = None
    for line in list_of_strings:
        if len(line) == length:
            result = line
    return result


def get_generic_column_names(df: pd.DataFrame, generic_tags: [str]):
    """
    Get generic names for inputs given a specified list of generic inputs
    :param df: dataframe
    :param generic_tags: list of generic tags
    :return:
    """
    xml = XML('./Settings/generic_tags.xml')

    rename_dict = dict()
    for generic_tag in generic_tags:
        list_of_subtags = xml.SearchForTag(generic_tag)
        list_of_subtags = list_of_subtags.split(',')
        for subtag in list_of_subtags:
            if subtag in df.columns:
                rename_dict[subtag] = generic_tag
    return rename_dict

def get_generic_column_names_for_plot(df: pd.DataFrame):
    """
    Get generic names for inputs given a specified list of generic inputs
    :param df: dataframe
    :param generic_tags: list of generic tags
    :return:
    """

    rename_dict = dict()
    for name in df.columns:
        print(name)
        if name == 'On_off':
            rename_dict[name] = 'On_off[Boolean]'
        elif name == 'OutletPressure':
            rename_dict[name] = 'OutletPressure[Bar]'
        elif name == 'Current':
            rename_dict[name] = 'Current[A]'
        elif name == 'Torque':
            rename_dict[name] = 'Torque[%]'
        elif name == 'ControlSignal':
            rename_dict[name] = 'ControlSignal[%]'
        elif name == 'Speed':
            rename_dict[name] = 'Speed[%]'
        elif name == 'OutletPressureAvg':
            rename_dict[name] = 'OutletPressureAvg[Bar]'
        elif name == 'CurrentAvg':
            rename_dict[name] = 'CurrentAvg[A]'
        elif name == 'TorqueAvg':
            rename_dict[name] = 'TorqueAvg[%]'
        elif name == 'ControlSignalAvg':
            rename_dict[name] = 'ControlSignalAvg[%]'
        elif name == 'SpeedAvg':
            rename_dict[name] = 'SpeedAvg[%]'
        elif name == 'OutletPressureStd':
            rename_dict[name] = 'OutletPressureStd[Bar]'
        elif name == 'CurrentStd':
            rename_dict[name] = 'CurrentStd[A]'
        elif name == 'TorqueStd':
            rename_dict[name] = 'TorqueStd[%]'
        elif name == 'ControlSignalStd':
            rename_dict[name] = 'ControlSignalStd[%]'
        elif name == 'SpeedStd':
            rename_dict[name] = 'SpeedStd[%]'

    return rename_dict

def get_generic_column_names_for_plot_non_df(tagnames):
    """
    Get generic names for inputs given a specified list of generic inputs
    :param df: dataframe
    :param generic_tags: list of generic tags
    :return:
    """

    rename_dict = []
    for name in tagnames:
        print(name)
        if name == 'On_off':
            rename_dict.append('On_off[Boolean]')
        elif name == 'OutletPressure':
            rename_dict.append('OutletPressure[Bar]')
        elif name == 'Current':
            rename_dict.append('Current[A]')
        elif name == 'Torque':
            rename_dict.append('Torque[%]')
        elif name == 'ControlSignal':
            rename_dict.append('ControlSignal[%]')
        elif name == 'Speed':
            rename_dict.append('Speed[%]')
        elif name == 'OutletPressureAvg':
            rename_dict.append('OutletPressureAvg[Bar]')
        elif name == 'CurrentAvg':
            rename_dict.append('CurrentAvg[A]')
        elif name == 'TorqueAvg':
            rename_dict.append('TorqueAvg[%]')
        elif name == 'ControlSignalAvg':
            rename_dict.append('ControlSignalAvg[%]')
        elif name == 'SpeedAvg':
            rename_dict.append('SpeedAvg[%]')
        elif name == 'OutletPressureStd':
            rename_dict.append('OutletPressureStd[Bar]')
        elif name == 'CurrentStd':
            rename_dict.append('CurrentStd[A]')
        elif name == 'TorqueStd':
            rename_dict.append('TorqueStd[%]')
        elif name == 'ControlSignalStd':
            rename_dict.append('ControlSignalStd[%]')
        elif name == 'SpeedStd':
            rename_dict.append('SpeedStd[%]')

    return rename_dict

def get_features(df):
    features = list(df.loc[:, df.columns != 'Time'])
    not_features = search_string_like_in_list('State', features)
    try:
        features.remove(not_features)
    except ValueError as e:
        print(e)
    return features

def equalize_tag_length(tags):
    """Takes a list of tags and makes sure they have measurement lists of equal length

    Args:
        tags (list[Tag]): List of tags

    Returns:
        list[Tag]: List of same-length tags
    """
    #double amount of measurements of the first tag
    #Instead dont x2 beause we want the shortest
    shortest_meas_length = len(tags[0].measurements)*2

    # for each tag
    for i in range(len(tags)):
        print(len(tags[i].measurements))
        #
        amount_of_measurements = len(tags[i].measurements)
        if amount_of_measurements < shortest_meas_length:
            shortest_meas_length = amount_of_measurements
            print('new shortest_meas_length =  {}'.format(shortest_meas_length))

    print('final shortest_meas_length = {}'.format(shortest_meas_length))

    b = 0
    #for each tag
    for i in range(len(tags)):
        current_meas_length = len(tags[i].measurements)
        print(tags[i].tag)
        print(current_meas_length > shortest_meas_length)
        if current_meas_length > shortest_meas_length:
            b = shortest_meas_length-current_meas_length
            #If this measurement length is longer than the shortest
            #Perhaps this should be b<0
            if b <-1:
                #Replaces the measurements of a tag with a sliced version of itself
                #lets say b = -20
                # -> measurements[1:b+1] will make a copy that does not include the first value and stops at len-20-1+1
                tags[i].measurements = tags[i].measurements[1:b+1]
                tags[i].timestamp = tags[i].timestamp[1:b+1]
            else:
                #Deletes the last element
                del tags[i].measurements[-1]
                del tags[i].timestamp[-1]
            #Looks like there might be something wrong here
            #My best guess is that at least the first if should be b<0, and slice from 0 to b-1



    #trimmed?
    print("Fjernet")

    #for each tag
    for i in range(len(tags)):
        #Prints length of measurements for each tag to check if they are the same.
        print(len(tags[i].measurements))
    return tags

def quantile_remove_up(df, quantile, except_col):

    cols = df.columns.to_list()
    use_cols = [x for x in cols if x not in except_col]
    tmp = df.loc[:, use_cols]

    df.drop(columns=use_cols, inplace=True)
    outliers = tmp.quantile(quantile, numeric_only=True)

    filt_df = tmp.apply(lambda x: x[(x < outliers.loc[x.name])], axis=0)
    filt_df.to_csv('filt_df.csv')

    #df = merge_df(df, filt_df)

    for x in except_col:
        try:
            filt_df = pd.concat([df.loc[:, [x]], filt_df], axis=1)

        except KeyError as e:
            print(e)

    filt_df = filt_df.fillna(method='ffill')
    #filt_df.dropna(axis=0, inplace=True)

    #filt_df.fillna(method='ffill', axis=1)
    inserted_cols = ['Time', 'On_off', 'OutletPressure', 'Current', 'Torque', 'ControlSignal', 'Speed', 'State',
                                'OutletPressureAvg', 'CurrentAvg', 'TorqueAvg', 'ControlSignalAvg', 'SpeedAvg',
                                'OutletPressureStd', 'CurrentStd', 'TorqueStd', 'ControlSignalStd', 'SpeedStd', 'State']
    cols = ([col for col in inserted_cols if col in filt_df]
            + [col for col in filt_df if col not in inserted_cols])
    filt_df = filt_df[cols]


    return filt_df


def quantile_remove_down(df, quantile, except_col):

    cols = df.columns.to_list()
    use_cols = [x for x in cols if x not in except_col]
    tmp = df.loc[:, use_cols]

    df.drop(columns=use_cols, inplace=True)
    outliers = tmp.quantile(quantile, numeric_only=True)

    filt_df = tmp.apply(lambda x: x[(x > outliers.loc[x.name])], axis=0)
    filt_df.to_csv('filt_df.csv')

    #df = merge_df(df, filt_df)

    for x in except_col:
        try:
            filt_df = pd.concat([df.loc[:, [x]], filt_df], axis=1)

        except KeyError as e:
            print(e)

    filt_df = filt_df.fillna(method='ffill')
    #filt_df.dropna(axis=0, inplace=True)

    #filt_df.fillna(method='ffill', axis=1)
    inserted_cols = ['Time', 'On_off', 'OutletPressure', 'Current', 'Torque', 'ControlSignal', 'Speed', 'State',
                                'OutletPressureAvg', 'CurrentAvg', 'TorqueAvg', 'ControlSignalAvg', 'SpeedAvg',
                                'OutletPressureStd', 'CurrentStd', 'TorqueStd', 'ControlSignalStd', 'SpeedStd', 'State']
    cols = ([col for col in inserted_cols if col in filt_df]
            + [col for col in filt_df if col not in inserted_cols])
    filt_df = filt_df[cols]


    return filt_df



def remove_on_off(df):
    df = df[df['On_off'] != 0]
    df = df.drop('On_off', 1)
    return df


if __name__ == '__main__':
    #test_split_sequence()
    a = ['test_one', 'test_two', 'nope']
    b = 'test'
    c = 4
    print(search_string_like_in_list(b, a))
    print(assert_length_in_list_of_string(c, a))
    print(search_string_exact_in_list(b, a))
    print(search_string_exact_in_list('nope', a))
