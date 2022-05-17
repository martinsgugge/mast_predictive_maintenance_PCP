from math import sqrt, factorial, exp
import numpy as np
import statistics as stats
import datetime
import sklearn.decomposition as sk
#from Lesing_og_plotting import plot
import plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import pandas as pd

import math


"""Lib for statistical functions
Author: Martin Holm
Created: 03.09.2020
"""


def mean(data):
    sum = 0
    for i in range(0, len(data)):
        sum += float(data[i])
    mean = sum / len(data)

    return mean


def test_mean():
    data = [1, 2, 3, 4, 5]
    mean_val = mean(data)
    print("mean of: ", (data))
    print("is: ", mean_val)
    data = [2, 4, 6, 8, 10]
    mean_val = mean(data)
    print("mean of: ", (data))
    print("is: ", mean_val)
    data = [3, 6, 9, 12, 15]
    mean_val = mean(data)
    print("mean of: ", (data))
    print("is: ", mean_val)


def variance(data):
    sum = 0
    mean_val = mean(data)
    # sum deviation
    for i in range(len(data)):
        sum += (data[i] - mean_val) ** 2
    # divide
    var = sum / len(data)

    return var

def variance_per_point(data):
    sum = 0
    mean_val = mean(data)
    var = []
    # sum deviation
    for i in range(len(data)):
        var.append((data[i] - mean_val) ** 2)
    # divide
    #var = sum / len(data)

    return var


def test_variance():
    data = [1, 2, 3, 4, 5]
    var = variance(data)
    print("var of: ", (data))
    print("is: ", var)
    data = [2, 4, 6, 8, 10]
    var = variance(data)
    print("var of: ", (data))
    print("is: ", var)
    data = [3, 6, 9, 12, 15]
    var = variance(data)
    print("var of: ", (data))
    print("is: ", var)


def standard_deviation(data):
    var = variance(data)
    # square root
    sd = sqrt(var)

    return sd
def pop_standard_deviation(data):
    sigma = 0
    avg = mean(data)
    for i in range(len(data)):
        sigma += (data[i]-avg)**2
    sigma /= len(data)
    sigma = sqrt(sigma)
    return sigma

def pop_standard_deviation_per_point(data):
    sigma = []
    avg = mean(data)
    for i in range(len(data)):
        sigma.append(data[i]-avg)
    #sigma /= len(data)
    #sigma = sqrt(sigma)
    return sigma

def sample_standard_deviation(data):
    sigma = 0
    avg = mean(data)
    for i in range(len(data)):
        sigma += (data[i]-avg)**2
    sigma /= len(data)-1
    sigma = sqrt(sigma)
    return sigma



def test_standard_deviation():
    data = [1, 2, 3, 4, 5]
    sd = standard_deviation(data)
    print("sd of: ", (data))
    print("is: ", sd)
    data = [2, 4, 6, 8, 10]
    sd = standard_deviation(data)
    print("sd of: ", (data))
    print("is: ", sd)
    data = [3, 6, 9, 12, 15]
    sd = standard_deviation(data)
    print("sd of: ", (data))
    print("is: ", sd)


def median(data):
    data = np.array(data)
    data = np.sort(data)
    x = np.median(data)
    length = len(data)
    if length % 2 == 0:
        i_one = int(length / 2 - 1)
        i_two = int(length / 2)
        median = (data[i_one] + data[i_two]) / 2
    else:
        median = data[len(data) % 2]

    return x


def test_median():
    data = [3, 2, 1, 4, 5, 6]
    median_val = median(data)
    print("median_val of: ", (data))
    print("is: ", median_val)
    data = [4, 6, 2, 8, 10]
    median_val = median(data)
    print("median_val of: ", (data))
    print("is: ", median_val)
    data = [3, 12, 9, 6, 15]
    median_val = median(data)
    print("median_val of: ", (data))
    print("is: ", median_val)

def zero_order_hold(data, alignData, return_list=True):
    """
    Author: Martin Holm
    Created: 21.05.2021
    Last Edited: 21.05.2021
    Aligns a boolean signals length to an analog tags length

    :param data: [Time, value], Array of boolean values and its timestamps
    :param alignData: Time, List or array of time which data should be aligned to
    :return: Double list [time, data]
    """
    print(data[1])
    try:
        meas = [data[1][0]]
    except IndexError as e:
        print(e)
        meas = [0]

    meas_time = alignData
    data = list([list(data[0]),list(data[1])])
    for i in range(len(meas_time)):
        try:

            if meas_time[i] >= data[0][0]:
                meas.append(data[1][0])

                del data[1][0]
                del data[0][0]
            else:
                meas.append(meas[i])
        except IndexError as e:
            print(e)
            meas.append(meas[i])
    if return_list == True:
        return [meas_time, meas]
    else:
        return meas_time, meas

def zero_order_hold2(x, y, align_data):
    df = pd.DataFrame({'time': [align_data]})

    f = interp1d(pd.to_datetime(df['time']).astype(int)/ 10**9,y, kind='zero', axis=0)
    df = pd.DataFrame({'time': [align_data]})
    #print(np.datetime_data(new_x).astype(np.long))
    #new_x = np.datetime64(new_x.astype(np.long))
    new_y = f(pd.to_datetime(df['time']).astype(int)/ 10**9)

    return align_data, new_y

def convert_to_date_or_datetime(datetime_var):
    if isinstance(datetime_var, str):
        try:
            datetime_result = datetime.datetime.strptime(datetime_var, '%Y-%m-%d')
        except ValueError as e:
            try:
                datetime_result = datetime.datetime.strptime(datetime_var, '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                print(e)
                datetime_result = datetime.datetime.strptime(datetime_var, '%Y-%m-%d %H:%M:%S.%f')

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
    if type(date_start) is not datetime.datetime or not pd.Timestamp:

        try:
            date_stop = datetime.datetime.strptime(date_stop, '%Y-%m-%d')
        except ValueError as e:
            try:
                date_stop = datetime.datetime.strptime(date_stop, '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                print(e)
                date_stop = datetime.datetime.strptime(date_stop, '%Y-%m-%d %H:%M:%S.%f')
        try:
            date_start = datetime.datetime.strptime(date_start, '%Y-%m-%d')

        except ValueError as e:
            try:
                date_start = datetime.datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S')

            except ValueError as e:
                print(e)
                date_start = datetime.datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S.%f')


    list_ms = []
    list_ms.append(date_start)

    # Makes list of days interval
    if day:
        while list_ms[-1] < date_stop:
            list_ms.append(list_ms[-1] + datetime.timedelta(days=interval))

    # Makes list of hours interval
    elif hour:
        while list_ms[-1] < date_stop:
            list_ms.append(list_ms[-1] + datetime.timedelta(hours=interval))

    # Makes list of minutes interval
    elif min:
        while list_ms[-1] < date_stop:
            list_ms.append(list_ms[-1] + datetime.timedelta(minutes=interval))

    # makes list of seconds interval
    elif second:
        while list_ms[-1] < date_stop:
            list_ms.append(list_ms[-1] + datetime.timedelta(seconds=interval))

    # Makes lists of ms intervals
    elif ms:
        while list_ms[-1] < date_stop:
            list_ms.append(list_ms[-1] + datetime.timedelta(milliseconds=interval))

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

def plot_single_time(filename, tagname, data, right_hand_axis=None):
    """
    Author: Martin Holm
    Date: 03.04.2020
    Last edited: 29.10.2020
    Base of original code: https://plotly.com/python/line-and-scatter/

    :param filename: Name of output file
    :param tagname: Name of data series to plot
    :param data: List with lists of time and data series to be plotted in the form [time, data, time, data..etc]
    :param right_hand_axis: Optional, the name of data series to have a right-hand y-axis
    :return: HTML file with data plotted
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    plotted = False

    for i in range(1, len(tagname)):
        print(i)
        plotted = False

        if right_hand_axis is not None:
            for j in range(len(right_hand_axis)):
                if right_hand_axis[j] in tagname[i]:
                    fig.add_trace(go.Scatter(x=data[0], y=data[i], name=tagname[i]), secondary_y=True)
                    plotted = True

        if plotted is False:
            #print(data[i], tagname[i])
            fig.add_trace(go.Scatter(x=data[0], y=data[i], name=tagname[i]), secondary_y=False)

    py.offline.plot(fig, filename=filename+'.html')

def plot(filename, tagname, data, right_hand_axis=None):
    """
    Author: Martin Holm
    Date: 03.04.2020
    Base of original code: https://plotly.com/python/line-and-scatter/

    :param filename: Name of output file
    :param tagname: Name of data series to plot
    :param data: List with lists of time and data series to be plotted in the form [time, data, time, data..etc]
    :param right_hand_axis: Optional, the name of data series to have a right-hand y-axis
    :return: HTML file with data plotted
    """

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    plotted = False

    for x in data:
        print(x)
    for i in range(len(tagname)):
        print(i)
        plotted = False

        if right_hand_axis is not None:
            for j in range(len(right_hand_axis)):
                if right_hand_axis[j] in tagname[i]:
                    fig.add_trace(go.Scatter(x=data[i*2], y=data[i*2+1], name=tagname[i]), secondary_y=True)
                    plotted = True

        if plotted is False:
            fig.add_trace(go.Scatter(x=data[i*2], y=data[i*2+1], name=tagname[i]), secondary_y=False)

    py.offline.plot(fig, filename=filename+'.html')


def average_hourly(time, meas):
    """
    Creates list of hourly averages

    :param time: list of datetime
    :param meas: list of float, measurements
    :return: hourly averages and new time list
    """
    sum = np.zeros([len(meas)])
    no_hours = 0

    for i in range(len(meas) - 1):
        if (time[i].hour) > (time[i - 1].hour):
            no_hours += 1

    no_hours += 1
    new_time = []
    hourly_avg = np.zeros([no_hours])

    no_hours = 0
    start_index = 0
    stop_index = 0
    for i in range(1, len(meas) - 1):

        if (time[i].hour) > (time[i - 1].hour):

            new_time.append(time[i - 1].replace(minute=0, second=0, microsecond=0))
            hourly_avg[no_hours] = (sum[i - 1])/len(sum[start_index:stop_index])
            no_hours += 1
            start_index = i
            last = False
        elif (time[i].hour) == (time[i - 1].hour):
            sum[i] = sum[i - 1] + meas[i]
            stop_index = i

    new_time.append(time[i - 1].replace(minute=0, second=0, microsecond=0))
    hourly_avg[no_hours] = (sum[i]) / len(sum[start_index:stop_index])
    #print('Lengde fra average hourly', len(new_time), len(hourly_avg))
    return hourly_avg, new_time



