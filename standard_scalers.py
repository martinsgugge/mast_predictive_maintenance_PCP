import datetime
import os
from os.path import exists

import numpy
import pandas
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly as pl
import plotly.express as px
import kaleido

import plots
from XML import XML
from database_high_level import *
from postgres import Tag, psql
from pickle import dump, load
from array_manipulation import get_generic_column_names, merge_df, get_generic_column_names_for_plot


def generate_previous_standard_scalers(pump_no_tag: str, table: str, start: str, end: str, settings_file: str, agg_time: int = None,
                                       avg: bool = False, std_dev: bool = False):
    """
    Create one scaler per physical pump found between start and stop time for the given pump tag
    :param pump_no_tag: name of tag which follows pump number
    :param table: name of table for process data
    :param start: where to start look for pump failures
    :param end: where to stop look for pump failures
    :param agg_time: How much time should be aggregated over
    :param avg: True if average is used as aggregation
    :param std_dev: True if standard deviation is used as aggregation
    :return: file in form of pickle object named after pump name and date of new pump
    """
    computer_specifics = XML(f'./Settings/{settings_file}')
    model_name = 'Scaler raw avg std'
    meta = Metadata(model_name)
    meta.feature_type = ['On_off', 'ControlSignal', 'Current', 'Torque', 'OutletPressure',
                         'ControlSignalAvg', 'CurrentAvg', 'TorqueAvg', 'OutletPressureAvg',
                         'ControlSignalStd', 'CurrentStd', 'TorqueStd', 'OutletPressureStd']
    meta.raw = bool(int(computer_specifics.SearchForTag('raw')))
    meta.aggregation_time = int(computer_specifics.SearchForTag('aggregation_time'))
    meta.average = bool(int(computer_specifics.SearchForTag('average')))
    meta.standard_deviation = bool(int(computer_specifics.SearchForTag('standard_deviation')))

    pump_nos = get_pump_numbers(start, end, pump_no_tag)
    print(f'avg={meta.average} std={meta.standard_deviation} agg_time={meta.aggregation_time}')
    # Get process measurement data
    for timestamp in pump_nos.timestamp:
        # print(timestamp)
        relevant_tagnames = get_relevant_pump_tags_general(pump_nos.tag, meta)

        #print(relevant_tagnames)
        # Get pump data from timestamp + 1hour
        if table == 'measurement' or table=='measurement2':
            extra_time = 12
        elif table == 'high_speed_big_reduced':
            extra_time = 1

        if not exists(f'./scalers/scaler {table} {tag} {str(timestamp)[:10]} raw={meta.raw} avg={meta.average} '
                       f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.pkl'):
            data_exists = False

            while not data_exists:
                # df = get_process_data(agg_time, avg, std_dev, relevant_tagnames, table,
                #                       timestamp + datetime.timedelta(hours=extra_time-12),
                #                       timestamp + datetime.timedelta(hours=extra_time))
                meta.data_training_stop = timestamp + datetime.timedelta(hours=extra_time)
                meta.data_training_start = timestamp + datetime.timedelta(hours=extra_time-12)
                if meta.raw:
                    df = get_process_data_for_physical_pump(pump_no_tag, meta, table=table, get_state=False)

                if meta.average:
                    df_avg = get_process_data_for_physical_pump(pump_no_tag, meta, table=table,
                                                                agg_time=22, avg=True, get_state=False)
                    if not meta.raw:
                        df = df_avg
                if meta.standard_deviation:
                    df_std = get_process_data_for_physical_pump(pump_no_tag, meta, table=table,
                                                                agg_time=22, std_dev=True, get_state=False)
                if not meta.failed:
                    if meta.average and meta.raw:
                        df = merge_df(df, df_avg, method='nearest')

                    if meta.standard_deviation:
                        df = merge_df(df, df_std, method='nearest')

                if df is None or df.empty:
                    print('getting more time')
                    extra_time += 12
                else:
                    print(df.head())
                    data_exists = True

                # if df.empty:
                #     print('getting more time')
                #     extra_time += 1
                # else:
                #     print(df.head())
                #     data_exists = True

            # Create the scaler use only date for naming
            #df = df.loc[:, df.columns != 'Time']
            #df = df.loc[:, df.columns != 'PU19']
            features = list(df.loc[:, df.columns != 'Time'])
            #generic_tags = ['OutletPressure', 'Current', 'Torque', 'ControlSignal', 'Speed', 'On_off']
            generic_tags = ['On_off', 'OutletPressure', 'Current', 'Torque', 'ControlSignal', 'Speed', 'State',
                            'OutletPressureAvg', 'CurrentAvg', 'TorqueAvg', 'ControlSignalAvg', 'SpeedAvg',
                            'OutletPressureStd', 'CurrentStd', 'TorqueStd', 'ControlSignalStd', 'SpeedStd']
            rename_dict = get_generic_column_names(df, generic_tags)
            df = df.rename(columns=rename_dict)
            #rename_dict = get_generic_column_names_for_plot(df)
            #df = df.rename(columns=rename_dict)
            if meta.raw:
                df['ControlSignal'] = numpy.linspace(0, 100, len(df['ControlSignal']))
            if meta.average:
                df['ControlSignalAvg'] = numpy.linspace(0, 100, len(df['ControlSignalAvg']))
            if meta.standard_deviation:
                df['ControlSignalStd'] = numpy.linspace(0, 100, len(df['ControlSignalStd']))

            print('Going into scaler creation')

            create_standard_scaler(df, table, tag[:8], str(timestamp)[:10], features, meta)


def create_standard_scaler(data: pandas.DataFrame, table: str, tag: str, date: str, features_org: list, meta):
    """
    Creates a standard scaler and write to file named using the tag and date
    :param data: Data to create the scaler
    :param tag: name of tag
    :param date: date of when pump was put in production
    :return:
    """
    scaler = StandardScaler()
    try:
        features = list(data.loc[:, data.columns != 'Time'])
        scale_features = features
        scale_features.remove('On_off')
    except ValueError as e:
        print(e)

    try:
        fig = px.line(data, x='Time', y=features, title=f'before scale{table}{tag}{date[:10]}')
        scaler.fit(data[scale_features])
        if not os.path.exists("scaler_images"):
            os.mkdir("scaler_images")
        plots.plot(f'./scaler_images/before scale{table}{tag}{date[:10]} raw={meta.raw} avg={meta.average} '
                       f'std={meta.standard_deviation} agg_time={meta.aggregation_time}', features, data, single_time=True)
        # fig.write_html(f'./scaler_images/before scale{table}{tag}{date[:10]} raw={meta.raw} avg={meta.average} '
        #                f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.html')
        print(f'./scalers/scaler {table} {tag} {date[:10]} raw={meta.raw} avg={meta.average} '
                       f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.pkl')
        dump(scaler, open(f'./scalers/scaler {table} {tag} {date[:10]} raw={meta.raw} avg={meta.average} '
                       f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.pkl', 'wb'))
        data[features] = scaler.transform(data[features])

        plots.plot(f'./scaler_images/after scale{table}{tag}{date[:10]} raw={meta.raw} avg={meta.average} '
                       f'std={meta.standard_deviation} agg_time={meta.aggregation_time}', features, data, single_time=True)
        # fig = px.line(data, x='Time', y=features, title=f'after scale{tag}{date[:10]}')
        # fig.write_html(f'./scaler_images/after scale{table}{tag}{date[:10]} raw={meta.raw} avg={meta.average} '
        #                f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.html')
        store_scaler_data(data, date, features_org, scaler, table, tag)

    except ValueError as e:
        print(e)





def read_standard_scaler(table: str, tag: str, date: str, meta: Metadata):
    """
    Reads a standard scaler using the tagname in combination with date
    :param tag: name of tag
    :param date: date of when pump was put in production
    :return:
    """
    try:
        with open(f'./scalers/scaler {table}2 {tag} {date[:10]} raw={meta.raw} avg={meta.average} '
                  f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.pkl', 'rb') as f:
            scaler = load(f)
    except FileNotFoundError as e:
        with open(f'./scalers/scaler {table} {tag} {date[:10]} raw={meta.raw} avg={meta.average} '
                           f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.pkl', 'rb') as f:
            scaler = load(f)

    return scaler


def check_if_scaler_exists(meta, pump_no_tag, scaler_table, timestamp):

    print(f'./scalers/scaler {scaler_table}2 {pump_no_tag[:8]} {str(timestamp)[:10]} raw={meta.raw} avg={meta.average} '
          f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.pkl')
    if not exists(
            f'./scalers/scaler {scaler_table}2 {pump_no_tag[:8]} {str(timestamp)[:10]} raw={meta.raw} avg={meta.average} '
            f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.pkl') \
            and not exists(
            f'./scalers/scaler {scaler_table} {pump_no_tag[:8]} {str(timestamp)[:10]} raw={meta.raw} avg={meta.average} '
            f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.pkl'):
        meta.failed = True
        print('Scaler does not exist')


if __name__ == '__main__':
    #TODO undersøke hvordan on_off blir seende ut etter skalering
    file = XML('./Settings/table_config.xml')
    table = file.SearchForTag('table')
    high_table = file.SearchForTag('high_table')
    print(high_table)
    pump_state_list = [ "HYG_PU19_Pump_no",
                       #"HYG_PU18_Pump_no",
                       "HYG_PU17_Pump_no",
                       "HYG_PU16_Pump_no",
                       "HYG_PU15_Pump_no",
                       #"HYG_PU14_Pump_no",
                       "HYG_PU13_Pump_no",
                       "HYG_PU12_Pump_no"
                       ]



    # for tag in pump_state_list:
    #     start = '2020-03-01'
    #     end = '2022-03-01'
    #
    #     generate_previous_standard_scalers(tag, table, start, end)
    agg_time = 22
    avg = True
    std = True
    settings_file = ['LSTM model_settings exp 1.xml', 'LSTM model_settings exp 2.xml', 'LSTM model_settings exp 2.xml',
                     'LSTM model_settings exp 4.xml', 'LSTM model_settings exp 5.xml', 'LSTM model_settings exp 6.xml',
                     'LSTM model_settings exp 7.xml', 'LSTM model_settings exp 8.xml', 'LSTM model_settings exp 9.xml',
                     'GRU model_settings exp 1.xml', 'GRU model_settings exp 2.xml', 'GRU model_settings exp 2.xml',
                     'GRU model_settings exp 4.xml', 'GRU model_settings exp 5.xml', 'GRU model_settings exp 6.xml',
                     'GRU model_settings exp 7.xml', 'GRU model_settings exp 8.xml', 'GRU model_settings exp 9.xml'
                     ]
    settings_file = ['GRU model_settings exp 1.xml', #'GRU model_settings exp 2.xml', 'GRU model_settings exp 2.xml',
                     #'GRU model_settings exp 4.xml', 'GRU model_settings exp 5.xml', 'GRU model_settings exp 6.xml',
                     #'GRU model_settings exp 7.xml', 'GRU model_settings exp 8.xml', 'GRU model_settings exp 9.xml'
    ]
    for file in settings_file:
        for tag in pump_state_list:
            start = '2020-03-01'
            end = '2022-04-01'
            print(tag)
            generate_previous_standard_scalers(tag, table, start, end, file, avg=avg, std_dev=std, agg_time=agg_time)

            start = '2020-03-01'
            end = '2022-02-01'
            #generate_previous_standard_scalers(tag, high_table, start, end)


