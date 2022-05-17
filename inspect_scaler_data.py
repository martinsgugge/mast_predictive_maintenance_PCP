import datetime
import os
from os.path import exists
from sklearn.model_selection import train_test_split

from plots import *
from XML import XML
from database_high_level import get_process_data_for_physical_pump, get_pump_numbers, update_metadata_record_LSTM, \
    delete_metadata_record_LSTM, insert_new_metadata_record_LSTM, get_previous_run_data, query_highest_nearest_state
from standard_scalers import read_standard_scaler
import pandas as pd
from array_manipulation import get_generic_column_names, search_string_like_in_list, get_features, merge_df, \
    convert_to_date_or_datetime, remove_quantiles, quantile_remove_up, quantile_remove_down, remove_on_off
from LSTM import *
from metadata import MetadataLSTM
from postgres import psql
from pre_process import pre_process_LSTM
from PCA_PU19 import pca
from get_data import get_data, get_data_viz


def gauge_scaled_data(start, end, pump_no_tag, checkpoint_path, table, avg=False, std=False):

    # Metadata
    computer_specifics = XML('./Settings/model_settings.xml')
    model_name = computer_specifics.SearchForTag('ModelName')
    meta = MetadataLSTM(model_name)

    meta.feature_type = ['On_off', 'ControlSignal', 'Current', 'Speed', 'Torque', 'OutletPressure',
                         'ControlSignalAvg', 'CurrentAvg', 'TorqueAvg', 'OutletPressureAvg',
                         'ControlSignalStd', 'CurrentStd', 'TorqueStd', 'OutletPressureStd']

    meta.use_all = bool(int(computer_specifics.SearchForTag('use_all')))
    meta.aggregation_time = int(computer_specifics.SearchForTag('aggregation_time'))
    meta.average = bool(int(computer_specifics.SearchForTag('average')))
    meta.standard_deviation = bool(int(computer_specifics.SearchForTag('standard_deviation')))
    meta.raw = bool(int(computer_specifics.SearchForTag('raw')))
    meta.derivatives = bool(int(computer_specifics.SearchForTag('derivatives')))
    meta.feature_type = ['On_off', 'ControlSignal', 'Current', 'Speed', 'Torque', 'OutletPressure',
                         'ControlSignalAvg', 'CurrentAvg', 'TorqueAvg', 'OutletPressureAvg',
                         'ControlSignalStd', 'CurrentStd', 'TorqueStd', 'OutletPressureStd']

    # pre-process parameters
    meta.sequence = False
    meta.train_size = float(computer_specifics.SearchForTag('train_size'))
    meta.test_size = float(computer_specifics.SearchForTag('test_size'))
    meta.validation_size = float(computer_specifics.SearchForTag('validation_size'))

    pump_nos = get_pump_numbers(start, end, pump_no_tag)

    sql = psql()

    for timestamp, measurement in zip(pump_nos.timestamp, pump_nos.measurements):
        meta.failed = False
        timestamp = timestamp.replace(tzinfo=None)
        scaler_table = find_scaler_table(table)
        local_end = query_highest_nearest_state(timestamp, end, pump_no_tag, meta)

        if not meta.failed:
            meta.data_training_stop = local_end.replace(tzinfo=None) + datetime.timedelta(days=1)
            meta.data_training_start = timestamp

        if not exists(f'./Pump_data_statistics/Correlation {pump_nos.tag} {str(measurement)} {table} '
                      f'{str(meta.data_training_start)[:10]} to {str(meta.data_training_stop)[:10]}'
                      f' raw={meta.raw} avg={meta.average} '
                       f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.html'):
            if not exists(f'./scalers/scaler {scaler_table}2 {pump_no_tag[:8]} {str(timestamp)[:10]} raw={meta.raw} avg={meta.average} '
                       f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.pkl'):
                meta.failed = True
                print('Scaler does not exist')

            if not meta.failed:
                df = get_data_viz(avg, end, meta, pump_no_tag, std, table, timestamp)

            if not meta.failed:
                meta.features = get_features(df)
                meta.no_of_features = len(meta.features)
                print(f'THESE ARE THE FEATURES: {meta.features}')

                # Convert names to standard names
                generic_tags = ['On_off', 'OutletPressure', 'Current', 'Speed', 'Torque', 'ControlSignal', 'State',
                                'OutletPressureAvg', 'CurrentAvg', 'TorqueAvg', 'ControlSignalAvg', 'SpeedAvg',
                                'OutletPressureStd', 'CurrentStd', 'TorqueStd', 'ControlSignalStd', 'SpeedStd']
                rename_dict = get_generic_column_names(df, generic_tags)
                df = df.rename(columns=rename_dict)
                if not exists(
                        f'./line_plots/new font_Lineplot inspect_scaler_data {table} {tag} {str(timestamp)[:10]} raw={meta.raw} avg={meta.average} '
                        f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.html'):
                    plot(
                        f'./line_plots/font_Lineplot inspect_scaler_data {table} {tag} {str(timestamp)[:10]} raw={meta.raw} avg={meta.average} '
                        f'std={meta.standard_deviation} agg_time={meta.aggregation_time}', meta.features,
                        df, single_time=True)
                # Scale data frames using created scalers
                scaler_features = list(df.loc[:, df.columns != 'Time'])
                scaler_features.remove('On_off')
                scaler_features.remove('State')

                scaler = read_standard_scaler(scaler_table, pump_no_tag[:8], str(timestamp)[:10], meta)
                print(f'./scalers/scaler {table}2 {tag} {str(timestamp)[:10]} raw={meta.raw} avg={meta.average} '
                  f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.pkl')

                print(f'./scalers/scaler {table} {tag} {str(timestamp)[:10]} raw={meta.raw} avg={meta.average} '
                           f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.pkl')
                df[scaler_features] = scaler.transform(df[scaler_features])
                if not exists(
                        f'./line_plots/new font_Lineplot scaled inspect_scaler_data {table} {tag} {str(timestamp)[:10]} raw={meta.raw} avg={meta.average} '
                        f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.html'):
                    plot(
                        f'./line_plots/font_Lineplot scaled inspect_scaler_data {table} {tag} {str(timestamp)[:10]} raw={meta.raw} avg={meta.average} '
                        f'std={meta.standard_deviation} agg_time={meta.aggregation_time}', meta.features,
                        df, single_time=True)
                # Split data into train, validation and test set
                train, validation = train_test_split(df, test_size=meta.validation_size + meta.test_size,
                                                     random_state=42)
                validation, test = train_test_split(validation,test_size=(meta.test_size / meta.validation_size) / 2,
                                                    random_state=42)
                train = remove_on_off(train)
                train = remove_quantiles(train, 0.99, 0.01, ['Current', 'Torque'])
                train = remove_quantiles(train, 1, 0.012, ['Speed'])


                histogram_ly(train.copy(deep=True), pump_nos.tag + str(measurement) + table + ' ' +
                             str(meta.data_training_start)[:10] + ' to ' + str(meta.data_training_stop)[:10])
                correlation_plots(train.copy(deep=True),
                                pump_nos.tag + str(measurement) + table + ' ' + str(meta.data_training_start)[:10] +
                                  ' to ' + str(meta.data_training_stop)[:10])
                #pca(train.copy(deep=True), 3, pump_nos.tag + str(measurement) + table + ' ' +
                    #str(meta.data_training_start)[:10] + ' to ' + str(meta.data_training_stop)[:10])
        else:
            print('already exists')


def find_scaler_table(table):
    """
    Used to get correct name for table in scalers no matter which computer is used
    :param table:
    :return:
    """
    if table == 'measurement2':
        scaler_table = 'measurement'
    elif table == 'measurement':
        scaler_table = 'measurement'
    else:
        scaler_table = 'high_speed_big_reduced'
    return scaler_table


if __name__ == '__main__':
    laptop = False
    if laptop:
        date_start = '2021-09-01'
        date_stop = '2021-10-01'
    else:
        date_start = '2021-05-21'
        date_stop = '2022-02-01'

    checkpoint_path = './checkpoints/'
    file = XML('./Settings/table_config.xml')
    table = file.SearchForTag('table')
    high_table = file.SearchForTag('high_table')
    pump_state_list = [
    #"HYG_PU19_Pump_no",
        #"HYG_PU18_Pump_no",
    "HYG_PU17_Pump_no",
    #"HYG_PU16_Pump_no", "HYG_PU15_Pump_no",
    #    "HYG_PU14_Pump_no",
    #"HYG_PU13_Pump_no", "HYG_PU12_Pump_no"
                       ]
    for tag in pump_state_list:
        gauge_scaled_data(date_start, date_stop, tag, checkpoint_path, table)
