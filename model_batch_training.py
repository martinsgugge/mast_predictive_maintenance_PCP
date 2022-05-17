import datetime
import os
from pprint import pprint

import plots
from XML import XML
from database_high_level import get_process_data_for_physical_pump, get_pump_numbers, update_metadata_record_LSTM, \
    delete_metadata_record_LSTM, insert_new_metadata_record_LSTM, get_previous_run_data, query_highest_nearest_state, \
    update_metadata_test_set
from standard_scalers import read_standard_scaler, check_if_scaler_exists
import pandas as pd
from array_manipulation import get_generic_column_names, search_string_like_in_list, get_features, merge_df, convert_to_date_or_datetime
from LSTM import *
from metadata import MetadataLSTM
from postgres import psql
from pre_process import pre_process_LSTM, rename_and_scale, skip_type_of_pumps
from outliers import *
from PCA_PU19 import pca
from get_data import get_data
from os.path import exists
from calculation import Calculation

def main_training_pipeline(start, end, pump_no_tag, checkpoint_path, table, settings_file, avg=False, std=False,
                           testing=False, skip_type_1_pump=False, skip_type_2_pump=False, use_csv_file=False):
    # Metadata
    computer_specifics = XML(f'./Settings/{settings_file}')
    model_name = computer_specifics.SearchForTag('ModelName')
    meta = MetadataLSTM(model_name)

    meta.table = table
    # data params
    meta.feature_type = ['On_off', 'ControlSignal', 'Current', 'Torque', 'OutletPressure',
                         'ControlSignalAvg', 'CurrentAvg', 'TorqueAvg', 'OutletPressureAvg',
                         'ControlSignalStd', 'CurrentStd', 'TorqueStd', 'OutletPressureStd']
    meta.derivative_pairs = [
        {
            "arg1": "ControlSignal",
            "arg2": "Current",
            "operator": "/"
        },
        {
            "arg1": "ControlSignal",
            "arg2": "OutletPressure",
            "operator": "/"
        },
        {
            "arg1": "ControlSignal",
            "arg2": "Torque",
            "operator": "/"
        },
        {
            "arg1": "Current",
            "arg2": "OutletPressure",
            "operator": "/"
        }
    ]
    meta.use_all = bool(int(computer_specifics.SearchForTag('use_all')))
    meta.aggregation_time = int(computer_specifics.SearchForTag('aggregation_time'))
    meta.average = bool(int(computer_specifics.SearchForTag('average')))
    meta.standard_deviation = bool(int(computer_specifics.SearchForTag('standard_deviation')))
    meta.derivatives = bool(int(computer_specifics.SearchForTag('derivatives')))
    meta.raw = bool(int(computer_specifics.SearchForTag('raw')))

    # pre-process parameters
    meta.sequence = bool(int(computer_specifics.SearchForTag('sequence')))
    meta.sequence_steps = int(computer_specifics.SearchForTag('sequence_steps'))
    meta.pad = bool(int(computer_specifics.SearchForTag('pad')))
    meta.train_size = float(computer_specifics.SearchForTag('train_size'))
    meta.test_size = float(computer_specifics.SearchForTag('test_size'))
    meta.validation_size = float(computer_specifics.SearchForTag('validation_size'))

    #Model parameters
    meta.number_of_epochs = int(computer_specifics.SearchForTag('number_of_epochs'))
    meta.batch_size = int(computer_specifics.SearchForTag('batch_size'))
    meta.neurons_layer_one = int(computer_specifics.SearchForTag('neurons_layer_one'))  # LSTM blocks
    meta.neurons_layer_two = int(computer_specifics.SearchForTag('neurons_layer_two')) # LSTM blocks
    try:
        meta.neurons_layer_three = int(computer_specifics.SearchForTag('neurons_layer_three'))  # LSTM blocks
    except UnboundLocalError as e:
        print(e)
    meta.layers = int(computer_specifics.SearchForTag('layers'))
    meta.stateful = bool(int(computer_specifics.SearchForTag('stateful')))
    meta.no_outputs = 3

    if meta.layers == 2:
        number_of_lstm_neurons = f'{meta.neurons_layer_one}|{meta.neurons_layer_two}'
    elif meta.layers == 3:
        number_of_lstm_neurons = f'{meta.neurons_layer_one}|{meta.neurons_layer_two}|{meta.neurons_layer_three}'
    else:
        number_of_lstm_neurons = str(meta.neurons_layer_one)

    pump_nos = get_pump_numbers(start, end, pump_no_tag)

    for timestamp, measurement in zip(pump_nos.timestamp, pump_nos.measurements):
        meta.failed = False
        timestamp = timestamp.replace(tzinfo=None)
        get_previous_run_data(meta, model_name)
        meta.checkpoint_path = f'{checkpoint_path}{str(timestamp)[:10]}_{pump_nos.tag}_{str(measurement)}'
        scaler_table = find_scaler_table(table)
        skip = False
        if not testing:
            for path, time_ in zip(meta.previous_checkpoint_path_list, meta.previous_data_training_start):
                if path == meta.checkpoint_path and time_ == timestamp:
                    print(path, time_)
                    skip = True
                    break
        skip = skip_type_of_pumps(meta, skip, skip_type_1_pump, skip_type_2_pump)
        if not skip:
            check_if_scaler_exists(meta, pump_no_tag, scaler_table, timestamp)

            if not meta.failed:
                if use_csv_file:
                    try:
                        df = pd.read_csv(
                            f'./pump={pump_no_tag}={measurement} time={str(timestamp)[:10]} raw={meta.raw} '
                            f'agg_time={meta.aggregation_time} avg={meta.average} '
                            f'std_dev={meta.standard_deviation} {table}.csv', index_col=0
                        )
                    except FileNotFoundError as e:
                        print(e)
                        df = get_data(end, meta, pump_no_tag, table, timestamp)
                        try:
                            df.fillna(method='backfill', inplace=True)
                            df.fillna(method='ffill', inplace=True)
                            df.to_csv(f'./pump={pump_no_tag}={measurement} time={str(timestamp)[:10]} raw={meta.raw} '
                                      f'agg_time={meta.aggregation_time} avg={meta.average} '
                                      f'std_dev={meta.standard_deviation} {table}.csv')
                        except AttributeError as e:
                            print(e)

                else:
                    df = get_data(end, meta, pump_no_tag, table, timestamp)

            if not meta.failed:
                if not testing:
                    insert_new_metadata_record_LSTM(meta, number_of_lstm_neurons, timestamp)
                df = rename_and_scale(df, meta, pump_no_tag, scaler_table, timestamp)
                # Calcualte defined equations in meta.derivative_pairs
                if meta.raw and meta.derivatives:
                    calculate_combinations(df, meta)
                meta.features = get_features(df)
                pprint(df.describe())
                if not exists(f'./line_plots/Lineplot {table} {tag} {str(timestamp)[:10]} raw={meta.raw} avg={meta.average} '
                           f'std={meta.standard_deviation} agg_time={meta.aggregation_time}.html'):
                    plots.plot(f'./line_plots/Lineplot {table} {tag} {str(timestamp)[:10]} raw={meta.raw} avg={meta.average} '
                               f'std={meta.standard_deviation} agg_time={meta.aggregation_time}',meta.features,
                               df, single_time=True)
                #fig = px.line(df, x='Time', y=[meta.features])
                #fig.show()


                # Split data, creates sequences of data for X and y and one hot encodes y
                X_train, X_validate, Y_train, Y_validate, X_test, Y_test = pre_process_LSTM(df, meta)
                meta.no_of_features = X_train.shape[2]

            if not meta.failed:
                #define model
                if not testing:
                    model = define_and_train_model(X_train, X_validate, Y_train, Y_validate, meta)
                    update_metadata_record_LSTM(meta, model_name)
                else:
                    model = tf.keras.models.load_model(f'./LSTM_models/{meta.model_name}/'
                                                       f'{meta.checkpoint_path[len("./checkpoints/"):]}')
                    accuracies = test_model(model, X_test, Y_test, meta=meta)
                    update_metadata_test_set(meta)

                #print(meta.test_accuracy)

                #save metadata


            if meta.failed:
                #Delete current row in meta as it will not be used
                delete_metadata_record_LSTM(meta, model_name)


def calculate_combinations(df, meta):
    if meta.derivatives:
        for calc in meta.derivative_pairs:
            calculation = Calculation(calc.get("arg1"), calc.get("arg2"), calc.get("operator"))
            calculation.calculate(df)
    print(df.head())


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
        date_start = '2020-03-01'
        date_stop = '2022-02-01'

    pump_no_tag = 'HYG_PU19_Pump_no'
    checkpoint_path = './checkpoints/'
    file = XML('./Settings/table_config.xml')
    table = file.SearchForTag('table')
    high_table = file.SearchForTag('high_table')
    pump_state_list = ["HYG_PU12_Pump_no",
                       "HYG_PU16_Pump_no",
                       "HYG_PU15_Pump_no",
                       #"HYG_PU14_Pump_no",
                       "HYG_PU13_Pump_no",
                       "HYG_PU19_Pump_no",
                       #"HYG_PU18_Pump_no",
                       "HYG_PU17_Pump_no"
                       ]

    settings_file = ['LSTM model_settings exp 1.xml', 'LSTM model_settings exp 2.xml', 'LSTM model_settings exp 3.xml',
                     'LSTM model_settings exp 4.xml', 'LSTM model_settings exp 5.xml', 'LSTM model_settings exp 6.xml',
                     'LSTM model_settings exp 7.xml', 'LSTM model_settings exp 8.xml', 'LSTM model_settings exp 9.xml',
                     'GRU model_settings exp 1.xml', 'GRU model_settings exp 2.xml', 'GRU model_settings exp 3.xml',
                     'GRU model_settings exp 4.xml', 'GRU model_settings exp 5.xml', 'GRU model_settings exp 6.xml',
                     'GRU model_settings exp 7.xml', 'GRU model_settings exp 8.xml', 'GRU model_settings exp 9.xml'
                     ]
    settings_file = [
                      'LSTM model_settings exp 3.xml'
                     

                     ]
    test = False
    for file in settings_file:
        print(file)
        for tag in pump_state_list:
            print(f'This tag now {tag}')
            main_training_pipeline(date_start, date_stop, tag, checkpoint_path, table, file, testing=test,
                                   use_csv_file=True)
