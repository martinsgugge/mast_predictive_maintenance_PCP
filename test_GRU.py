import datetime
import os
import pickle
import numpy as np
from SVM import train_svm, test_svm
from XML import XML
from database_high_level import get_process_data_for_physical_pump, get_pump_numbers, update_metadata_record_LSTM, \
    delete_metadata_record_LSTM, insert_new_metadata_record_LSTM, get_previous_run_data, query_highest_nearest_state, \
    update_metadata_record_SVM, insert_new_metadata_record_SVM, delete_metadata_record_SVM, \
    update_metadata_full_test_set
from standard_scalers import read_standard_scaler, check_if_scaler_exists
import pandas as pd
from array_manipulation import get_generic_column_names, search_string_like_in_list, get_features, merge_df, convert_to_date_or_datetime
#from LSTM import *
import tensorflow as tf
from model_batch_training_GRU import calculate_combinations
from GRU import test_model
from metadata import MetadataLSTM, MetadataSVM
from postgres import psql
from pre_process import pre_process_LSTM, rename_and_scale, pre_process_LSTM, rename_and_scale_test_set, \
    pre_process_LSTM_test
from plots import *
from PCA_PU19 import pca
from get_data import get_data
from os.path import exists

def get_data_pre_process(start, end, pump_no_tag, checkpoint_path, table, file, avg=False, std=False,
                         use_csv_file=False):

    # Metadata
    computer_specifics = XML(f'./Settings/{file}')
    model_name = computer_specifics.SearchForTag('ModelName')
    meta = MetadataLSTM(model_name)

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

    # Model parameters
    meta.number_of_epochs = int(computer_specifics.SearchForTag('number_of_epochs'))
    meta.batch_size = int(computer_specifics.SearchForTag('batch_size'))
    meta.neurons_layer_one = int(computer_specifics.SearchForTag('neurons_layer_one'))  # LSTM blocks
    meta.neurons_layer_two = int(computer_specifics.SearchForTag('neurons_layer_two'))  # LSTM blocks
    meta.layers = int(computer_specifics.SearchForTag('layers'))
    meta.no_outputs = 3

    if meta.layers == 2:
        number_of_lstm_neurons = f'{meta.neurons_layer_one}|{meta.neurons_layer_two}'
    else:
        number_of_lstm_neurons = str(meta.neurons_layer_one)

    pump_nos = get_pump_numbers(start, end, pump_no_tag)
    train_x = []
    train_y = []
    validate_x = []
    validate_y = []
    test_x = []
    test_y = []

    for timestamp, measurement in zip(pump_nos.timestamp, pump_nos.measurements):
        meta.failed = False

        timestamp = timestamp.replace(tzinfo=None)
        print(timestamp)
        get_previous_run_data(meta, model_name)
        meta.checkpoint_path = f'{checkpoint_path}{str(timestamp)[:10]}_{pump_nos.tag}_{str(measurement)}'
        scaler_table = find_scaler_table(table)
        skip = False
        for path, time_ in zip(meta.previous_checkpoint_path_list, meta.previous_data_training_start):
            if path == meta.checkpoint_path and time_ == timestamp:
                print(path, time_)
                skip = True
                break

        if not skip:

            # check_if_scaler_exists(meta, pump_no_tag, scaler_table, timestamp)

            if not meta.failed:
                print(meta.checkpoint_path)

                if use_csv_file:
                    try:
                        df = pd.read_csv(f'./pump={pump_no_tag}={measurement} time={str(timestamp)[:10]} raw={meta.raw} '
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
                # print(f'inserting meta: meta : {meta.failed}')
                #
                # print(f'first pre_process: meta : {meta.failed}')
                df = rename_and_scale_test_set(df, meta, pump_no_tag, scaler_table, timestamp)
                # print(df.describe())
                # print(f'second pre_process: meta : {meta.failed}')
                # print(df.describe())
                if meta.raw and meta.derivatives:
                    calculate_combinations(df, meta)
                X_test, Y_test = pre_process_LSTM_test(df, meta)

                test_x.append(X_test)
                test_y.append(Y_test)
                # print(f'second pre_process finished: meta : {meta.failed}')

    #print(meta.failed, pump_nos.timestamp)
    # return np.array(train_x), np.array(train_y), np.array(validate_x), np.array(validate_y), \
    #        np.array(test_x), np.array(test_y), meta
    return train_x, train_y, validate_x, validate_y, \
           test_x, test_y, meta

    # else:
    #     #Delete current row in meta as it will not be used
    #     delete_metadata_record_SVM(meta, model_name)




def restructure_data(meta, test_x, test_y):
    print(test_x[0].shape)
    meta.no_of_features = test_x[0].shape[2]
    train_len = []
    val_len = []
    test_len = []
    a, b, c, d, e, f = 0, 0, 0, 0, 0, 0
    for i in range(len(test_x)):
        print(test_x[i].shape[0])

        a = test_x[i].shape[0]

        d += a

        test_len.append(a)
        print(d)

    X_test = np.zeros(shape=(d, meta.sequence_steps,meta.no_of_features))
    Y_test = np.zeros(shape=(d, meta.sequence_steps, 3))
    print(f'meta number of features {meta.no_of_features}')
    print('shapes arrays')
    print(X_test.shape)
    print(Y_test.shape)
    test_index = 0
    for i in range(len(test_len)):
        # for x in test_x[i]:
        #     print(x)
        print(test_x[i].shape)
        print(test_y[i].shape)
        X_test[test_index:test_index + test_len[i], :] = test_x[i]
        Y_test[test_index:test_index + test_len[i]] = test_y[i]


        test_index += test_len[i]
    return X_test, Y_test


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


def test_GRU(date_start, date_stop, pump_state_list, checkpoint_path, table, file, use_csv_file=False):

    test_x = []
    test_y = []
    for tag in pump_state_list:

        X_train, Y_train, X_validate, Y_validate, X_test, Y_test, meta = \
            get_data_pre_process(date_start, date_stop, tag, checkpoint_path, table, file, use_csv_file=use_csv_file)

        try:
            if len(X_test) > 0:
                print(X_test[0])
                test_x += X_test
                test_y += Y_test
        except IndexError:
            pass
    # print(type(test_x))
    # print((test_x[0].shape))
    # print(type(test_x[1].shape[1]))
    X_test, Y_test = restructure_data(meta, test_x, test_y)
    print('Entering training area')
    print(meta.previous_checkpoint_path)
    model = tf.keras.models.load_model(f'./GRU_models/{meta.model_name}/{meta.previous_checkpoint_path[14:]}')
    test_model(model, X_test, Y_test, meta)
    update_metadata_full_test_set(meta)


if __name__ == '__main__':
    laptop = False
    if laptop:
        date_start = '2021-09-01'
        date_stop = '2021-10-14'
    else:
        date_start = '2020-03-01'
        date_stop = '2022-05-01'

    pump_no_tag = 'HYG_PU19_Pump_no'
    checkpoint_path = './checkpoints/'
    file = XML('./Settings/table_config.xml')
    table = file.SearchForTag('table')
    high_table = file.SearchForTag('high_table')
    pump_state_list = [#"HYG_PU12_Pump_no",
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
    settings_file = [#'GRU model_settings exp 1.xml', 'GRU model_settings exp 2.xml', 'GRU model_settings exp 3.xml',
                     #'GRU model_settings exp 4.xml', 'GRU model_settings exp 5.xml',
                     #'GRU model_settings exp 6.xml',
                     #'GRU model_settings exp 7.xml',#need debug
        'GRU model_settings exp 11.xml',
        #'GRU model_settings exp 9.xml',
                     #'GRU model_settings exp 10.xml',
        #'GRU model_settings exp 11.xml',#need debug
        #'GRU model_settings exp 12.xml', 'GRU model_settings exp 13.xml', 'GRU model_settings exp 14.xml',
        #'GRU model_settings exp 15.xml'
                     #'GRU model_settings exp 4.xml', 'GRU model_settings exp 5.xml',
                     ]
    for file in settings_file:
        print(file)

        test_GRU(date_start, date_stop, pump_state_list, checkpoint_path, table, file, use_csv_file=True)

