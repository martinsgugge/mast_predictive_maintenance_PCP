import datetime
import os
import pickle

from SVM import train_svm, test_svm
from XML import XML
from database_high_level import get_process_data_for_physical_pump, get_pump_numbers, update_metadata_record_LSTM, \
    delete_metadata_record_LSTM, insert_new_metadata_record_LSTM, get_previous_run_data, query_highest_nearest_state, \
    update_metadata_record_SVM, insert_new_metadata_record_SVM, delete_metadata_record_SVM, \
    update_metadata_record_SVM_test, get_previous_run_data_SVM, get_previous_run_data_SVM_no_meta
from standard_scalers import read_standard_scaler, check_if_scaler_exists
import pandas as pd
from array_manipulation import get_generic_column_names, search_string_like_in_list, get_features, merge_df, convert_to_date_or_datetime
from LSTM import *
from metadata import MetadataLSTM, MetadataSVM
from postgres import psql
from pre_process import pre_process_LSTM, rename_and_scale, pre_process_SVM, rename_and_scale_test_set, \
    pre_process_SVM_test
from plots import *
from PCA_PU19 import pca
from get_data import get_data
from os.path import exists

def get_data_pre_process(start, end, pump_no_tag, checkpoint_path, table, file, avg=False, std=False, use_csv_file=False):

    # Metadata
    computer_specifics = XML(f'./Settings/{file}')
    model_name = computer_specifics.SearchForTag('ModelName')
    meta = MetadataSVM(model_name)

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
    meta.off_removal = bool(int(computer_specifics.SearchForTag('off_removal')))
    meta.top_removal = bool(int(computer_specifics.SearchForTag('top_removal')))
    meta.bottom_removal = bool(int(computer_specifics.SearchForTag('bottom_removal')))

    meta.top_removal_limit = float(computer_specifics.SearchForTag('top_removal_limit'))
    meta.bottom_removal_limit = float(computer_specifics.SearchForTag('bottom_removal_limit'))

    meta.top_removal_list = computer_specifics.SearchForTag('top_removal_list').split(',')
    meta.bottom_removal_list = computer_specifics.SearchForTag('bottom_removal_list').split(',')


    meta.train_size = float(computer_specifics.SearchForTag('train_size'))
    meta.test_size = float(computer_specifics.SearchForTag('test_size'))
    meta.validation_size = float(computer_specifics.SearchForTag('validation_size'))
    print(meta.top_removal, meta.top_removal_limit, meta.bottom_removal_limit)
    #Model parameters
    try:
        meta.gamma = float(computer_specifics.SearchForTag('gamma'))
    except ValueError:
        meta.gamma = str(computer_specifics.SearchForTag('gamma'))
    meta.c = float(computer_specifics.SearchForTag('C'))
    meta.coef0 = float(computer_specifics.SearchForTag('coef0'))
    meta.method = computer_specifics.SearchForTag('method')
    meta.degree = int(computer_specifics.SearchForTag('degree'))
    meta.no_outputs = 3

    pump_nos = get_pump_numbers(start, end, pump_no_tag)
    train_x = []
    train_y = []
    validate_x = []
    validate_y = []
    test_x = []
    test_y = []

    #Used to skip pumps that has been trained on
    computer_specifics = XML(f'./Settings/GRU model_settings exp 3.xml')
    model_name = computer_specifics.SearchForTag('ModelName')
    meta2 = Metadata(model_name)
    get_previous_run_data(meta2, meta2.model_name)
    for timestamp, measurement in zip(pump_nos.timestamp, pump_nos.measurements):
        meta.failed = False

        timestamp = timestamp.replace(tzinfo=None)
        print(timestamp)
        get_previous_run_data(meta, model_name)
        meta.checkpoint_path = f'{checkpoint_path}{str(timestamp)[:10]}_{pump_nos.tag}_{str(measurement)}'
        scaler_table = find_scaler_table(table)
        skip = False

        for path, time_ in zip(meta2.previous_checkpoint_path_list, meta2.previous_data_training_start):
            if path == meta.checkpoint_path and time_ == timestamp:
                print(path, time_)
                skip = True
                break

        if not skip:

            #check_if_scaler_exists(meta, pump_no_tag, scaler_table, timestamp)

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
                # print(f'inserting meta: meta : {meta.failed}')
                #
                # print(f'first pre_process: meta : {meta.failed}')
                df = rename_and_scale_test_set(df, meta, pump_no_tag, scaler_table, timestamp)
                # print(df.describe())
                # print(f'second pre_process: meta : {meta.failed}')
                # print(df.describe())
                X_test, Y_test = pre_process_SVM_test(df, meta)
                print(f'preprocessed x: {X_test.shape} y: {Y_test.shape}')

                test_x.append(X_test)
                test_y.append(Y_test)
                # print(f'second pre_process finished: meta : {meta.failed}')

    print(meta.failed, pump_nos.timestamp)
    # return np.array(train_x), np.array(train_y), np.array(validate_x), np.array(validate_y), \
    #        np.array(test_x), np.array(test_y), meta
    return test_x, test_y, meta

    # else:
    #     #Delete current row in meta as it will not be used
    #     delete_metadata_record_SVM(meta, model_name)


def main_training_pipeline(start, end, pump_no_tag, checkpoint_path, table, file, train_run, use_csv_file=False):

    test_x = []
    test_y = []
    for tag in pump_state_list:
        X_test, Y_test, meta = \
            get_data_pre_process(start, end, tag, checkpoint_path, table, file, use_csv_file=use_csv_file)

        try:
            test_x += X_test
            test_y += Y_test
        except IndexError:
            pass


    X_test, Y_test = restructure_data(meta, test_x, test_y)
    print('Entering training area')
    meta.train_run = train_run
    test_svm_model(X_test, Y_test, meta, table)


def restructure_data(meta, test_x, test_y):
    meta.no_of_features = test_x[0].shape[1]

    test_len = []
    a, b, c, d, e, f = 0, 0, 0, 0, 0, 0
    for i in range(len(test_x)):

        print(test_x[i].shape[0])
        c = test_x[i].shape[0]

        f += c

        test_len.append(c)
        print(d, e, f)

    X_test = np.zeros(shape=(f, meta.no_of_features))
    Y_test = np.zeros(shape=(f))

    test_index = 0
    for i in range(len(test_len)):
        print(f'x shape: {test_x[i].shape}')
        print(f'y shape: {test_y[i].shape}')
        X_test[test_index:test_index + test_len[i], :] = test_x[i]
        Y_test[test_index:test_index + test_len[i]] = test_y[i]

        test_index += test_len[i]

    return X_test, Y_test


def test_svm_model(X_test, Y_test, meta, table):

    #insert_new_metadata_record_SVM(meta, meta.training_start)
    # Load model
    #get_previous_run_data(meta, model_name)
    get_previous_run_data_SVM(meta, meta.model_name)
    try:
        with open(f'./SVM_models/SVM train_run={meta.train_run} {table} '
                            f'{meta.model_name} raw={meta.raw} avg={meta.average} '
                            f'std={meta.standard_deviation}.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError as e:
        print('Could not find model')
        print(f'./SVM_models/SVM train_run={meta.train_run} {table} '
                            f'{meta.model_name} raw={meta.raw} avg={meta.average} '
                            f'std={meta.standard_deviation}.pkl')

    print(f'test model: meta.failed : {meta.failed}')
    meta.test_accuracy_hits = test_svm(model, X_test, Y_test, meta)
    # accuracies = test_model(model, X_validate=X_test, y_test=Y_test, meta=meta)
    # print(accuracies)
    extract_matrix_to_string(meta)
    # save metadata
    update_metadata_record_SVM_test(meta)



def extract_matrix_to_string(meta):
    text = ''
    print(meta.test_accuracy_hits.shape)
    for i in range(meta.test_accuracy_hits.shape[0]):
        text += '['
        for j in range(meta.test_accuracy_hits.shape[1]):
            text += str(meta.test_accuracy_hits[i, j]) + ','
        text[:len(text) - 1]
        text += ']'

    meta.test_accuracy_hits = text
    print(meta.test_accuracy_hits)


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


    settings_file = [#'LSTM model_settings exp 1.xml', 'LSTM model_settings exp 2.xml',
                     'LSTM model_settings exp 3.xml',
                     'LSTM model_settings exp 4.xml', 'LSTM model_settings exp 5.xml', 'LSTM model_settings exp 6.xml',
                     #'LSTM model_settings exp 7.xml',
                     'LSTM model_settings exp 8.xml', 'LSTM model_settings exp 9.xml',
                     'GRU model_settings exp 1.xml', 'GRU model_settings exp 2.xml', 'GRU model_settings exp 3.xml',
                     'GRU model_settings exp 4.xml', 'GRU model_settings exp 5.xml', 'GRU model_settings exp 6.xml',
                     'GRU model_settings exp 7.xml', 'GRU model_settings exp 8.xml', 'GRU model_settings exp 9.xml'
                     ]
    settings_file = ['SVM model_settings exp 1.xml'
        #, 'SVM model_settings exp 8.xml'
                     ]

    train_run, model_name = get_previous_run_data_SVM_no_meta()
    settings = zip(train_run, model_name)

    for x in settings:
        print('for x in settings')
        print(x)
        if x[1] == 'SVM exp 1':
            xml_file = 'SVM model_settings exp 1.xml'
        if x[1] == 'SVM exp 2':
            xml_file = 'SVM model_settings exp 2.xml'
        if x[1] == 'SVM exp 3':
            xml_file = 'SVM model_settings exp 3.xml'
        if x[1] == 'SVM exp 4':
            xml_file = 'SVM model_settings exp 4.xml'
        if x[1] == 'SVM exp 5':
            xml_file = 'SVM model_settings exp 5.xml'
        if x[1] == 'SVM exp 6':
            xml_file = 'SVM model_settings exp 6.xml'
        if x[1] == 'SVM exp 7':
            xml_file = 'SVM model_settings exp 7.xml'
        if x[1] == 'SVM exp 8':
            xml_file = 'SVM model_settings exp 8.xml'
        if x[1] == 'SVM exp 9':
            xml_file = 'SVM model_settings exp 9.xml'
        if x[1] == 'SVM exp 10':
            xml_file = 'SVM model_settings exp 10.xml'
        if x[1] == 'SVM exp 11':
            xml_file = 'SVM model_settings exp 11.xml'
        if x[1] == 'SVM exp 12':
            xml_file = 'SVM model_settings exp 12.xml'
        if x[1] == 'SVM exp 13':
            xml_file = 'SVM model_settings exp 13.xml'
        if x[1] == 'SVM exp 14':
            xml_file = 'SVM model_settings exp 14.xml'
        if x[1] == 'SVM exp 15':
            xml_file = 'SVM model_settings exp 15.xml'
        if x[1] == 'SVM exp 16':
            xml_file = 'SVM model_settings exp 16.xml'
        if x[1] == 'SVM exp 17':
            xml_file = 'SVM model_settings exp 17.xml'
        if x[1] == 'SVM exp 18':
            xml_file = 'SVM model_settings exp 18.xml'
        if x[1] == 'SVM exp 19':
            xml_file = 'SVM model_settings exp 19.xml'
        if x[1] == 'SVM exp 20':
            xml_file = 'SVM model_settings exp 20.xml'

        try:
            print(xml_file)
            main_training_pipeline(date_start, date_stop, pump_state_list, checkpoint_path, table, str(xml_file), x[0],
                                   use_csv_file=True)
        except NameError as e:
            print(e)

