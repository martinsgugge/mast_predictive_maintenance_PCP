import pandas as pd
from sklearn.preprocessing import StandardScaler

from array_manipulation import split_sequences, one_hot_encode, get_generic_column_names, split_sequencesv2, \
    get_features, remove_quantiles
from sklearn.model_selection import train_test_split
from keras_preprocessing import sequence
from database_high_level import store_test_set
import numpy as np
# import tensorflow.keras.preprocessing.sequence.TimeseriesGenerator
# import keras.preprocessing.sequence
from standard_scalers import read_standard_scaler
from array_manipulation import remove_on_off


def pre_process_LSTM(df, meta):

    # Rename
    sensors = list(df.columns[1:])
    tmp = df.loc[:, sensors]
    generic_tags = ['On_off', 'OutletPressure', 'Current', 'Torque', 'ControlSignal', 'Speed']
    rename_dict = get_generic_column_names(tmp, generic_tags)
    df[sensors] = tmp.rename(columns=rename_dict)
    df.fillna(method='backfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    # Use sequence if LSTM
    print(f'pre-process columns{df.columns}')
    if meta.sequence:
        print('Opening split sequence')
        X_, Y_ = split_sequencesv2(df, meta.sequence_steps, meta.pad)

        X_train, X_validation, Y_train, Y_validation = train_test_split(X_, Y_,
                                                                        test_size=meta.validation_size + meta.test_size,
                                                                        random_state=42)
        X_validation, X_test, Y_validation, Y_test = train_test_split(X_validation, Y_validation,
                                                                      test_size=(meta.test_size/meta.validation_size)/2,
                                                                      random_state=42)

        Y_out_train = []
        Y_out_validation = []
        Y_out_test = []
        if not meta.pad:
            for x in Y_train:
                Y_out_train.append(x[0])

            for x in Y_validation:
                Y_out_validation.append(x[0])

            for x in Y_test:
                Y_out_test.append(x[0])

            Y_out_train = one_hot_encode(Y_out_train, 3)
            Y_out_validation = one_hot_encode(Y_out_validation, 3)
            Y_out_test = one_hot_encode(Y_out_test, 3)

        else:
            for x in Y_train:
                Y_out_train.append(one_hot_encode(x))
            for x in Y_validation:
                Y_out_validation.append(one_hot_encode(x))
            for x in Y_test:
                Y_out_test.append(one_hot_encode(x))


        #Y_ = one_hot_encode(Y_, 3)
    else:
        #X_ = df.loc[:, df.columns != 'Time']
        X_ = df.loc[:, df.columns != 'State']
        Y_ = df['State']

        X_train, X_validation, Y_train, Y_validation = train_test_split(X_, Y_,
                                                                        test_size=meta.validation_size + meta.test_size,
                                                                        random_state=42)
        X_validation, X_test, Y_validation, Y_test = train_test_split(X_validation, Y_validation,
                                                                      test_size=(meta.test_size / meta.validation_size) / 2,
                                                                      random_state=42)
        Y_out_train = one_hot_encode(Y_train, 3)
        Y_out_validation = one_hot_encode(Y_validation, 3)
        Y_out_test = one_hot_encode(Y_test, 3)


    #Save test set for later
    generic_tags.append('State')
    #store_test_set(X_test, Y_test, generic_tags)

    return X_train, X_validation, np.array(Y_out_train), np.array(Y_out_validation), X_test, np.array(Y_out_test)

def pre_process_LSTM_test(df, meta):

    # Rename
    sensors = list(df.columns[1:])
    tmp = df.loc[:, sensors]
    generic_tags = ['On_off', 'OutletPressure', 'Current', 'Torque', 'ControlSignal', 'Speed']
    rename_dict = get_generic_column_names(tmp, generic_tags)
    df[sensors] = tmp.rename(columns=rename_dict)
    df.fillna(method='backfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    # Use sequence if LSTM
    print(f'pre-process columns{df.columns}')
    if meta.sequence:
        print('Opening split sequence')
        X_, Y_ = split_sequencesv2(df, meta.sequence_steps, meta.pad)

        X_train, X_validation, Y_train, Y_validation = train_test_split(X_, Y_,
                                                                        test_size=0.65,
                                                                        random_state=42)
        X_validation, X_test, Y_validation, Y_test = train_test_split(X_validation, Y_validation,
                                                                      test_size=(meta.test_size/meta.validation_size)/2,
                                                                      random_state=42)

        Y_out_train = []
        Y_out_validation = []
        Y_out_test = []
        if not meta.pad:
            for x in Y_train:
                Y_out_train.append(x[0])

            Y_out_train = one_hot_encode(Y_out_train, 3)

        else:
            for x in Y_train:
                Y_out_train.append(one_hot_encode(x))

        #Y_ = one_hot_encode(Y_, 3)
    else:
        #X_ = df.loc[:, df.columns != 'Time']
        X_ = df.loc[:, df.columns != 'State']
        Y_ = df['State']

        X_train, X_validation, Y_train, Y_validation = train_test_split(X_, Y_,
                                                                        test_size=0.65,
                                                                        random_state=42)

        Y_out_train = one_hot_encode(Y_train, 3)



    #Save test set for later
    generic_tags.append('State')
    #store_test_set(X_test, Y_test, generic_tags)

    return X_train, np.array(Y_out_train)


def pre_process_SVM(df, meta):

    df.fillna(method='backfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.drop(columns=['Time'], inplace=True)

    # Use sequence if LSTM
    print(f'pre-process columns{df.columns}')

    train, validation = train_test_split(df, test_size=meta.validation_size + meta.test_size, random_state=42)
    validation, test = train_test_split(validation, test_size=(meta.test_size / meta.validation_size) / 2,
                                        random_state=42)
    print(train.describe())
    temp_set = [train, validation, test]
    X_temp_set = []
    Y_temp_set = []
    for i in range(len(temp_set)):
        if meta.off_removal:
            try:
                temp_set[i] = remove_on_off(temp_set[i])
                print('Removing off')
                print(temp_set[i].describe())
            except KeyError as e:
                print(e)
        if meta.top_removal:
            temp_set[i] = remove_quantiles(temp_set[i], meta.top_removal_limit, 0.0, meta.top_removal_list)
            print('Removing top values')
            print(temp_set[i].describe())
        if meta.bottom_removal:
            temp_set[i] = remove_quantiles(temp_set[i], 1, meta.bottom_removal_limit, meta.bottom_removal_list)
            print('Removing bottom values')
            print(temp_set[i].describe())

        X_temp_set.append(temp_set[i].loc[:, temp_set[i].columns != 'State'])
        Y_temp_set.append(temp_set[i]['State'])

    for x in X_temp_set:
        print(x.columns)
        print(x.describe())

    Y_train = Y_temp_set[0]
    Y_validation = Y_temp_set[1]
    Y_test = Y_temp_set[2]

    # Y_out_train = []
    # Y_out_validation = []
    # Y_out_test = []
    #
    # for x in Y_train:
    #     Y_out_train.append(x)
    #
    # for x in Y_validation:
    #     Y_out_validation.append(x)
    #
    # for x in Y_test:
    #     Y_out_test.append(x)
    #
    # Y_out_train = one_hot_encode(Y_out_train, 3)
    # Y_out_validation = one_hot_encode(Y_out_validation, 3)
    # Y_out_test = one_hot_encode(Y_out_test, 3)

    X_train = X_temp_set[0]
    X_validation = X_temp_set[1]
    X_test = X_temp_set[2]


    return np.array(X_train), np.array(X_validation), np.array(Y_train), np.array(Y_validation), \
           np.array(X_test), np.array(Y_test)

def pre_process_SVM_test(df, meta):

    df.fillna(method='backfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.drop(columns=['Time'], inplace=True)

    # Use sequence if LSTM
    print(f'pre-process columns{df.columns}')

    train, validation = train_test_split(df, test_size=meta.validation_size + meta.test_size, random_state=42)
    validation, test = train_test_split(validation, test_size=(meta.test_size / meta.validation_size) / 2,
                                        random_state=42)
    print(train.describe())
    temp_set = [train, validation, test]
    X_temp_set = []
    Y_temp_set = []
    for i in range(len(temp_set)):
        if meta.off_removal:
            try:
                temp_set[i] = remove_on_off(temp_set[i])
                print('Removing off')
                print(temp_set[i].describe())
            except KeyError as e:
                print(e)
        if meta.top_removal:
            temp_set[i] = remove_quantiles(temp_set[i], meta.top_removal_limit, 0.0, meta.top_removal_list)
            print('Removing top values')
            print(temp_set[i].describe())
        if meta.bottom_removal:
            temp_set[i] = remove_quantiles(temp_set[i], 1, meta.bottom_removal_limit, meta.bottom_removal_list)
            print('Removing bottom values')
            print(temp_set[i].describe())

        X_temp_set.append(temp_set[i].loc[:, temp_set[i].columns != 'State'])
        Y_temp_set.append(temp_set[i]['State'])

    for x in X_temp_set:
        print(x.columns)
        print(x.describe())

    Y_train = Y_temp_set[0]
    Y_validation = Y_temp_set[1]
    Y_test = Y_temp_set[2]

    # Y_out_train = []
    # Y_out_validation = []
    # Y_out_test = []
    #
    # for x in Y_train:
    #     Y_out_train.append(x)
    #
    # for x in Y_validation:
    #     Y_out_validation.append(x)
    #
    # for x in Y_test:
    #     Y_out_test.append(x)
    #
    # Y_out_train = one_hot_encode(Y_out_train, 3)
    # Y_out_validation = one_hot_encode(Y_out_validation, 3)
    # Y_out_test = one_hot_encode(Y_out_test, 3)

    X_train = X_temp_set[0]
    X_validation = X_temp_set[1]
    X_test = X_temp_set[2]


    return np.array(X_train), np.array(Y_train)


def rename_and_scale(df, meta, pump_no_tag, scaler_table, timestamp):
    meta.features = get_features(df)

    print(f'THESE ARE THE FEATURES: {meta.features}')
    # Convert names to standard names
    generic_tags = ['On_off', 'OutletPressure', 'Current', 'Torque', 'ControlSignal', 'Speed', 'State',
                    'OutletPressureAvg', 'CurrentAvg', 'TorqueAvg', 'ControlSignalAvg', 'SpeedAvg',
                    'OutletPressureStd', 'CurrentStd', 'TorqueStd', 'ControlSignalStd', 'SpeedStd']
    rename_dict = get_generic_column_names(df, generic_tags)
    df = df.rename(columns=rename_dict)
    df.head()
    # Scale data frames using created scalers
    scaler = read_standard_scaler(scaler_table, pump_no_tag[:8], str(timestamp)[:10], meta)
    scaler_features = list(df.loc[:, df.columns != 'Time'])
    try:
        scaler_features.remove('On_off')
    except ValueError as e:
        print(e)
    try:
        scaler_features.remove('State')
    except ValueError as e:
        print(e)

    df[scaler_features] = scaler.transform(df[scaler_features])

    return df


def rename_and_scale_test_set(df, meta, pump_no_tag, scaler_table, timestamp):
    meta.features = get_features(df)

    print(f'THESE ARE THE FEATURES: {meta.features}')
    # Convert names to standard names
    generic_tags = ['On_off', 'OutletPressure', 'Current', 'Torque', 'ControlSignal', 'Speed', 'State',
                    'OutletPressureAvg', 'CurrentAvg', 'TorqueAvg', 'ControlSignalAvg', 'SpeedAvg',
                    'OutletPressureStd', 'CurrentStd', 'TorqueStd', 'ControlSignalStd', 'SpeedStd']
    rename_dict = get_generic_column_names(df, generic_tags)
    df = df.rename(columns=rename_dict)
    df.head()
    # Scale data frames using created scalers

    # scaler = read_standard_scaler(scaler_table, pump_no_tag[:8], str(timestamp)[:10], meta)
    scaler_features = list(df.loc[:, df.columns != 'Time'])
    try:
        scaler_features.remove('On_off')
    except ValueError as e:
        print(e)
    try:
        scaler_features.remove('State')
    except ValueError as e:
        print(e)
    scaler = StandardScaler()
    scaler.fit(df[scaler_features])
    df[scaler_features] = scaler.transform(df[scaler_features])
    return df

def skip_type_of_pumps(meta, skip, skip_type_1_pump, skip_type_2_pump):
    meta.pump_type_2 = ['./checkpoints/2021-11-25_HYG_PU17_Pump_no_7.0',
                        './checkpoints/2021-09-07_HYG_PU17_Pump_no_6.0',
                        './checkpoints/2022-02-08_HYG_PU19_Pump_no_11.0',
                        './checkpoints/2021-11-22_HYG_PU19_Pump_no_10.0',
                        './checkpoints/2020-03-04_HYG_PU19_Pump_no_0.0'
                        ]
    if skip_type_1_pump:
        skip = True
        for path in meta.pump_type_2:
            if path == meta.checkpoint_path:
                print(f'{path} != {meta.checkpoint_path}')
                skip = False
                # break
    elif skip_type_2_pump:
        for path in meta.pump_type_2:
            if path == meta.checkpoint_path:
                print(f'{path} == {meta.checkpoint_path}')
                skip = True
                break
    if meta.checkpoint_path == './checkpoints/2021-09-14_HYG_PU19_Pump_no_8.0':
        skip = True
    return skip

if __name__ == '__main__':
    pass