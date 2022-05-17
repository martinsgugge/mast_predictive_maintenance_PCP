import pandas as pd

import metadata
from database_high_level import query_highest_nearest_state, get_process_data_for_physical_pump, query_next_states
from array_manipulation import merge_df, get_features, get_generic_column_names, convert_tag_to_df

import datetime


def get_data(end, meta, pump_no_tag, table, timestamp):
    # Get process measurements
    print(f'raw: {meta.raw} | avg: {meta.average} | std: {meta.standard_deviation}')
    print(f'Use all?: {meta.use_all}')
    if meta.use_all:
        # local_end = query_highest_nearest_state(timestamp, end, pump_no_tag, meta)
        state = query_next_states(timestamp, end, pump_no_tag, meta)
        try:
            local_end = state.timestamp[-1]
        except IndexError as e:
            print(e)
        print(state.timestamp)

        if not meta.failed:
            meta.data_training_stop = local_end.replace(tzinfo=None) + datetime.timedelta(days=1)
            meta.data_training_start = timestamp
        if meta.raw:
            df = get_process_data_for_physical_pump(pump_no_tag, meta, table=table)

        if meta.average:
            df_avg = get_process_data_for_physical_pump(pump_no_tag, meta, table=table,
                                                        agg_time=22, avg=True)
            if not meta.raw:
                df = df_avg
        if meta.standard_deviation:
            df_std = get_process_data_for_physical_pump(pump_no_tag, meta, table=table,
                                                        agg_time=22, std_dev=True)
        if not meta.failed:
            if meta.average and meta.raw:
                df = merge_df(df, df_avg, method='nearest')

            if meta.standard_deviation:
                df = merge_df(df, df_std, method='nearest')

    else:
        print('Not using all data')
        state = query_next_states(timestamp, end, pump_no_tag, meta)
        print('Local starts: ', state.timestamp)
        print(state.measurements)

        new_starts, new_stops = get_subset_of_data(state.timestamp, meta)
        if not meta.failed:
            dfs = []
            for start, stop in zip(new_starts, new_stops):
                print(start, stop)
                if not meta.failed:
                    meta.data_training_stop = stop
                    meta.data_training_start = start
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
                    dfs.append(df)
                checked_dfs = []

                if dfs != []:
                    meta.failed = False

            for x in dfs:
                if x is not None:
                    checked_dfs.append(x)

            try:
                df = pd.concat(checked_dfs)
                # Format data for data frame

            except IndexError as e:
                print(e)
                print('Not complete failure')
                df = None
                meta.failed = True

            except UnboundLocalError as e:
                print('checked_dfs does not exist')

    if not meta.failed:
        state_df = convert_tag_to_df(state)
        print(state_df.head())
        try:
            df = merge_df(df, state_df)
        except AttributeError as e:
            print(e)
            meta.failed = True
        except UnboundLocalError as e:
            print('checked_dfs does not exist')
    if meta.failed:
        df = None

    return df


def get_subset_of_data(local_starts, meta):
    # one day per week from state 0 to 1

    try:
        number_of_days = (local_starts[1] - local_starts[0]).days
    except IndexError:
        print('not enough states in time frame')
        meta.failed = True
    try:
        if not meta.failed:
            number_of_weeks = (number_of_days // 7)
            new_starts = [local_starts[0]]
            new_stops = [new_starts[0] + datetime.timedelta(days=1)]
            for i in range(number_of_weeks):
                new_starts.append(new_starts[i] + datetime.timedelta(days=7))
                new_stops.append(new_stops[i] + datetime.timedelta(days=7))

            new_starts.append(local_starts[1])
            new_stops.append(local_starts[1] + datetime.timedelta(days=6))

            new_starts.append(local_starts[2])
            new_stops.append(local_starts[2] + datetime.timedelta(days=1))

        else:
            new_starts = []
            new_stops = []
    except IndexError as e:
        meta.failed = True
        new_starts = []
        new_stops = []

    return new_starts, new_stops

def test_get_subset_of_data():

    meta = metadata.MetadataLSTM('Test')
    timestamp = datetime.datetime.strptime('2020-03-04 00:21:00', '%Y-%m-%d %H:%M:%S')
    end = datetime.datetime.strptime('2021-03-04 12:21:00', '%Y-%m-%d %H:%M:%S')
    pump_no_tag = 'HYG_PU13_Pump_no'

    local_starts = query_next_states(timestamp, end, pump_no_tag, meta)
    new_starts, new_stops = get_subset_of_data(local_starts)
    #for x,y in zip(new_starts, new_stops):
    #    print(x,y)

def get_data_viz(avg, end, meta, pump_no_tag, std, table, timestamp):
    # Get process measurements


    df = get_process_data_for_physical_pump(pump_no_tag, meta, table=table)

    if avg:
        df_avg = get_process_data_for_physical_pump(pump_no_tag, meta, table=table,
                                                    agg_time=22, avg=True)
    if std:
        df_std = get_process_data_for_physical_pump(pump_no_tag, meta, table=table,
                                                    agg_time=22, std_dev=True)
    if not meta.failed:
        if avg:
            df = merge_df(df, df_avg, method='nearest')

        if std:
            df = merge_df(df, df_std, method='nearest')

    return df
