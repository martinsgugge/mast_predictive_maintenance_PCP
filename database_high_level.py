import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from metadata import Metadata, MetadataLSTM
from XML import XML
from array_manipulation import split_sequences, one_hot_encode
from main import get_df
from postgres import psql, Tag


def get_process_data(agg_time, avg, std_dev, relevant_tagnames, table, timestamp_from, timestamp_to, ):
    """
    Gets data for relevant_tagnames between timestamp_from to timestamp_to with aggregations
    :param agg_time: How much time should be aggregated over
    :param avg: True if average is used as aggregation
    :param std_dev: True if standard deviation is used as aggregation
    :param relevant_tagnames: tagnames for all measurements that should be included in the returned data frames
    :param table: name of table where the data should be retrieved from
    :param timestamp_from: query start time for process data
    :param timestamp_to: query end time for process data
    :param n_steps: number of timesteps to use in a sequence
    :param sequence: True if data should be split into sequences, often used for LSTM
    :return: training data, test data
    """


    #Get process data for the given tags in either measurement table or high_speed_big
    if table == 'measurement' or table == 'measurement2':
        df = get_df(timestamp_from, timestamp_to, relevant_tagnames, table, min_per_h=None,
                    agg_time=agg_time, avg=avg, std_dev=std_dev)


    elif table == 'high_speed_big_reduced':
        lower_relevant_tagnames = [x.lower().removeprefix('hyg_') for x in relevant_tagnames]

        remove_list = ['pu19', 'pu18', 'pu17', 'pu16', 'pu15', 'pu14', 'pu13', 'pu12']
        for pump in remove_list:
            try:
                lower_relevant_tagnames.remove(pump)
            except ValueError:
                pass
        #lower_relevant_tagnames = lower_relevant_tagnames[1:]
        lower_relevant_tagnames = ['meas_time', *lower_relevant_tagnames]

        df = get_df(timestamp_from, timestamp_to, lower_relevant_tagnames, table, 60,
                    agg_time=agg_time, avg=avg, std_dev=std_dev)

    return df



def get_relevant_pump_tags_general(pump_tag, meta: Metadata):
    """
    Retrieve tagnames for specified pump
    :param pump_tag: name of pump number tag
    :return: list of tagnames
    """
    # Get data
    pump_tags = XML('./Settings/pump_tags.xml')
    #print(pump_tag)
    string_tags = pump_tags.SearchForTag(pump_tag)
    # Transform data
    relevant_tagnames = string_tags.split(',')
    # Get generic tags
    generic_pump_tags = XML('./Settings/generic_tags.xml')
    generic_tags = [generic_pump_tags.SearchForTag(x).split(',') for x in meta.feature_type]
    # print(f'generic tags: {generic_tags}')
    useful_tagnames = []
    for x in generic_tags:
        for name in relevant_tagnames:
            if name in x:
                useful_tagnames.append(name)
    #meta gir generiske navn
    #finnes faktiske navn i generisk navn?
    # print(f'useful tagnames: {useful_tagnames}')

    return useful_tagnames


def query_highest_nearest_state(start, end, tag_no, meta: Metadata):
    """
    Searches for the first highest state in an ordered list
    :param start: timestamp when pump was set in production
    :param end: end of process data search area
    :param tag_no: name of tag for the specific pump
    :return:
    """
    tagname = tag_no[:8] + '_State'
    state = Tag(tagname)
    state.get_measurement(start, end, table='measurement_temp')
    max = 0
    for i in range(len(state.timestamp)):

        if state.measurements[i] <= max:
            if i > 0:
                break
        if state.measurements[i] > max:
            max = state.measurements[i]
            index = i
    try:
        end_timestamp = state.timestamp[index]

    except IndexError as e:
        print(f'No failures in this time epoch for {tagname}')
        meta.failed = True
        end_timestamp = None
    except UnboundLocalError as e:
        print(f'No failures in this time epoch for {tagname}')
        meta.failed = True
        end_timestamp = None

    return end_timestamp


def query_next_states(start, end, tag_no, meta: Metadata):
    """
    Searches for the first highest state in an ordered list
    :param start: timestamp when pump was set in production
    :param end: end of process data search area
    :param tag_no: name of tag for the specific pump
    :return:
    """
    tagname = tag_no[:8] + '_State'
    state = Tag(tagname)
    state.get_measurement(start, end, table='measurement_temp')
    try:
        max = state.measurements[0]
        start_points = [state.timestamp[0]]
        start_values = [state.measurements[0]]
    except IndexError as e:
        meta.failed = True
    for i in range(len(state.timestamp)):
        if state.measurements[i] <= max:
            if i > 0:
                break
        if state.measurements[i] > max:
            max = state.measurements[i]
            start_points.append(state.timestamp[i])
            start_values.append(state.measurements[i])

    try:
        state.timestamp = start_points
        state.measurements = start_values

    except IndexError as e:
        print(f'No failures in this time epoch for {tagname}')
        meta.failed = True
        end_timestamp = None
    except UnboundLocalError as e:
        print(f'No failures in this time epoch for {tagname}')
        meta.failed = True
        end_timestamp = None

    return state


def get_process_data_for_physical_pump(pump_no_tag: str, meta: Metadata, table: str,
                                       agg_time: int = None, avg: bool = False, std_dev: bool = False, get_state=True):
    """
    Create one scaler per physical pump found between start and stop time for the given pump tag
    :param meta: object to describe metadata in the data pipeline and training
    :param sequence: Whether to split the data into sequences
    :param n_steps: number of steps used in one sequence of LSTM
    :param HYG_PU_ID: tag_id from database of the pump that is being monitored
    :param timestamp: timestamp on where we want to start looking for data
    :param pump_no_tag: name of tag which follows pump number
    :param table: name of table for process data
    :param end: where to stop look for pump failures
    :param agg_time: How much time should be aggregated over
    :param avg: True if average is used as aggregation
    :param std_dev: True if standard deviation is used as aggregation
    :return: file in form of pickle object named after pump name and date of new pump
    """

    if not meta.failed:

        relevant_tagnames = get_relevant_pump_tags_general(pump_no_tag, meta)
        if get_state:
            relevant_tagnames.append(pump_no_tag[:8] + '_State')

        df = get_process_data(agg_time, avg, std_dev, relevant_tagnames, table, meta.data_training_start, meta.data_training_stop)

        return df
    else:
        return None


def get_pump_numbers(start, end, pump_no_tag):
    # Initialize the pump number tag
    pump_nos = Tag(pump_no_tag)
    # Get all pump failures for "tag" for all time
    pump_nos.get_measurement(start, end, 'measurement_temp')
    return pump_nos


def store_scaler_data(data, date, features, scaler, table, tag):
    sql = psql()
    insert_query = """insert into scaler_data (scaler_name,	features, data_training_start, data_training_stop,
        variance_, mean_)
        values
        (%s, %s, %s, %s, %s, %s);"""
    feature_string = ''
    for x in features:
        feature_string += x + ';'
    var_string = ''
    for x in scaler.var_:
        var_string += str(x) + ';'
    mean_string = ''
    for x in scaler.mean_:
        mean_string += str(x) + ';'
    insert_data = (f'scaler {table} {tag} {date[:10]}', feature_string, data['Time'].iloc[0], data['Time'].iloc[-1],
                   var_string, mean_string)
    for x in insert_data:
        print(x)
    sql.send_q(insert_query, insert_data)


def delete_metadata_record_LSTM(meta, model_name):
    sql = psql()
    query_delete = """delete from metadata
            where training_run = %s;"""
    print(meta.failed)
    try:
        print('DELETING AS THERE IS NO FAILED PUMPS')
        sql.send_q(query_delete, (meta.train_run + 1,))

    except TypeError as e:
        print('FAILED "DELETING AS THERE IS NO FAILED PUMPS"')
        query_previous_timestamp = """select training_run
                            from metadata
                            where model_name = %s
                            order by training_run desc
                            limit 1"""
        previous_run_data = sql.q_select(query_previous_timestamp, (model_name,))
        meta.train_run = previous_run_data[0][0]
        sql.send_q(query_delete, (meta.train_run,))

def delete_metadata_record_SVM(meta, model_name):
    sql = psql()
    query_delete = """delete from metadata_svm
            where training_run = %s;"""
    print(meta.failed)
    try:
        print('DELETING AS THERE IS NO FAILED PUMPS')
        sql.send_q(query_delete, (meta.train_run + 1,))

    except TypeError as e:
        print('FAILED "DELETING AS THERE IS NO FAILED PUMPS"')
        query_previous_timestamp = """select training_run
                            from metadata_svm
                            where model_name = %s
                            order by training_run desc
                            limit 1"""
        previous_run_data = sql.q_select(query_previous_timestamp, (model_name,))
        meta.train_run = previous_run_data[0][0]
        sql.send_q(query_delete, (meta.train_run,))


def update_metadata_record_LSTM(meta, model_name):
    sql = psql()
    try:
        query_previous_timestamp = """select training_run
                                from metadata
                                where model_name = %s
                                order by training_run desc
                                limit 1"""

        previous_run_data = sql.q_select(query_previous_timestamp, (model_name,))

        meta.train_run = previous_run_data[0][0]

        query_metadata_update = """update metadata
                set features=%s,
                training_stop = %s,
                data_training_stop = %s,
                accuracy = %s,
                number_of_epochs = %s
                where training_run = %s;"""
        metadata_update_data = (meta.features, datetime.datetime.now(), meta.data_training_stop, meta.accuracy,
                                meta.number_of_epochs, meta.train_run)
        print(metadata_update_data)
        sql.send_q(query_metadata_update, metadata_update_data)

    except TypeError as e:
        print(e)
        query_previous_timestamp = """select training_run
                        from metadata
                        where model_name = %s
                        order by training_run desc
                        limit 1"""

        previous_run_data = sql.q_select(query_previous_timestamp, (model_name,))

        meta.train_run = previous_run_data[0][0]
        query_metadata_update = """update metadata
                                set features=%s,
                                training_stop = %s,
                                data_training_stop = %s,
                                accuracy = %s,
                                number_of_epochs = %s
                                where training_run = %s;"""
        metadata_update_data = (meta.features, datetime.datetime.now(), meta.data_training_stop,
                                meta.accuracy, meta.number_of_epochs, meta.train_run)
        sql.send_q(query_metadata_update, metadata_update_data)

def update_metadata_test_set(meta):
    sql = psql()
    try:
        query_previous_timestamp = """select training_run
                                from metadata
                                where model_name = %s and checkpoint_path = %s
                                order by training_run desc
                                limit 1"""

        previous_run_data = sql.q_select(query_previous_timestamp, (meta.model_name, meta.checkpoint_path))

        meta.train_run = previous_run_data[0][0]

        query_metadata_update = """update metadata
                set accuracy = %s
                where training_run = %s;"""
        metadata_update_data = (meta.test_accuracy, meta.train_run)
        print(metadata_update_data)
        print(query_metadata_update)
        print(sql.cur.mogrify(query_metadata_update, metadata_update_data))
        sql.send_q(query_metadata_update, metadata_update_data)

    except TypeError as e:
        print(e)


def update_metadata_full_test_set(meta):
    sql = psql()
    try:
        query_previous_timestamp = """select training_run
                                from metadata
                                where model_name = %s
                                order by training_run desc
                                limit 1"""

        previous_run_data = sql.q_select(query_previous_timestamp, (meta.model_name,))

        meta.train_run = previous_run_data[0][0]

        query_metadata_update = """update metadata
                set test_accuracy = %s
                where training_run = %s;"""
        metadata_update_data = (meta.test_accuracy, meta.train_run)
        print(metadata_update_data)
        print(query_metadata_update)
        print(sql.cur.mogrify(query_metadata_update, metadata_update_data))
        sql.send_q(query_metadata_update, metadata_update_data)

    except TypeError as e:
        print(e)

def test_update_metadata_test_set():
    settings = XML('./Settings/model_settings.xml')

    meta = MetadataLSTM('test_update_metadata_test_set')
    # pre-process parameters
    meta.sequence = bool(int(settings.SearchForTag('sequence')))
    meta.sequence_steps = int(settings.SearchForTag('sequence_steps'))
    meta.pad = bool(int(settings.SearchForTag('pad')))
    meta.train_size = float(settings.SearchForTag('train_size'))
    meta.test_size = float(settings.SearchForTag('test_size'))
    meta.validation_size = float(settings.SearchForTag('validation_size'))
    print(meta.sequence, meta.sequence_steps, meta.pad)
    # Model parameters
    meta.number_of_epochs = int(settings.SearchForTag('number_of_epochs'))
    meta.batch_size = int(settings.SearchForTag('batch_size'))
    meta.neurons_layer_one = int(settings.SearchForTag('neurons_layer_one'))  # LSTM blocks
    meta.neurons_layer_two = int(settings.SearchForTag('neurons_layer_two'))  # LSTM blocks
    meta.layers = int(settings.SearchForTag('layers'))
    meta.no_outputs = 3

    if meta.layers == 2:
        number_of_lstm_neurons = f'{meta.neurons_layer_one}|{meta.neurons_layer_two}'
    else:
        number_of_lstm_neurons = str(meta.neurons_layer_one)


    meta.checkpoint_path = 'Testing insertion and update'
    insert_new_metadata_record_LSTM(meta, number_of_lstm_neurons, datetime.datetime.now())

    meta.test_accuracy = 0.95
    update_metadata_test_set(meta)

def update_metadata_record_SVM(meta):
    sql = psql()
    try:
        query_previous_timestamp = """select training_run
                                from metadata_svm
                                where model_name = %s
                                order by training_run desc
                                limit 1"""

        previous_run_data = sql.q_select(query_previous_timestamp, (meta.model_name,))

        meta.train_run = previous_run_data[0][0]

        query_metadata_update = """update metadata_svm
                set features=%s,
                training_stop = %s,
                data_training_stop = %s,
                accuracy = %s,
                method_ = %s, 
                c_ = %s, 
                gamma = %s, 
                coef0 = %s, 
                degree_ = %s
                where training_run = %s;"""
        metadata_update_data = (meta.features, datetime.datetime.now(), meta.data_training_stop, meta.accuracy,
                                meta.method, meta.c, meta.gamma, meta.coef0, meta.degree,
                                meta.train_run)
        print(metadata_update_data)
        print(sql.cur.mogrify(query_metadata_update, metadata_update_data))
        sql.send_q(query_metadata_update, metadata_update_data)

    except TypeError as e:
        print(e)
        query_previous_timestamp = """select training_run
                        from metadata
                        where model_name = %s
                        order by training_run desc
                        limit 1"""

        previous_run_data = sql.q_select(query_previous_timestamp, (meta.model_name,))

        meta.train_run = previous_run_data[0][0]
        query_metadata_update = """update metadata
                                set features=%s,
                                training_stop = %s,
                                data_training_stop = %s,
                                accuracy = %s,
                                number_of_epochs = %s,
                                stateful = %s
                                where training_run = %s;"""
        metadata_update_data = (meta.features, datetime.datetime.now(), meta.data_training_stop,
                                meta.accuracy, meta.number_of_epochs, meta.stateful, meta.train_run)
        sql.send_q(query_metadata_update, metadata_update_data)

def update_metadata_record_SVM_test(meta):
    sql = psql()
    try:
        # query_previous_timestamp = """select training_run
        #                         from metadata_svm
        #                         where model_name = %s
        #                         order by training_run desc
        #                         limit 1"""
        #
        # previous_run_data = sql.q_select(query_previous_timestamp, (meta.model_name,))
        #
        # meta.train_run = previous_run_data[0][0]

        query_metadata_update = """update metadata_svm
                set test_accuracy = %s,
                test_accuracy_hits = %s
                where training_run = %s;"""
        metadata_update_data = (meta.test_accuracy, meta.test_accuracy_hits,
                                meta.train_run)
        print(metadata_update_data)
        print(sql.cur.mogrify(query_metadata_update, metadata_update_data))
        sql.send_q(query_metadata_update, metadata_update_data)

    except TypeError as e:
        print(e)
        query_previous_timestamp = """select training_run
                        from metadata
                        where model_name = %s
                        order by training_run desc
                        limit 1"""

        previous_run_data = sql.q_select(query_previous_timestamp, (meta.model_name,))

        meta.train_run = previous_run_data[0][0]
        query_metadata_update = """update metadata
                                set features=%s,
                                training_stop = %s,
                                data_training_stop = %s,
                                accuracy = %s,
                                number_of_epochs = %s,
                                stateful = %s
                                where training_run = %s;"""
        metadata_update_data = (meta.features, datetime.datetime.now(), meta.data_training_stop,
                                meta.accuracy, meta.number_of_epochs, meta.stateful, meta.train_run)
        sql.send_q(query_metadata_update, metadata_update_data)


def insert_new_metadata_record_LSTM(meta, number_of_lstm_neurons, timestamp):
    sql = psql()
    query_init = """insert into metadata
        (model_name, training_start, data_training_start, train_validation_test_split, pre_process_steps, layers, 
        no_of_neurons, checkpoint_path)
        values
        (%s, %s, %s, %s, %s, %s, %s, %s);"""
    data_init = (meta.model_name, datetime.datetime.now(), timestamp,
                 f'{meta.train_size}_{meta.validation_size}_{meta.test_size}',
                 f'sequence={meta.sequence};sequence_steps={meta.sequence_steps};pad={meta.pad};', meta.layers,
                 number_of_lstm_neurons, meta.checkpoint_path)
    sql.send_q(query_init, data_init)

def insert_new_metadata_record_SVM(meta, timestamp):
    sql = psql()
    query_init = """insert into metadata_svm
        (model_name, training_start, data_training_start, train_validation_test_split, pre_process_steps,
        checkpoint_path, method_, c_, gamma, coef0, degree_)
        values
        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
    data_init = (meta.model_name, datetime.datetime.now(), timestamp,
                 f'{meta.train_size}_{meta.validation_size}_{meta.test_size}',
                 f'off_removal={meta.off_removal};top_removal={meta.top_removal};bottom_removal={meta.bottom_removal};'
                 f'top_removal_limit={meta.top_removal_limit};bottom_removal_limit={meta.bottom_removal_limit}'
                 f';top_removal_list={meta.top_removal_list};bottom_removal_list={meta.bottom_removal_list};',
                 meta.checkpoint_path, meta.method, meta.c, meta.gamma, meta.coef0, meta.degree)
    for x in data_init:
        print(x)

    sql.cur.mogrify(query_init, data_init)
    sql.send_q(query_init, data_init)


def get_previous_run_data(meta, model_name):
    sql = psql()
    # Get previous round training run and checkpoint path
    query_previous_timestamp = """select training_run, checkpoint_path, data_training_start
        from metadata
        where model_name = %s
        order by training_run desc;"""
    try:
        print(sql.cur.mogrify(query_previous_timestamp, (model_name,)))
        previous_run_data = sql.q_select(query_previous_timestamp, (model_name,))
        meta.previous_checkpoint_path_list = []
        meta.previous_data_training_start = []
        meta.train_run = previous_run_data[0][0]
        meta.previous_checkpoint_path = previous_run_data[0][1]
        print(previous_run_data)
        print(f'train_run {meta.train_run}')
        print(meta.previous_checkpoint_path)
        for x in previous_run_data:
            meta.previous_checkpoint_path_list.append(x[1])
            meta.previous_data_training_start.append(x[2])
    except IndexError as e:
        print(f'No previous training run for model {meta.model_name}')

def get_previous_run_data_SVM(meta, model_name):
    sql = psql()
    # Get previous round training run and checkpoint path
    query_previous_timestamp = """select training_run, checkpoint_path, data_training_start
        from metadata_svm
        where model_name = %s
        order by training_run desc;"""
    try:
        print(sql.cur.mogrify(query_previous_timestamp, (model_name,)))
        previous_run_data = sql.q_select(query_previous_timestamp, (model_name,))
        meta.previous_checkpoint_path_list = []
        meta.previous_data_training_start = []
        meta.train_run = previous_run_data[0][0]
        meta.previous_checkpoint_path = previous_run_data[0][1]
        print(previous_run_data)
        print(f'train_run {meta.train_run}')
        print(meta.previous_checkpoint_path)
        for x in previous_run_data:
            meta.previous_checkpoint_path_list.append(x[1])
            meta.previous_data_training_start.append(x[2])
    except IndexError as e:
        print(f'No previous training run for model {meta.model_name}')

def get_previous_run_data_SVM_no_meta():
    sql = psql()
    # Get previous round training run and checkpoint path
    query_previous_timestamp = """select training_run, model_name
        from metadata_svm
        where features is not null
        order by training_run desc;"""
    print(sql.cur.mogrify(query_previous_timestamp))
    try:
        print(sql.cur.mogrify(query_previous_timestamp))
        previous_run_data = sql.q_select(query_previous_timestamp)
        train_run = []
        model_name = []

        print(previous_run_data)

        for x in previous_run_data:
            train_run.append(x[0])
            model_name.append(x[1])
    except IndexError as e:
        print('Could not find any models')
    return train_run, model_name



def store_test_set(x_test, y_test, features):
    dim1 = x_test[0].shape[0]
    dim2 = x_test[0][0].shape[0] + 1
    print(dim1, dim2)
    data = np.array([])
    #data = x_test.concatenate(x_test, y_test)
    print(features)
    sql = psql()
    xml = XML('./Settings/tag_id_generic_tags.xml')
    records_list_template = ','.join(['%s'] * (len(features[:-1])))
    query = """insert into test_data values {};""".format(records_list_template)
    for i in range(len(x_test)):
        for j in range(len(x_test[0])):
            insert_data = []
            tag_id = xml.SearchForTag('State')
            insert_data.append((x_test[i][j][0], tag_id, int(y_test[i][j])))

            for k, feat in enumerate(features[1:-1]):
                tag_id = xml.SearchForTag(feat)
                insert_data.append((x_test[i][j][0], tag_id, x_test[i][j][k+1]))

            sql.send_q(query, insert_data)







