from postgres import *
from CSV import *
from stat_fun import average_hourly
from XML import XML
import time
import datetime
import sys
import numpy as np


def connect(xml_file='C:/Users/prod/PycharmProjects/Hent_data_fra_postgres_20.10.20/DBConnection.xml'):
    """
    Connects to a database using an xml file
    :return: connected psql object
    """
    xml = XML(xml_file)

    database = xml.SearchForTag('database')
    host = xml.SearchForTag('host')
    user = xml.SearchForTag('user')
    pw = xml.SearchForTag('pw')
    sql = psql(database, host, user)
    sql.connect(pw)
    return sql



def select_wide_vib(start, stop):
    xml = XML('C:/Users/prod/PycharmProjects/Aggregation/DBConnection.xml')
    database = xml.SearchForTag('database')
    host = xml.SearchForTag('host')
    user = xml.SearchForTag('user')
    pw = xml.SearchForTag('pw')
    sql = psql()
    query = """select * from vibration_high_ts2
    where meas_time >= '{}' and meas_time <= '{}' order by meas_time asc;""".format(start, stop)
    rows = sql.q_select(query)
    time = []
    HYG_PU19_PL = []
    HYG_PU19_PI = []
    HYG_PU19_PO = []
    HYG_PU19_VL = []
    HYG_PU19_VI = []
    HYG_PU19_VO = []
    HYG_FT02 = []
    HYG_PT15 = []
    HYG_PT16 = []
    HYG_PU19_MO = []
    HYG_PU19_PW = []
    HYG_PU19_TQ = []
    HYG_PU19_SF = []
    for row in rows:
        #print(row)
        time.append(row[0])
        HYG_PU19_PL.append(row[1])
        HYG_PU19_PI.append(row[2])
        HYG_PU19_PO.append(row[3])
        HYG_PU19_VL.append(row[4])
        HYG_PU19_VI.append(row[5])
        HYG_PU19_VO.append(row[6])
        HYG_FT02.append(row[7])
        HYG_PT15.append(row[8])
        HYG_PT16.append(row[9])
        HYG_PU19_MO.append(row[10])
        HYG_PU19_PW.append(row[11])
        HYG_PU19_TQ.append(row[12])
        HYG_PU19_SF.append(row[13])
    del rows
    sql.disconnect()
    del sql
    return [time, HYG_PU19_PL, HYG_PU19_PI, HYG_PU19_PO, HYG_PU19_VL, HYG_PU19_VI, HYG_PU19_VO,
            HYG_FT02, HYG_PT15, HYG_PT16, HYG_PU19_MO, HYG_PU19_PW, HYG_PU19_TQ, HYG_PU19_SF]

def select_high_hyg(start, stop, tagname='*', table='high_speed_big_reduced', q_tag_list=False, numpy_arr=True):
    sql = psql()
    if q_tag_list == True:
        rows = sql.q_select("SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'high_speed_big';")
        tags = []
        #print('got column names')

        for row in rows:
            tags.append(row[0])

    taglist = ''

    if len(tagname) > 1:
        for tag in tagname:
            taglist += tag + ', '
        taglist = taglist[:-2]
        #print(taglist)

    query = """select {} from {} where meas_time >= '{}' and meas_time <= '{}' order by meas_time asc;"""\
        .format(taglist, table, start, stop)
    #print(sql.cur.mogrify(query))
    rows = sql.q_select(query)
    data = []

    if numpy_arr == True:
        data = np.array(rows).T
    else:
        for row in rows:
            data.append(row)

    #print(len(data))

    del rows
    sql.disconnect()
    del sql
    if q_tag_list == True:
        return tags, data
    else:
        return data

def select_high_hyg_agg(start, stop, agg_time, tagname='*', table='high_speed_big_reduced', q_tag_list=False, numpy_arr=True,
                        aggregation=''):
    sql = psql()
    if q_tag_list == True:
        rows = sql.q_select("SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'high_speed_big';")
        tags = []
        #print('got column names')

        for row in rows:
            tags.append(row[0])

    taglist = ''

    if len(tagname) > 1:
        #Skip meas_time
        if aggregation == '':
            for tag in tagname[1:]:
                taglist += tag + ', '
            taglist = taglist[:-2]
        else:
            for tag in tagname[1:]:
                taglist += aggregation + '(' + tag + '), '
            taglist = taglist[:-2]


        #print(taglist)

    query = """  --High speed 1min
    SELECT time_bucket('{} second', meas_time) AS five_min, {}
    FROM high_speed_big_reduced
    where meas_time between %s and %s
    GROUP BY five_min
    ORDER BY five_min asc;"""\
        .format(agg_time, taglist)
    #print(sql.cur.mogrify(query))
    rows = sql.q_select(query, (start, stop))
    data = []

    if numpy_arr == True:
        data = np.array(rows).T
    else:
        for row in rows:
            data.append(row)

    #print(len(data))

    del rows
    sql.disconnect()
    del sql
    if q_tag_list == True:
        return tags, data
    else:
        return data


if __name__ =='__main__':

    date_stop = str(datetime.datetime.date(datetime.datetime.now()) - datetime.timedelta(days=1)) #10.03.2021
    date_start = str(datetime.datetime.date(datetime.datetime.now()) - datetime.timedelta(days=8))
    """date_stop = ['2020-03-01', '2020-05-01', '2020-07-01',
                 '2020-09-01', '2020-11-01',  '2021-01-01', '2021-03-01']
    date_start = ['2020-01-01', '2020-03-01', '2020-05-01', '2020-07-01',
                  '2020-09-01', '2020-11-01', '2021-01-01']
    aggregate_all_values(date_start, date_stop)"""
    #aggregate_all_values(['2021-03-01'], ['2021-03-24'])

    #move_data_prod_to_prod('dmf_trans_db', 'dmf_trans_db_dev', 'measurement', 'measurement',date_start, date_stop)

    select_high_hyg()