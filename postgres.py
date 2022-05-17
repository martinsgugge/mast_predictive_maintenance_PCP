import psycopg2 as pg
from XML import XML

class psql:
    database = ""
    host = ""
    username = ""
    conn = pg
    cur = conn
    pw = None
    connected = False
    error = None

    def __init__(self, database=None, host=None, username=None, pw=None,
                 xml_file='./DBConnection.xml'):
        """
        Creates a psql object

        :param database: databasename
        :param server: hostaddress
        :param username: username for the database
        """

        if xml_file == None:
            self.database = database
            self.host = host
            self.username = username
            self.pw = pw
            self.connect()
        else:
            xml = XML(xml_file)

            self.database = xml.SearchForTag('database')
            self.host = xml.SearchForTag('host')
            self.username = xml.SearchForTag('user')
            self.pw = xml.SearchForTag('pw')

            self.connect()


    def connect(self):
        """
        connects the psql object to a database and creates a cursor within the psql object

        :param pw: password
        :return:
        """

        try:

            self.conn = self.conn.connect("dbname = '{0}' user = '{1}' host = '{2}' password = '{3}'".format(
                self.database, self.username, self.host, self.pw))

            self.cur = self.conn.cursor()
            #print("connected to {0}".format(self.database))
            self.connected = True
        except pg.errors.OperationalError as e:
            print("Failed to connect")
            print(e)
            self.error = "Failed to connect"
            self.connected = False

    def reconnect(self):
        try:
            self.conn = self.conn.connect("dbname = '{0}' user = '{1}' host = '{2}' password = '{3}'".format(
                self.database, self.username, self.server, self.pw))

            self.cur = self.conn.cursor()
            print("connected to {0}".format(self.database))
        except:
            print("Failed to connect")

    def disconnect(self):
        """
        disconnects from the database

        :return:
        """
        self.conn.close()
        print("disconnected from {}".format(self.database))

    def select_from_measurement(self, tag_id):
        """

        :param tag_id: id for the wanted tag
        :return: 2D touple with timestamp and values
        """
        self.cur.execute("select tid, verdi from measurement where tag_id = '{0}'".format(tag_id))
        rows = self.cur.fetchall()
        return rows

    async def insert_measurements(self, comp_id, value, timestamp):
        """
        :param comp_id:
        :param value:
        :param timestamp:
        :return void:
        """

        query = "CALL public.insert_measurements(%s, %s, %s)"
        await self.cursor.execute(query, (comp_id, value, timestamp))
        await self.connection.commit()

    def test_insert_measurements(self):
        """
        test function to validate insertion of arrayed values
        test function to validate unique PK (component_id and meas_timestamp)
        :return: void
        """
        x = [1, 2, 3]
        y = [3.14, 3.14, 3.14]
        z = ['2019-09-09 00:00:00', '2019-09-09 01:00:00', '2019-09-09 02:00:00']
        con = psql('postgres', '', 'localhost', '5432', 'process_data')
        con.insert_measurements(x, y, z)

    """https://stackoverflow.com/questions/20699196/python-list-to-postgresql-array"""

    def insert(self, query):
        self.cur.execute(query)

    def q_select(self, query, args=None):
        """
        Sends a query to the connected database and returns the data

        :param query: string, PostgreSQL query
        :return: object of rows
        """
        rows = []
        try:
            self.cur.execute(query, args)
            rows = self.cur.fetchall()
        except pg.errors.InFailedSqlTransaction as e:
            print(e)
            rows = False

        return rows

    def get_tag_list(self, query):
        """
        Sends a query to the database and transposes the result

        :param query: string, PostgreSQL query
        :return: Data from the query
        """
        self.cur.execute(query)
        # rows = self.transpose_sql_query(self.cur.fetchall)
        rows = self.cur.fetchall()

        print(rows[0][0])
        print(rows[0][1])
        #rows = self.transpose_sql_query(rows)
        # rows = self.transpose_sql_query(rows)

        return rows

    def transpose_sql_query(self, rows):
        """
        Extracts the 0th and 1st columns from rows and transposes them

        :param rows: cursor object, data returned from a query
        :return: list, 0th and 1st column of rows
        """
        time = []
        val = []
        for row in rows:
            time.append(row[0])
            val.append(row[1])
        data = [time, val]
        return data

    def get_tagnames_from_station(self, station_name):
        """
        Selects all tags related to a station name

        :param station_name: string, name of the station
        :return: list of string, tagnames related to station_name
        """
        query = """select tag from tag where station_id = 
        (select station_id from station where station_name = '{}')
        order by tag_id desc""".format(station_name)
        self.cur.execute(query)
        tagnames = self.cur.fetchall()
        tagnames2 = []
        for row in tagnames:
            # print(row)
            a = row[0]
            b = "(',)"
            for char in b:
                a = a.replace(char, "")
            tagnames2.append(a)
        return tagnames2

    def send_q(self, query, data=None):
        try:
            self.cur.execute(query, data)
            self.conn.commit()
        except pg.errors.UniqueViolation as e:
            print(e)
            self.conn.rollback()
        except pg.errors.SyntaxError as e:
            pass
            print(e)
        except pg.errors.InvalidTextRepresentation as e:
            pass
            print(e)

    def insert_calculations(self, der_name, data):
        # Make tagname a tuple for passing to psycopg2
        t = tuple((der_name,))
        der_id = self.q_select("""select derivative_id from derivative where derivative_name = %s;""", t)

        # Unpack derivative id
        der_id = der_id[0]

        # Insert calculation to database
        for x in data:
            try:
                self.send_q("""select insert_calculation(%s, %s::timestamp, %s::interval);""", (der_id, x[0], x[1]))
            except pg.errors.CannotCoerce as e:
                self.conn.rollback()
                self.send_q("""select insert_calculation(%s, %s::timestamp, %s::real);""", (der_id, x[0], x[1]))

    def insert_calculation(self, der_name, data):
        # Make tagname a tuple for passing to psycopg2
        t = tuple((der_name,))
        der_id = self.q_select("""select derivative_id from derivative where derivative_name = %s;""", t)

        # Unpack derivative id
        der_id = der_id[0]

        # Insert calculation to database
        try:
            self.send_q("""select insert_calculation(%s, %s::timestamp, %s::interval);""", (der_id, data[0], data[1]))
        except pg.errors.CannotCoerce as e:
            self.conn.rollback()
            self.send_q("""select insert_calculation(%s, %s::timestamp, %s::real);""", (der_id, data[0], data[1]))


class Tag:
    """
    Class to hold a process tags information
    """
    tagID = None
    connected_tag = None
    station_id = None
    unit_type = None
    tag = None
    tag_desc = None
    interval = None
    interval_unit = None
    comment = None

    pw = None
    timestamp = None
    measurements = None

    sql = None
    cursor = None
    failed = False

    def __init__(self, tagname, create=False):
        """
        Object of a tag

        :param tagname: string, name of tag
        :param create: bool, if create is false, get tag meta info
        """

        """Connects to database"""
        self.sql = psql()

        if not create:
            """Gets metainformation of the given tag"""
            self.get_tag(tagname)
            #print(self.tag + ' initialized')

    def prepare_inserts(self, data):
        """
        Transforms data so that it can be inserted and creates the query for inserting into aggregation

        :param data: 2D list [timestamp, tag_id, value]
        :return: query and data for insertion into aggregation
        """

        tup_list = []
        for i in range(len(data[0])):
            tup_list.append((data[0][i], data[1][i], data[2][i]))

        records_list_template = ','.join(['%s'] * (len(tup_list)))
        insert_query = 'insert into aggregations(meas_time, tag_id, agg_val) values {}'.format(records_list_template)

        return insert_query, tup_list


    def prepare_inserts_wide(self, data):

        tup_list = []
        for i in range(len(data[0])):
            tup_list.append((data[0][i], data[1][i], data[2][i]))
        # print(len(tup_list))
        for d in data:
            print(d)

        records_list_template = ','.join(['%s'] * (len(data)))
        insert_query = 'insert into vibration_high_ts2(meas_time, tag_id, agg_val) values {}'.format(
            records_list_template)

        return insert_query, tup_list


    def upload_aggregate(self):
        """
        Uploads aggregated data to the database after processing for a tag

        :return:
        """
        id_list = []
        for i in range(len(self.measurements)):
            id_list.append(self.tagID)
        query, data = self.prepare_inserts([list(self.timestamp), id_list, list(self.measurements)])

        print(self.sql.cur.mogrify(query, data).decode('utf8'))

        try:

            self.sql.cur.execute(query, data)

            self.sql.conn.commit()
        except pg.errors.UniqueViolation as e:
            print(e)
            self.sql.conn.rollback()
        except pg.errors.SyntaxError as e:
            print(e)
        #self.sql.disconnect()

    def upload_meta1(self):
        """
        Uploads meta information (such as attributes of this class) to the connected database

        :return:
        """

        #Sett til meas_interval
        print(self.tagID, self.tag_desc, self.tag)
        try:

            self.sql.cur.execute("""insert into tag(tag_id, station_id, unit_type, tag, tag_description, meas_interval,
            meas_interval_unit)
            values(%s,%s,%s,%s,%s,%s,%s)
            on conflict (tag_id)
            do update 
                set station_id=%s,
                tag_description=%s
            where tag.tag_id = %s;
            """,(self.tagID, self.station_id, self.unit_type, self.tag, self.tag_desc, self.interval,
                 self.interval_unit, self.station_id, self.tag_desc, self.tagID
                 ))
        except pg.errors.UniqueViolation as e:
            #self.sql.cur.tr
            print(e)
            self.sql.conn.rollback()


        #self.sql.insert(query)

    def upload_meta2(self):
        """
        Uploads foreign keys to the database for this tag
        :return:
        """
        self.sql.cur.execute("""update tag set FK_tag_id = %s where tag_id = %s;""",
                             (self.connected_tag, self.tagID))
        print(self.connected_tag, self.tagID)

    def update_meta(self):
        """
        Updates a tags name, description and tag id
        :return:
        """
        self.sql.cur.execute("""update tag set tag = %s, tag_description = %s where tag_id = %s;""",
                             (self.tag, self.tag_desc, self.tagID))

    def update_tag(self):
        rows = self.sql.cur.execute("""select pInsertOrUpdateTag(%s::int, %s::int, %s::int, %s::text, %s::text, %s::text ,
        %s::smallint, %s::text, %s::text)""",
                             (self.tagID, self.connected_tag, self.station_id, self.unit_type, self.tag, self.tag_desc, self.interval,
                              self.interval_unit, self.comment))
        return rows

    def get_tag(self, tagname):
        """
        Gets meta information of a tag from table tag in database and sets properties in the Tag object

        :param tagname: string, name of tag
        :return:
        """

        query = """select * from tag where tag = %s"""
        #print(self.sql.cur.mogrify(query, (tagname,)))
        rows = self.sql.q_select(query, (tagname,))
        self.tagID = rows[0][0]
        self.connected_tag = rows[0][1]
        self.station_id = rows[0][2]
        self.unit_type = rows[0][3]
        self.tag = tagname
        self.tag_desc = rows[0][5]
        self.interval = rows[0][6]


    def get_measurement(self, time_from, time_to, table='measurement'):
        """
        Gets measurement for the given time period and tag

        :param time_from: string or datetime, Start time for dataset
        :param time_to: string or datetime, Stop time for dataset
        :return:
        """

        query = """select meas_time, meas_value
                from {}
                where tag_id = %s and meas_time between %s::timestamp and %s::timestamp
                order by meas_time asc;""".format(table)
        #print(self.sql.cur.mogrify(query, (self.tagID, time_from, time_to)))
        rows = self.sql.q_select(query, (self.tagID, time_from, time_to))

        self.timestamp = []
        self.measurements = []
        for row in rows:
            self.timestamp.append(row[0])
            self.measurements.append(float(row[1]))

    def append_measurement(self, time_from, time_to, table='measurement'):
        """
        Gets measurement for the given time period and tag

        :param time_from: string or datetime, Start time for dataset
        :param time_to: string or datetime, Stop time for dataset
        :return:
        """

        query = """select meas_time, meas_value
                from {}
                where tag_id = %s and meas_time between %s::timestamp and %s::timestamp
                order by meas_time asc;""".format(table)
        # print(self.sql.cur.mogrify(query, (self.tagID, time_from, time_to)))
        rows = self.sql.q_select(query, (self.tagID, time_from, time_to))

        # skip = False
        # try:
        #     print(rows[0][0])
        # except IndexError as e:
        #     skip = True

        for row in rows:
            self.timestamp.append(row[0])
            self.measurements.append(float(row[1]))

        #self.sql.disconnect()

    def get_avg_measurement(self, time_from, time_to, agg_time, aggregation, table='measurement'):
        """
        Gets measurement for the given time period and tag

        :param time_from: string or datetime, Start time for dataset
        :param time_to: string or datetime, Stop time for dataset
        :return:
        """

        query = """
        SELECT time_bucket('{} minutes', meas_time) AS avg_min, {}(meas_value)
        FROM {}
        where tag_id = %s and meas_time between %s and %s
        GROUP BY avg_min
        ORDER BY avg_min ASC;""".format(agg_time, aggregation, table)

        rows = self.sql.q_select(query, (self.tagID, time_from, time_to))
        self.timestamp = []
        self.measurements = []



        if aggregation == 'stddev_samp':
            for row in rows:
                try:
                    self.timestamp.append(row[0])
                    self.measurements.append(float(row[1]))
                except TypeError as e:
                    self.timestamp.append(row[0])
                    if len(self.measurements) > 0:
                        self.measurements.append(self.measurements[-1])
                    else:
                        self.measurements.append(0.0)
        else:
            for row in rows:
                try:

                    self.timestamp.append(row[0])
                    self.measurements.append(float(row[1]))
                except TypeError as e:
                    self.timestamp.append(row[0])
                    self.measurements.append(self.measurements[-1])

    def append_agg_measurement(self, time_from, time_to, agg_time, aggregation, table='measurement'):
        """
        Gets measurement for the given time period and tag

        :param time_from: string or datetime, Start time for dataset
        :param time_to: string or datetime, Stop time for dataset
        :return:
        """

        query = """
        SELECT time_bucket('{} minutes', meas_time) AS avg_min, {}(meas_value)
        FROM {}
        where tag_id = %s and meas_time between %s and %s
        GROUP BY avg_min
        ORDER BY avg_min ASC;""".format(agg_time, aggregation, table)

        #print(self.sql.cur.mogrify(query, (self.tagID, time_from, time_to)))
        rows = self.sql.q_select(query, (self.tagID, time_from, time_to))

        if aggregation == 'stddev_samp':
            for row in rows:
                try:
                    self.timestamp.append(row[0])
                    self.measurements.append(float(row[1]))
                except TypeError as e:
                    self.timestamp.append(row[0])
                    if len(self.measurements) > 0:
                        self.measurements.append(self.measurements[-1])
                    else:
                        self.measurements.append(0.0)

        else:
            for row in rows:
                try:

                    self.timestamp.append(row[0])
                    self.measurements.append(float(row[1]))
                except TypeError as e:
                    self.timestamp.append(row[0])
                    self.measurements.append(self.measurements[-1])

def connect(xml_file):
    from XML import XML
    """
    Function to connect to a database with the given xml file
    Last edited 05.08.2021
    By Martin Holm
    :return: connected psycopg sql object
    """

    sql = psql()
    sql.connect()
    return sql


if __name__ == '__main__':
    sql = connect("C:/Users/prod/PycharmProjects/Hent_data_fra_postgres_20.10.20/DBconnection.xml")
    data = sql.get_tag_list("""select * from tag where tag in ('HYG_FT01',
                'HYG_PU12_PW_PV', 'HYG_PU12_TQ_PV', 'HYG_PU12_SF_PV', 'HYG_PU12_MO', 'HYG_PT03',
                'HYG_PT04', 'HYG_PU13_PW_PV', 'HYG_PU13_TQ_PV', 'HYG_PU13_SF_PV', 'HYG_PU13_MO', 'HYG_PT05',
                'HYG_PT06', 'HYG_PU14_PW_PV', 'HYG_PU14_TQ_PV', 'HYG_PU14_SF_PV', 'HYG_PU14_MO', 'HYG_PT07',
                'HYG_PT08', 'HYG_PU15_PW_PV', 'HYG_PU15_TQ_PV', 'HYG_PU15_SF_PV', 'HYG_PU15_MO', 'HYG_PT09',
                'HYG_PT10', 'HYG_PU16_PW_PV', 'HYG_PU16_TQ_PV', 'HYG_PU16_SF_PV', 'HYG_PU16_MO', 'HYG_PT11',
                'HYG_TT02', 'HYG_TT03', 'HYG_TT04', 'HYG_TT05', 'HYG_FT02',
                'HYG_PU17_PW_PV', 'HYG_PU17_TQ_PV', 'HYG_PU17_SF_PV', 'HYG_PU17_MO', 'HYG_PT14',
                'HYG_PT12', 'HYG_PU18_PW_PV', 'HYG_PU18_TQ_PV', 'HYG_PU18_SF_PV', 'HYG_PU18_MO', 'HYG_PT13',
                'HYG_PT15', 'HYG_PU19_PW_PV', 'HYG_PU19_TQ_PV', 'HYG_PU19_SF_PV', 'HYG_PU19_MO', 'HYG_PT16',
                'HYG_TT06')""")

    from CSV import ArrayTocsv

    ArrayTocsv(['Tag_id', 'fk_tag_id', 'station_id', 'unit_type', 'tagname', 'tag_description', 'meas_interval',
                'interval_unit', 'comment'], data, 'taglist.csv')