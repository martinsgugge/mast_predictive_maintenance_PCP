import csv
import datetime
import os

import numpy

class CSV():
    error_msg = None
    generic_error = False
    def ArrayTocsv(self, header_array,ArrayToWrite,file):
        """
        Writes a 2D array or list to csv file
        :param header_array: list, headers for the csv file
        :param ArrayToWrite: list of list or 2D array, data which should be written
        :param file: string, name of the file
        :return:
        """
        ArrayToWrite = zip(*ArrayToWrite)
        with open(file, 'w', newline='') as csvFile:

            writer = csv.writer(csvFile, delimiter=',')
            if header_array is not None:
                print('skriver overskrift')
                writer.writerow(header_array)
            print('Skriver verdier')
            for row in ArrayToWrite:
                # print(row)
                writer.writerow(row)
            # writer.writerows(ArrayToWrite)
        csvFile.close()


    def format_time(self, data, header):
        """
        Formats time from postgres format(millisecond and timezone) to dd.mm.yyyy hh:min:sek
        Created by Martin Holm
        Changelog: 21.01.21 Created

        :param data: Data matrix with time and data
        :return: Formatted data matrix with time and data
        """
        for i in range(len(header)):
            for j in range(len(data[i * 2])):
                try:
                    self.generic_error = False
                    data[i * 2][j] = data[i * 2][j].replace(microsecond=0)
                except TypeError as e:
                    print(e)
                    print(data[i * 2][j])
                    self.generic_error = True
                    self.error_msg = e
                try:
                    self.generic_error = False
                    data[i * 2][j] = data[i * 2][j].replace(tzinfo=None)
                except TypeError as e:
                    print(e)
                    print(data[i*2][j])
                    self.generic_error = True
                    self.error_msg = e

        return data


    def format_data(self, data, header):
        """
        Formats data values from . to ,
        Created by Martin Holm
        Changelog: 21.01.21 Created

        :param data: Data matrix with time and data
        :return: Formatted data matrix with time and data
        """
        for i in range(len(header)):
            temp = []

            for j in range(len(data[i*2+1])):

                temp.append(str(data[i * 2 + 1][j]).replace('.', ','))
            data[i * 2 + 1] = temp

        return data


    def plotarrayTocsv(self, header, data, filename, dec=','):
        """
        Writes a list of list or 2D array which has been processed for plotting to a CSV file

        :param header: list of string, name of headers
        :param data: list of list or 2D array
        :param filename: name of file to be written
        :param dec: string, decimal delimiter
        :return:
        """
        with open(filename, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=';')
            print('skal skrive')
            data = self.format_time(data, header)
            if dec == ',':
                data = self.format_data(data, header)
            for i in range(len(header)):
                header.insert(i*2, 'Tid')

            data = zip(*data)

            writer.writerow(header)
            for row in data:
                writer.writerow(row)


    def plotarrayTocsvOneTime(self, header, data, filename):
        """
        Writes a list of list or 2D array to a csv file where there is only one time column

        :param header: list of string, header for the file
        :param data: list of list or 2D array
        :param filename: name of the file
        :return:
        """
        with open(filename, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=';')
            print(len(header), len(data))
            write_list = [data[0]]

            headers = ['Tid']
            for i in range(len(header)):
                headers.append(header[i])
            for i in range(len(header)-1):
                #header.insert(i*2,'Tid')
                print(header[i], data[i*2])
                write_list.append(data[i*2+1])
            data = write_list
            data = zip(*data)
            writer.writerow(headers)
            for row in data:
                writer.writerow(row)


    def plotTagsTocsv(self, header, tags, filename):
        """
        Write measurements for a list of tags to a csv file

        :param header: list of strings, name of headers excluding time headers
        :param tags: Tag object with data
        :param filename: string, name of file
        :return:
        """
        with open(filename, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=';')
            print('skal skrive')
            for i in range(len(header)):
                header.insert(i*2, 'Tid')

            writer.writerow(header)
            data = []
            for i in range(len(tags)):
                data.append(tags[i].timestamp)
                data.append(tags[i].measurement)

            writer.writerows(data)


    def csvToArray(self, file, delimiter):
        """
        Reads a csv file

        :param file: string, name of file
        :param delimiter:
        :return:
        """
        try:
            self.generic_error = False
            with open(file, 'r') as csvFile:
                reader = csv.reader(csvFile, delimiter=delimiter)
                for row in reader:
                    x = list(reader)
                    result = numpy.array(x)

                csvFile.close()
            return result
        except OSError as e:
            print(e)
            self.generic_error = True
            self.error_msg = e
            return None


    def merge_lists(self, l1, l2, header=None):
        """
        merge two lists together in a table like manner such as to create rows

        :param l1: list of data
        :param l2: list of data
        :param header:
        :return:
        """
        list = []
        for j in range(len(l1)):
            while True:
                try:
                    tup = (l1[j], l2[j])
                except IndexError:
                    if len(l1) > len(l2):
                        l2.append('')
                        tup = (l1[j], l2[j])
                    elif len(l1) < len(l2):
                        l1.append('')
                        tup = (l1[j], l2[j])
                    continue
                list.append(tup)

                break
        return list


    def create_or_append_csv(self, filename, header, data):
        filename = filename + '.csv'
        if not os.path.exists(filename):
            print('test')
            self.ArrayTocsv(header, [data], filename)
        else:
            with open(filename, 'a') as f_object:
                # Pass this file object to csv.writer()
                # and get a writer object
                writer_object = csv.writer(f_object)

                # Pass the list as an argument into
                # the writerow()
                writer_object.writerow(data)

                # Close the file object
                f_object.close()


    def remove_file(self, filename):
        """
            attempts to remove the given file from the file system
            :param filename: string, name of the file
            :return:
        """
        try:
            os.remove(filename)
        except OSError:
            pass
    def getstuff(self, filename):
        with open(filename, "r") as csvfile:
            datareader = csv.reader(csvfile, delimiter=';')
            for i in range(len(datareader)//1000):
                return datareader[i*1000:(i+1)*1000]


    def getdata(self, filename):

            for row in self.getstuff(filename):
                yield row


    def find_csv_delimiter(self, example_file):
        with open(example_file, 'rb') as csvfile:
            # detect the delimiter used
            dialect = csv.Sniffer().sniff(csvfile.read(1024))

            # return to the beginning of the file
            csvfile.seek(0)

            # file should now open with the correct delimiter.
            reader = csv.reader(csvfile, dialect)
            return dialect