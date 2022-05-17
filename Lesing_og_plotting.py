from CSV import *
import plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import numpy as np
from postgres import *





def plot_sql(tagname, query, pw):
	"""
	deprecated, connects to a database, queries database and plots data in a line chart

	:param tagname: string, name of tags
	:param query: string, postgresql query
	:param pw: string, password
	:return:
	"""
	fig = go.Figure()
	sql = psql("dmf_prosess", "localhost", "postgres")
	sql.connect(pw)
	for i in range(len(tagname)):
		print(i)
		rows = sql.q_select(query[i])

		time = []
		data = []

		for row in rows:
			time.append(row[0])
			data.append(row[1])
		data = low_pass_filter(data[0], 0.5, 30, data)

		fig.add_trace(go.Scatter(x = time, y = data, name = tagname[i]))

	sql.disconnect()

	#fig.update_layout(yaxis_title=unit)
	py.offline.plot(fig, filename=tagname[i]+'.html')
	#fig.show()

def plot_array(filename, tagname,data):
	"""
	Plots data in a line chart

	:param filename: string, name of html file
	:param tagname: string, name of tag
	:param data: list of lists, time-series data to be plotted
	:return:
	"""
	fig = make_subplots(specs=[[{"secondary_y": True}]])
	for i in range(len(tagname)):
		print(i)
		fig.add_trace(go.Scatter(x=data[i * 2], y=data[i * 2 + 1], name=tagname[i]), secondary_y=False)

	#fig.update_layout(yaxis_title=unit)
	py.offline.plot(fig, filename=filename+'.html')

def create_colors():
	"""
	Changes the color scheme of plotly

	:return: object of colors
	"""
	color1 = py.colors.qualitative.Light24
	color2 = py.colors.qualitative.Dark24
	color3 = py.colors.qualitative.Alphabet

	colors = []
	for i in range(len(color1)):
		colors.append(color1[i])
	for i in range(len(color2)):
		colors.append(color2[i])
	for i in range(len(color3)):
		colors.append(color3[i])
	return colors

def plot(filename, tagname, data, right_hand_axis=None, single_time=False):
	"""
	Plots data with tagnames and stores in filename.html

	:param filename: name of html file output
	:param tagname: List of tagnames to be plotted
	:param data: Matrix of data to be plotted
	:param right_hand_axis: List of tags which should be plotted on right hand y-axis, can contain part of the names.
	e.g. to plot all tags from HYG insert 'HYG' into list
	:return: No return
	"""

	"""Creates a subplot for the second hand y-axis"""
	fig = make_subplots(specs=[[{"secondary_y": True}]])
	plotted = False
	first_axis = ''
	second_axis = ''
	"""Sets another colorscheme than standard Plotly, for other schemes see in console:
	import plotly.express as px
	fig = px.colors.qualitative.swatches()
	fig.show()"""
	colors = create_colors()

	calculation_suffixes = ['_MA' '_LPF', '_sum', '_RoT'] #Used to remove suffixes for the axis labels
	tag_length = len(tagname)
	"""Attempt at making lineshift on axis labels
	y_ax_1_multiplier = 1
	y_ax_2_multiplier = 1"""
	#num_label_excluded = 0
	k = 0
	second_color = False

	for i in range(tag_length):
		print(i)
		plotted = False	#Variable to check if the tag has been plotted on the right hand axis
		"""Sets up specified data on right hand y-axis"""
		if right_hand_axis is not None:
			for j in range(len(right_hand_axis)):
				if right_hand_axis[j] in tagname[i]:

					if single_time == False:
						#Adds the tagdata to the plot
						fig.add_trace(go.Scatter(x=data[i * 2], y=data[i * 2 + 1], name=tagname[i],
												 marker=dict(color=colors[i], size=15)), secondary_y=True)
						plotted = True

					elif single_time == True:
						# Adds the tagdata to the plot
						fig.add_trace(go.Scatter(x=data[0], y=data[i+1], name=tagname[i],
												 marker=dict(color=colors[i], size=15)), secondary_y=True)
						plotted = True

					"""Removes unwanted tagnames from the right y-axis"""
					if remove_tags_from_taglist(calculation_suffixes, tagname[i]) != None:
						second_axis += remove_tags_from_taglist(calculation_suffixes, tagname[i]) + ", "
						print("RHA")


		"""If the data is not specified to be on right hand y-axis, put on left hand y-axis"""
		if plotted is False:
			if not single_time:
				# Adds the tagdata to the plot
				fig.add_trace(go.Scatter(x=data[i * 2], y=data[i * 2 + 1],
										 name=tagname[i],
										marker=dict(color=colors[i],
										size=15)),
							  			secondary_y=False)

			elif single_time:
				# Adds the tagdata to the plot
				fig.add_trace(go.Scatter(x=data[0], y=data[i+1],
										 name=tagname[i],
										 marker=dict(color=colors[i],
										size=15)),
							  			secondary_y=False)


			"""Removes unwanted tagnames from the left y-axis"""
			if remove_tags_from_taglist(calculation_suffixes, tagname[i]) != None:
				first_axis += remove_tags_from_taglist(calculation_suffixes, tagname[i]) + ", "
				print("LHA")

	#fig.update_layout(yaxis_title=unit)
	"""Removes the last space and comma from the axis text"""
	first_axis = first_axis[:-2]
	second_axis = second_axis[:-2]
	"""Updates axes"""
	fig.update_xaxes(title_text='Time')
	fig.update_yaxes(title_text=first_axis, secondary_y=False)
	fig.update_yaxes(title_text=second_axis, secondary_y=True)
	#fig.update_layout(coloraxis=dict(colorscale='Viridis'))
	#colorscales = py.colors.named_colorscales()
	"""Creates a html file of the plot"""
	py.offline.plot(fig, filename=filename+'.html')
	#fig.show()

def plot_histogram(filename, tagname, data):
	pass

def hent_tag(tagnames):
	"""
	Gets data to be plotted for tagnames

	:param tagnames: list of string, name of tags
	:return: Data in format [[Time], [values], [Time], [values]...]
	"""
	matrix = []
	tags = []
	for i in range(len(tagnames)):
		tags.append(Tag(tagnames[i]))
		tags[i].get_measurement()
		matrix.append(np.asarray(tags[i].timestamp))
		matrix.append(np.asarray(tags[i].measurements))
		print(len(matrix), len(tags[i].timestamp))

	return matrix

def transform_to_one_time(matrix):
	"""
	Removes timestamps for all data sets except for first.

	:param matrix: list of lists, data on format [[Time], [values], [time], [values]]
	:return: new matrix with only on time column
	"""
	new_matrix = []
	new_matrix.append(matrix[0])
	#new_matrix.append([i for i in matrix if i % 2 != 0])
	for i in range(int((len(matrix)+1))):
		if i % 2 != 0:
			new_matrix.append((matrix[i]))
	return new_matrix


def find_longest_subarray(array):
	"""
	Finds longest array in a matrix

	:param array: Matrix of any data
	:return: integer number of the longest subarray
	"""
	max = 0
	for i in range(len(array)):
		if len(array[i]) > max:
			max = len(array[i])
	print(max)
	return max

def transpose_undefined_matrix(matrix):
	"""
	Transposes a matrix

	:param matrix: Matrix to be transposed
	:return: Transposed matrix
	"""
	max = find_longest_subarray(matrix)
	data = []
	temp = []
	for i in range(max):
		temp.clear()
		for j in range(len(matrix)):

			try:
				temp.append(matrix[j][i])
				# print(tag_u_tid[j][i])
				# print('matriseplass ', j,',',i)
			except IndexError:
				temp.append('')

		data.append(temp.copy())
	return data

def insert_value_every_x_to_list(list, value, x):
	"""
	Inserts value every xth element in list

	:param list: list, to be changed
	:param value: value which should be inserted
	:param x: int, Interval of insertion
	:return:
	"""
	for i in range(len(list)):
		list.insert(i * x, value)
	return list

def get_tags_from_station_to_csv(stationname, pw):
	"""
	deprecated, gets tags from station in database
	:param stationname: string, name of station
	:return:
	"""

	sql = psql("dmf_prosess", "localhost", "postgres")
	sql.connect(pw)
	tagnames = list(sql.get_tagnames_from_station(stationname))

	sql.disconnect()
	tags = hent_tag(tagnames)

	#tag_u_tid = transform_to_one_time(tags)

	#max = find_longest_subarray(tags)

	data = transpose_undefined_matrix(tags)

	tagnames = insert_value_every_x_to_list(tagnames, 'Tid', 2)

	#ArrayTocsv(tagnames, data, stationname+'.csv')

def moving_average(data, n=120):
	"""
	Calculates the moving average of a data array

	:param data: Data to be calculated
	:param n: number of datapoints to average over
	:return: Calculated moving average
	"""

	ret = np.cumsum(data, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n


def low_pass_filter(init, T_f, ts, values):
	"""
	Removes high frequency noise

	:param init: float, Initial value of the dataset, usually the first datapoint
	:param T_f: Parameter, higher values yields higher degree of filtering
	:param ts: float, timestep
	:param values: List of float, values to be filtered
	:return:
	"""
	"""Calculates low pass filter for a data array"""
	a = ts/(T_f + ts)
	filtered_val = np.zeros(len(values))
	filtered_val[0] = init
	for i in range(1,len(values)):
		filtered_val[i] = (1-a)*filtered_val[i-1]+a*values[i]
		#print(filtered_val[i], values[i])
	return filtered_val

def integrate_meas(time, meas, dt):
	"""
	Intergrates the whole data set

	:param time: List of timestamps
	:param meas: list of float, data to be integrated
	:param dt: float, timestep of integration
	:return: List of cumulative values
	"""
	sum = np.zeros([len(meas)])
	sum[0] = meas[0] * 0.5
	for i in range(1, len(meas)-1):

		if (time[i].date()) > (time[i-1].date()):
			sum[i-1] = sum[i-2] + meas[i-1]*0.5
			sum[i] = meas[i] * 0.5
		elif (time[i].date()) == (time[i-1].date()):
			sum[i] = sum[i-1] + meas[i]

	sum[len(meas)-1] = sum[len(meas)-2]+meas[len(meas)-1]*0.5
	sum *= (1/dt)

	return sum

def integrate_meas_extract_daily(time, meas, dt):
	"""
	Integrates data set per day

	:param time: List of timestamps
	:param meas: list of float, data to be integrated
	:param dt: float, timestep of integration
	:return: List of cumulative values per day
	"""
	sum = np.zeros([len(meas)])
	no_days = 0
	for i in range(len(meas) - 1):
		if (time[i].date()) > (time[i - 1].date()):
			no_days += 1
	new_time = []
	daily_top = np.zeros([no_days+1])
	sum[0] = meas[0] * 0.5
	no_days = 0
	for i in range(1, len(meas)-1):

		if (time[i].date()) > (time[i-1].date()):
			sum[i-1] = sum[i-2] + meas[i-1]*0.5
			sum[i] = meas[i] * 0.5
			new_time.append(time[i-1].date())
			daily_top[no_days] = (sum[i-1])
			no_days += 1
		elif (time[i].date()) == (time[i-1].date()):
			sum[i] = sum[i-1] + meas[i]

	sum[len(meas)-1] = sum[len(meas)-2]+meas[len(meas)-1]*0.5
	daily_top[-1] = sum[len(meas)-1]
	daily_top *= (1/dt)
	sum *= (1/dt)
	new_time.append(time[len(time)-1].date())

	return daily_top, new_time

def average_daily(time, meas):
	"""
	Creates an average per day for the given dataset

	:param time: list of datetime
	:param meas: list of float, values to be averaged
	:return: Average per day and belonging dates
	"""
	sum = np.zeros([len(meas)])
	no_days = 0
	for i in range(len(meas) - 1):
		if (time[i].date()) > (time[i - 1].date()):
			no_days += 1
	new_time = []
	daily_avg = np.zeros([no_days + 1])
	#sum[0] = meas[0] * 0.5
	no_days = 0
	start_index = 0
	stop_index = 0
	for i in range(1, len(meas) - 1):

		if (time[i].date()) > (time[i - 1].date()):

			new_time.append(time[i - 1].date())
			daily_avg[no_days] = (sum[i - 1])/len(sum[start_index:stop_index])
			no_days += 1
			start_index = i
		elif (time[i].date()) == (time[i - 1].date()):
			sum[i] = sum[i - 1] + meas[i]
			stop_index = i

	return daily_avg, new_time

def integrate_total_flow_RT():
	"""
	Integrates flow measurements from RTs

	:return:
	"""
	tagnames =['GSB_FT07', 'GSB_FT08', 'ANU_FT312']
	tags = []
	values = []


	for i in range(len(tagnames)):
		tags.append(Tag(tagnames[i]))
		tags[i].get_measurement()
		#plot([tags[i].tag], [tags[i].timestamp, tags[i].measurements])
		values.append(np.asarray(integrate_meas(tags[i].measurements,0.5)))

	total = np.zeros(len(values[0]))
	for i in range(len(values)):

		total = np.add(total, values[i])
		print(total[0])

	print(total)

def differentiate_meas(meas, dt):
	"""
	finds the rate of change between each datapoint in the dataset

	:param meas: List of float, measurements to be differentiated
	:param dt: float, timestep for the differentiation
	:return: List of differentiated measurements
	"""
	sum = np.zeros([len(meas)])
	sum[0] = (0)
	for i in range(1,len(meas)):
		sum[i] = (meas[i]-meas[i-1])*dt
	return sum

if __name__=='__main__':

	"""Forskjell på måleintervall overrasker"""

	"""
	csvnames = [ "AFM_FT0325-29. Feb",  "AFM_FT0425-29. Feb",
				"FOR_FT0225-29. Feb", "FOR_FT0525-29. Feb",
				"FOR_FT0825-29. Feb", "FOR_FT0925-29. Feb",
				"FOR_FT1825-29. Feb", "FOR_FT1925-29. Feb",
				"FOR_HX11_Hastighet25-29. Feb",
				"FOR_HX11_Moment25-29. Feb", "FOR_HX11_Pådrag25-29. Feb",
				"FOR_HX11_Strøm25-29. Feb",
				"FOR_PU07_Hastighet25-29. Feb", "FOR_PU07_Moment25-29. Feb",
				"FOR_PU07_Pådrag25-29. Feb", "FOR_PU07_Strøm25-29. Feb", "HYG_FT0125-29. Feb",
				"HYG_LT0125-29. Feb", "HYG_LT30125-29. Feb"]
	"""
	tag_description = ["Innløpspumpe 1 til hygeniseringstanker", "Innløpspumpe 1.5 til hygeniseringstanker",
					   "Innløpspumpe 2.5 til hygeniseringstanker", "Innløpspumpe 2 til hygeniseringstanker"]
	csvnames = ["HYG_PU12", "HYG_PU13", "HYG_PU14", "HYG_PU15"]
	"""
	csvnames2 = ["AFM_FT0301-09. Mars","AFM_FT0401-09. Mars","FOR_FT0201-09. Mars", "FOR_FT0501-09. Mars",
				 "FOR_FT0801-09. Mars", "FOR_FT0901-09. Mars", "FOR_FT1801-09. Mars", "FOR_FT1901-09. Mars",
				 "FOR_HX11_Hastighet01-09. Mars", "FOR_HX11_Moment01-09. Mars", "FOR_HX11_Pådrag01-09. Mars",
				 "FOR_HX11_Strøm01-09. Mars", "FOR_PU07_Hastighet01-09. Mars", "FOR_PU07_Moment01-09. Mars",
				 "FOR_PU07_Pådrag01-09. Mars", "FOR_PU07_Strøm01-09. Mars", "HYG_FT0101-09. Mars", "HYG_LT0101-09. Mars",
				 "HYG_LT30101-09. Mars"]
				 """
	"""Flowmåling gjennom fabrikken"""
	"""
	csvnames = ["FOR_FT02", "AFM_FT03", "AFM_FT04", "HYG_FT01"]
	tag_description = ["Mengde etter pulper", "Mengde fra mottakstank 1",
					   "Mengde fra mottakstank 2", "Innløp til hygenisering"]
	plot(csvnames, tag_description,"uvisst")
	for i in range(0,1000,1):
		print(i)
	csvnames = [ "ANU_FT09", "ANU_FT10",  "ANU_FT311"]
	tag_description = ["Flytende mengde ut fra RT1", "Flytende mengde ut fra RT2",
						"Flytende mengde ut fra RT3"]
	plot(csvnames, tag_description, "m3/h")
	
	csvnames = ["GSB_FT07", "GSB_FT08","ANU_FT312"]
	tag_description = ["Gass fra RT1",
						"Gass fra RT2", "Gass fra RT3"]
						"""
	"""Test fra MSSQL Guard

	csvnames = ["FOR_FT02 20.02-09.03","FOR_FT08 20.02-09.03", "FOR_FT09 20.02-09.03", "FOR_FT18 20.02-09.03",
				"FOR_FT19 20.02-09.03", "GSB_FT07 20.02-09.03", "GSB_FT08 20.02-09.03", "HYG_FT01 20.02-09.03",
				"HYG_LT01 20.02-09.03"]
"""
	#plot(csvnames, csvnames, "Guard data")
	def HYG_FT01_moving_avg():
		plot_sql(['HYG_FT01_filter','HYG_FT01'], ["""SELECT measurement.meas_timestamp,
				AVG(measurement.meas_value)
				OVER(ORDER BY measurement.meas_timestamp desc ROWS BETWEEN 2880 PRECEDING AND CURRENT ROW) AS avg_value
				FROM measurement
				where measurement.tag_id = (select tag.tag_id from tag where tag.tag = 'HYG_FT01')
				and measurement.meas_timestamp > '2020-03-01';""",
				"""select meas_timestamp, meas_value from measurement
					where measurement.tag_id = (select tag.tag_id from tag where tag.tag = 'HYG_FT01') and meas_timestamp > '2020-03-01'
					order by meas_timestamp desc"""])
	def HYG_FT01_unfiltered():
		plot_sql(['HYG_FT01'], ["""select meas_timestamp, meas_value from measurement
					where measurement.tag_id = (select tag.tag_id from tag where tag.tag = 'HYG_FT01') and meas_timestamp > '2020-03-01'
					order by meas_timestamp desc"""])

	def HYG_FT01_lpf():
		sql = psql("dmf_prosess", "localhost", "postgres")
		sql.connect('01or5gofark')
		query = """select meas_timestamp, meas_value from measurement
					where measurement.tag_id = (select tag.tag_id from tag where tag.tag = 'Kran_Akk_Vekt_Døgn') and 
					meas_timestamp >= '2020-03-20' 
					order by meas_timestamp asc"""
		rows = sql.q_select(query)
		sql.disconnect()
		data = sql.transpose_sql_query(rows)
		print(data)
		#Converts to m3/h
		data[1] = np.asarray(data[1])*0.06

		#Lowpass filter
		val_lpf = low_pass_filter(data[1][0], 60, 30, data[1])
		data.append(val_lpf)
		return data

def remove_tags_from_taglist(calculation_suffixes, tagname):
	"""
	Removes tags in calculation suffixes from the tagname list

	:param calculation_suffixes: List of string, entries in tagname which consist of any of these will not be part of output
	:param tagname: List of strings, tagnames which should be filtered
	:return: list of filtered strings
	"""
	num_label_included = 0
	result = ''
	for k in range(len(calculation_suffixes)):
		if calculation_suffixes[k] in tagname:
			pass
		else:
			num_label_included += 1

		if num_label_included == len(calculation_suffixes):
			unit_index = tagname.find("[")
			label_tag = tagname[:unit_index]
			result = label_tag

	if result != '':
		return result

	""" Malmberg CSV
	temp = csvToArray("Inkommande_FlödeT1.csv")
	timestamp = []
	GassInn1 = []
	GassUt1 = []
	CH4Inn1 = []
	CH4Ut1 = []
	GassInn2 = []
	GassUt2 = []
	CH4Inn2 = []
	CH4Ut2 = []
	# print(temp)
	for row in temp:
		# print(len(row))
		#print(row)
		try:
			timestamp.append(row[0])
			GassInn1.append(float(row[1]))
			GassUt1.append(float(row[2]))
			CH4Inn1.append(float(row[3]))
			CH4Ut1.append(float(row[4]))
			GassInn2.append(float(row[5]))
			GassUt2.append(float(row[6]))
			CH4Inn2.append(float(row[7]))
			CH4Ut2.append(float(row[8]))
		except ValueError :
			pass

	data = [timestamp, GassInn1, timestamp, GassUt2, timestamp, CH4Inn1, timestamp, CH4Ut1,
			timestamp, GassInn2, timestamp, GassUt2, timestamp, CH4Inn2, timestamp, CH4Ut2]
	length = len(data)
	for i in range(int((length +1)/ 2)):
		data.append(timestamp)
		data.append(low_pass_filter(data[i*2+1][0], 60, 30, data[i*2+1]))
		data.append(timestamp)
		data.append(moving_average(data[i*2+1], 120))
	

	plot("Malmberg", ["Gas_flow_Plant_in",
					  "FT10701 Gass ut",
					  "QT60001 CH4Inn",
					  "QT65001 CH4 Ut",
					  "Gas_flow_Plant_in_T2",
					  "FT10701 Gass ut_T2",
					  "QT60001 CH4Inn_T2",
					  "QT65001 CH4 Ut_T2", "Gas_flow_Plant_in_LPF", "Gas_flow_Plant_in_MA",
					  "FT10701 Gass ut_LPF", "FT10701 Gass ut_MA","QT60001 CH4Inn:LPF","QT60001 CH4Inn_MA",
					  "QT65001 CH4 Ut_LPF", "QT65001 CH4 Ut_MA", "Gas_flow_Plant_in_T2_LPF", "Gas_flow_Plant_in_T2_MA",
					  "FT10701 Gass ut_T2_LPF", "FT10701 Gass ut_T2_MA","QT60001 CH4Inn_T2_LPF", "QT60001 CH4Inn_T2_MA",

					  "QT65001 CH4 Ut_T2_LPF", "QT65001 CH4 Ut_T2_MA"], data)
	"""

	"""Henter tags for HYG_PU12 til 19"""
	#HYG_PU12_19()
	"""Henter alle tags på valgt stasjon"""
	#get_tags_from_station_to_csv('Hygenisering')

if __name__ == '__main__':
	csvnames = ["HYG_PU18Strøm"]

	#plot(csvnames2, csvnames2, "Mars")
	#HYG_FT01_lpf()

	#data = ukentlig_rapport_tagbasert()
	#print(len(data))
	"""tagnames = ['HYG_FT01', 'HYG_FT01_lpf', 'HYG_FT01_avg',
					 'HYG_FT02', 'HYG_FT02_lpf', 'HYG_FT02_avg',
		  'GSB_FT07', 'GSB_FT07_lpf', 'GSB_FT07_avg',
		  'GSB_FT08', 'GSB_FT08_lpf',  'GSB_FT08_avg',
		  'ANU_FT312', 'ANU_FT312_lpf', 'ANU_FT312_avg',
		  'GSB_FT06', 'GSB_FT06_lpf', 'GSB_FT06_avg']"""
	"""tagnames = ["GSB_FT06", "T1_Rågass", "T2_Rågass", "T1_FT10701", "T2_FT10701",
				"T1_QT60001", "T2_QT60001", "T1_QT65001", "T2_QT65001"]
	#data, tagnames = lag_matrise_tags(tagnames)
#    for row in data:
#    if data[3] == data[6]:
	get_tags_from_station_to_csv('Forbehandling')"""
	#temp = np.add(data[2], data[4])

	#tagnames.append('Sum rådata')
	#data.append(temp)


	#plot("testrapport", tagnames, data)
	#plotarrayTocsv(tagnames, data, 'Flow hyg-gassoppgradering 04.-30.03.20.csv')
	#data = HYG_FT01_lpf()
	#data.append(integrate_meas(data[1],0.5))
	#plot(['Kran_Akk_Vekt_Døgn'], data)

	# HYG_FT01_moving_avg()


	#plot_csv("Inkommande_FlödeT1", "Inkommande_FlödeT2")
	#print(len(temp), temp)
	#pd.read_sql
	"""tag = Tag('HYG_FT01')
	print(tag.timestamp)
	tag.get_measurement()
	for i in range(len(tag.timestamp)):
		print(tag.timestamp, tag.measurements)"""

	#integrate_total_flow_RT()


