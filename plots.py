import math
import matplotlib.pyplot as plt
import plotly
from plotly.graph_objs import Font
from plotly.subplots import make_subplots
import plotly as py
import plotly.graph_objects as go
import math
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly

from array_manipulation import get_generic_column_names_for_plot


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
	rename_dict = get_generic_column_names_for_plot(data)
	data = data.rename(columns=rename_dict)
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
				fig.add_trace(go.Scatter(x=data[i * 2], y=data[tagname[i+1]],
										 name=tagname[i+1],
										marker=dict(color=colors[i],
										size=15)),
							  			secondary_y=False)

			elif single_time:
				# Adds the tagdata to the plot
				fig.add_trace(go.Scatter(x=data['Time'], y=data[data.columns[i+1]],
										 name=data.columns[i+1],
										 marker=dict(color=colors[i],
										size=15)),
							  			secondary_y=False)


			"""Removes unwanted tagnames from the left y-axis"""
			if remove_tags_from_taglist(calculation_suffixes, tagname[i]) != None:
				first_axis += remove_tags_from_taglist(calculation_suffixes, tagname[i]) + ", "
				print("LHA")
	try:
		fig.add_trace(go.Scatter(x=data['Time'], y=data['State'],
								 name='State',
								 marker=dict(color=colors[i],
											 size=15)),
					  secondary_y=False)
	except KeyError as e:
		print(e)
	#fig.update_layout(yaxis_title=unit)
	"""Removes the last space and comma from the axis text"""
	first_axis = first_axis[:-2]
	second_axis = second_axis[:-2]
	"""Updates axes"""
	fig.update_xaxes(title_text='Time', title_font_size=18)
	fig.update_yaxes(title_text=first_axis, secondary_y=False, title_font_size=18)
	fig.update_yaxes(title_text=second_axis, secondary_y=True, title_font_size=18)
	fig.update_layout(
		title_font_size=18,
		font=dict(size=18)
	)
	fig.update_layout(legend=dict(
		yanchor="top",
		y=0.99,
		xanchor="left",
		x=0.01
	))
	fig.update_layout(legend_font_size=18)
	#fig.update_layout(coloraxis=dict(colorscale='Viridis'))
	#colorscales = py.colors.named_colorscales()
	"""Creates a html file of the plot"""
	print('creating plot at filename')
	py.offline.plot(fig, filename=filename+'.html', auto_open=False)
	#fig.show()

def plot_confusion_matrix(confusion_matrix, labels, filename):
	import plotly.figure_factory as ff

	x = labels
	y = labels

	# change each element of confusion_matrix to type string for annotations
	confusion_matrix_text = [[str(y) for y in x] for x in confusion_matrix]

	# set up figure
	fig = ff.create_annotated_heatmap(confusion_matrix, x=x, y=y, annotation_text=confusion_matrix_text,
									  colorscale='Viridis')

	# add title
	# fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
	# 				  # xaxis = dict(title='x'),
	# 				  # yaxis = dict(title='x')
	# 				  )
	fig.update_layout(autosize=False,
					  font=Font(
						  family="Gill Sans MT",
						  size=16),
					  width=720+250,
					  height=720+50
					  )
	# add custom xaxis title
	fig.add_annotation(dict(font=dict(color="black", size=16),
							x=0.5,
							y=-0.15,
							showarrow=False,
							text="Predicted value",
							xref="paper",
							yref="paper"))

	# add custom yaxis title
	fig.add_annotation(dict(font=dict(color="black", size=16),
							x=-0.35,
							y=0.5,
							showarrow=False,
							text="True value",
							textangle=-90,
							xref="paper",
							yref="paper"))

	# adjust margins to make room for yaxis title
	fig.update_layout(margin=dict(t=200, l=350))


	# add colorbar
	fig['data'][0]['showscale'] = True
	py.offline.plot(fig, filename=filename + '.html', auto_open=False)
	fig.show()


def histogram_ly(df, filename, drop_time=False):
	rename_dict = get_generic_column_names_for_plot(df)
	df = df.rename(columns=rename_dict)
	if drop_time:
		df.drop(columns=["Time"], axis=1, inplace=True)
	print(df.columns)
	"""if "PU19_State" in df.columns:
		colorIndex = df["PU19_State"]
		df.drop(columns=["PU19_State"], axis=1, inplace=True)"""
	columns = list(df.columns)
	columns.remove('State')
	size = len(columns)
	rows = math.ceil(math.sqrt(size))
	col = rows
	#print(rows, col, size)
	fig = make_subplots(rows,col)
	trace = []
	j = 1
	k = 1
	for i in range(size):
		layout = dict(xaxis=dict(title=df.columns[i]))
		trace.append(go.Histogram(x=df[columns[i]], nbinsx=500, name=columns[i]))
		# trace[i].update_xaxes(title_text=df[df.columns[i]].name)
		# trace[i].update_yaxes(title_text='Count')
		#trace[i].update(layout=layout)
		print(j, k)
		fig.append_trace(trace[i], j, k)

		print(type(fig))
		fig.update_xaxes(title_text=columns[i], row=j, col=k)
		fig.update_yaxes(title_text='Count', row=j, col=k)

		k += 1
		if k == col+1:
			j += 1
			k = 1

	fig.update_layout(
		title_font_size=18,
		font=dict(size=18)
	)

	plotly.offline.plot(fig, filename='./Pump_data_statistics/histogram ' + filename + '.html', auto_open=False)


def histogram(df, no_bins):

    print(df.dtypes)
    df.hist(bins=no_bins, figsize=(20,15))
    plt.show()
    plt.savefig("")
    #hist = df.hist(column='pu19_pw_pv')
    """for name in tagnames[1:]:
        hist = df.hist(column=name, bins=no_bins)
        plt.show()"""


def correlation_plots(df, filename):

	rename_dict = get_generic_column_names_for_plot(df)
	df = df.rename(columns=rename_dict)

	recolor = False
	#timeIndex = pd.to_datetime(df['meas_time']).astype(np.int64)
	"""if "PU19_State" in df.columns:
		colorIndex = df["PU19_State"]
		df.drop(columns=["PU19_State"], axis=1, inplace=True)
		recolor = True"""
	recolor = True
	stats = df.describe()
	stats.to_csv('./Pump_data_statistics/Statistics value ' + filename + '.csv', sep=',')
	#CSV.ArrayTocsv(df.columns, stats, 'Statistics value.csv')

	corr_matrix = df.corr()
	corr_matrix.to_csv('./Pump_data_statistics/Correlation matrix ' + filename + '.csv', sep=',')
	#CSV.ArrayTocsv(df.columns, corr_matrix, 'Correlation matrix.csv')
	#print(colorIndex.describe())
	print(stats)
	if recolor:
		print('State plot')
		fig = px.scatter_matrix(df, dimensions=df.columns[:-1], color="State")
	else:
		fig = px.scatter_matrix(df[1:])
	fig.update_layout(
		title_font_size=18,
		font=dict(size=18)
	)
	plotly.offline.plot(fig, filename='./Pump_data_statistics/Correlation ' + filename + '.html', auto_open=False)
	#scatter_matrix(df[1:], figsize=(12, 8))
	#plt.show()
	#plt.savefig("scatter_matrix_plot")
	#df = pd.concat([df, colorIndex], axis=1)
	#df["PU19_State"] = colorIndex.values
	#df = pd.merge_asof(df, colorIndex, on='meas_time')