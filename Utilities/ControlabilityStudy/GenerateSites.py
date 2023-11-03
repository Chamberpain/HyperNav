from HyperNav.Utilities.ControlabilityStudy.ControlBase import ControlHawaii, ControlSoCal, ControlMonterey, ControlPR, ControlCrete, ControlTahiti, ControlBermuda, ControlCanary
import geopy
import os
import random
import numpy as np
import datetime

def generate(start_point,start_class):
	run = 0
	while run < 99: 
		for run in range(100):
			filename = start_class.make_pickle_filename(run,0)
			if not os.path.exists(filename):
				break
		start_time = random.choice(start_class.uv_class.dataset_time[:-1600])
		idx = start_class.uv_class.dataset_time.index(start_time)
		diff_series = np.diff(start_class.uv_class.dataset_time[idx:idx+1600])
		if not all([x < start_class.uv_class.time_step+datetime.timedelta(days=3) for x in diff_series]):
			print('there was a gap in the time records of the dataset')
			print('starting at ')
			print(start_time)
			print('the gap was at ')
			print(np.where([x != diff_series[0] for x in diff_series]))
			continue
		print(idx,' this is a good one')
		holder = start_class(run,start_point,start_time)
		holder.maintain_location()

def generate_hawaii():
	start_point = geopy.Point(19.5,-156.4)
	start_class = ControlHawaii
	generate(start_point,start_class)

def generate_socal():
	start_point = geopy.Point(33.7,-119.6)
	start_class = ControlSoCal
	generate(start_point,start_class)

def generate_monterey():
	start_point = geopy.Point(36.7,-122.2)
	start_class = ControlMonterey
	generate(start_point,start_class)

def generate_PR():
	start_point = geopy.Point(17.8,-66.7,0)
	start_class = ControlPR
	generate(start_point,start_class)

def generate_crete():
	start_point = geopy.Point(35.75,25.0,0)
	start_class = ControlCrete
	generate(start_point,start_class)

def generate_tahiti():
	start_point = geopy.Point(-17.8,-149.75,0)
	start_class = ControlTahiti
	generate(start_point,start_class)

def generate_bermuda():
	start_point = geopy.Point(32,-64.5,0)
	start_class = ControlBermuda
	generate(start_point,start_class)

def generate_canary():
	start_point = geopy.Point(27.75,-16.5,0)
	start_class = ControlCanary
	generate(start_point,start_class)
