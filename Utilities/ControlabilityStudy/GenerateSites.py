from HyperNav.Utilities.ControlabilityStudy.ControlBase import ControlHawaii, ControlSoCal, ControlMonterey
import geopy
import os
import random


def generate_hawaii():
	deployed_point = geopy.Point(19.5,-156.4)
	run = 0
	while run < 99: 
		for run in range(100):
			filename = ControlHawaii().make_pickle_filename(run,0)
			if not os.path.exists(filename):
				break
		start_time = random.choice(ControlHawaii.uv_class.dataset_time[:-300])
		start_time_list,start_point_list,drift_depth_list = ControlHawaii().maintain_location(deployed_point,start_time,run)

def generate_socal():
	deployed_point = geopy.Point(33.7,-119.6)
	run = 0
	while run < 99: 
		for run in range(100):
			filename = ControlSoCal().make_pickle_filename(run,0)
			if not os.path.exists(filename):
				break
		start_time = random.choice(ControlSoCal.uv_class.dataset_time[:-300])
		start_time_list,start_point_list,drift_depth_list = ControlSoCal().maintain_location(deployed_point,start_time,run)

def generate_monterey():
	deployed_point = geopy.Point(36.7,-122.2)
	run = 0
	while run < 99: 
		for run in range(100):
			filename = ControlMonterey().make_pickle_filename(run,0)
			if not os.path.exists(filename):
				break
		start_time = random.choice(ControlSoCal.uv_class.dataset_time[:-300])
		start_time_list,start_point_list,drift_depth_list = ControlMonterey().maintain_location(deployed_point,start_time,run)
