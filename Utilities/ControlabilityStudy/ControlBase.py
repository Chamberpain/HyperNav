from HyperNav.Utilities.Data.HYCOM import HYCOMMonterey,HYCOMSouthernCalifornia,HYCOMHawaii,HYCOMPuertoRico
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
import gc
import geopy
import datetime
from copy import deepcopy
from HyperNav.Utilities.Compute.RunParcels import create_prediction,ParticleList,ParticleDataset
import os
import pickle
import shutil

class ControlParticleList(ParticleList):

	def __init__(self,*args,profile,run,**kwargs):
		super().__init__(*args,**kwargs)

class ControlBase():

	def make_zarr_filename(self,run,profile_num,days,drift_depth):
		filename = '{}_run_{}_profile_{}_days_{}_drift'.format(run,profile_num,days,drift_depth)
		return self.file_handler.tmp_file(filename+'.zarr')

	def make_pickle_filename(self,run,profile_num):
		filename = '{}_run_{}_profile'.format(run,profile_num)
		return self.file_handler.tmp_file(filename+'.pickle')

	def make_prediction(self,start_time,start_point,profile_num,run):
		uv_instance = self.uv_class.load(start_time-datetime.timedelta(days=2),start_time+datetime.timedelta(days=7))
		surface_time = 5400
		vertical_speed = 0.076
		for days in range(1,6):
			data,dimensions = uv_instance.return_parcels_uv(start_time-datetime.timedelta(days=1),start_time+datetime.timedelta(days=7))
			date_end = start_time + datetime.timedelta(days=days)
			total_cycle_time = 3600*24*days
			for drift_depth in [50,100,200,300,400,500,600,700]:
				filename = self.make_zarr_filename(run,profile_num,days,drift_depth)
				if not os.path.isfile(filename):
					argo_cfg = {
					'lat': start_point.latitude, 'lon': start_point.longitude,
					'time': start_time.timestamp(), 'end_time': date_end.timestamp(), 'depth': 10, 'min_depth': 10, 
					'drift_depth': abs(drift_depth),'max_depth': abs(drift_depth),
					'surface_time': surface_time, 'total_cycle_time': total_cycle_time,
								'vertical_speed': vertical_speed,
								}
					prediction = create_prediction(argo_cfg,data,dimensions,filename)
					gc.collect(generation=2)

	def make_pickle(self,profile,run,start_time,start_point):
		self.make_prediction(start_time,start_point,profile,run)
		pl = ParticleList()
		for days in range(1,6):
			for drift_depth in [50,100,200,300,400,500,600,700]:
				filename = self.make_zarr_filename(run,profile,days,drift_depth)
				nc = ParticleDataset(filename)
				pl.append(nc)
		start_time,start_point,drift_depth = pl.closest_to_point(deployed_point)
		pickle_name = self.make_pickle_filename(run,profile)
		file = open(pickle_name, 'wb')
		pickle.dump([start_time,start_point,drift_depth],file)
		for days in range(1,6):
			for drift_depth in [50,100,200,300,400,500,600,700]:
				filename = self.make_zarr_filename(run,profile,days,drift_depth)
				try:
					shutil.rmtree(filename)
				except FileNotFoundError:
					continue
		return

	# def find_closest(self,pl,deployed_point)


	# 	return (new_time,new_point)

	def maintain_location(self,deployed_point,start_time,run):
		start_time_list = []
		start_point_list = []
		drift_depth_list = []
		start_point = deepcopy(deployed_point)
		profile = 0 
		while profile<60:
			pickle_name = self.make_pickle_filename(run,profile)
			try:
				file = open(pickle_name, 'rb')
				data = pickle.load(file)
				start_time,start_point,drift_depth = data
				file.close()
			except FileNotFoundError:
				pl = self.make_pickle(profile,run,start_time,start_point)
				continue
			print('new start point is ')
			print(start_point)
			print('new start time is ')
			print(start_time)
			print('drift depth is ')
			print(drift_depth)
			start_time_list.append(start_time)
			start_point_list.append(start_point)
			drift_depth_list.append(drift_depth)
			profile+=1
		return (start_time_list,start_point_list,drift_depth_list)


class ControlMonterey(ControlBase):
	uv_class = HYCOMMonterey
	file_handler = FilePathHandler(ROOT_DIR,'MontereyControl')
	# deployed_point = geopy.Point(36.7,-122.2)
	# start_time = datetime.datetime(2018,1,3)


class ControlSoCal(ControlBase):
	uv_class = HYCOMSouthernCalifornia
	file_handler = FilePathHandler(ROOT_DIR,'SoCalControl')
deployed_point = geopy.Point(33.7,-119.6)
start_time = datetime.datetime(2020,12,1)

class ControlHawaii(ControlBase):
	uv_class = HYCOMHawaii
	file_handler = FilePathHandler(ROOT_DIR,'HawaiiControl')
deployed_point = geopy.Point(19.5,-156.4)
start_time = datetime.datetime(2018,1,3)


class ControlPR(ControlBase):
	uv_class = HYCOMPuertoRico
	file_handler = FilePathHandler(ROOT_DIR,'PRControl')
